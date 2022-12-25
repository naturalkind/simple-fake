import os, sys, glob, time, json
from collections import defaultdict
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from dface import MTCNN, FaceNet
from concurrent.futures import ThreadPoolExecutor
import timm.models.efficientnet as effnet
from sklearn.cluster import DBSCAN
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast
from moviepy.editor import * 

import ctypes

import os, sys, io, gc, re
import glob, uuid, base64, json, time

from tornado.escape import json_encode

from tornado import websocket, web, ioloop

import tornado.ioloop
import tornado.web
import tornado.websocket

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
import faiss
from clickhouse_driver import Client

import keras
from keras.preprocessing import image as image_utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import tensorflow as tf


device = 'cuda'

margin = 0
scan_fps = 1
batch_size = 32
face_size = None

mtcnn = None
facenet = None
deepware = None

def imgs(x):
    cv2.imshow('Rotat', np.array(x))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class EffNet(nn.Module):
    def __init__(self, arch='b3'):
        super(EffNet, self).__init__()
        fc_size = {'b1':1280, 'b2':1408, 'b3':1536, 'b4':1792,
                   'b5':2048, 'b6':2304, 'b7':2560}
        assert arch in fc_size.keys()
        effnet_model = getattr(effnet, 'tf_efficientnet_%s_ns'%arch)
        self.encoder = effnet_model()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(fc_size[arch], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = []
        for i, model in enumerate(self.models):
            y = model(x)
            preds.append(y)
        final = torch.mean(torch.stack(preds), dim=0)
        return final


#import face_alignment
#face_detector = 'sfd'
#face_detector_kwargs = {
#    "filter_threshold" : 0.8
#}

# Run the 3D face alignment on a test image, without CUDA.
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device='cuda',
#                                  face_detector=face_detector)


#def get_frames(video, batch_size=10, target_fps=1):
#    vid = cv2.VideoCapture(video)
#    global total
#    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#    if total <= 0:
#        return None
#    global fps
#    fps = vid.get(cv2.CAP_PROP_FPS)
#    
#    if target_fps > fps:
#        target_fps = fps
#    nfrm = int(total/fps*target_fps)
#    idx = np.linspace(0, total, nfrm, endpoint=False, dtype=int)
#    batch = []
#    
#    global h, w
#    
#    for i in range(total):
#        ok = vid.grab()
#        ok, frm = vid.retrieve()
#        if i not in idx:
#            continue
#        if not ok:
#            continue
#        h, w = frm.shape[:2]
#        if w*h > 1920*1080:
#            scale = 1920/max(w, h)
#            frm = cv2.resize(frm, (int(w*scale), int(h*scale)))
#        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#        batch.append(frm)
#        if len(batch) == batch_size:
#            yield batch
#            batch = []
#    if len(batch) > 0:
#        yield batch
#    vid.release()

do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)



def gen_segments_melspec(X, window_size, overlap_sz):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_mels,n_samples)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    overlap_sz : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    
    window_step = (window_size-overlap_sz)
    append = np.zeros((window_size, (window_step - (X.shape[-1]-overlap_sz) % window_step)))
    X = np.hstack((X, append))
    #print (window_size, overlap_sz, window_size-overlap_sz, X.shape)
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided

def normlize(x):
    return ((x-np.min(x))/(np.max(x)-np.min(x)))


def chunks(lst, count):
    start = 0
    for i in range(count):
          stop = start + len(lst[i::count, :])
          yield lst[start:stop, :]
          start = stop 
           
def get_frames(video, batch_size=10, target_fps=1):
    clip = VideoFileClip(video)
    if abs(clip.rotation) in (90, 270):
        clip = clip.resize(clip.size[::-1])
        clip.rotation = 0
    w = clip.w
    h = clip.h
    
    global resolution
    resolution = clip.size
    
    global duration
    duration = clip.duration
    
    global channel_audio
    if clip.audio.nchannels == 1:
        channel_audio = "mono"
    elif clip.audio.nchannels == 2:
        channel_audio = "streo"
    
    global total
    total = clip.reader.nframes
    if total <= 0:
        return None
    global fps
    fps = clip.fps
    
    global fps_audio 
    fps_audio = clip.audio.fps
    
    global jpg_as_text
    jpg_as_text = []
    
    wavedata = clip.audio.to_soundarray(nbytes=4)
    
    W_size = 128
    utterance_melspec = librosa.feature.melspectrogram(y=wavedata[:,0], 
                                        sr=fps_audio, 
                                        n_fft=(int)(fps*fps_audio/1000), 
                                        hop_length=(int)((10)*fps_audio/1000), 
                                        n_mels=W_size, fmin=20, fmax=8000)
    segments_melspec = gen_segments_melspec(utterance_melspec, window_size=W_size, overlap_sz=30)


    for num in     range(0, segments_melspec.shape[0]):
        #print (num)
        static = librosa.power_to_db(segments_melspec[num], ref=np.max)
        delta = librosa.feature.delta(static, order=1)
        delta2 = librosa.feature.delta(static, order=2)

        static = normlize(static)*255
        delta = normlize(delta)*255
        delta2 = normlize(delta2)*255

        images = np.dstack((static,delta,delta2))
        #images = cv2.resize(images, (images.shape[0]*2, images.shape[1]*2))
        images = cv2.resize(images, (images.shape[0], images.shape[1]))
        
        _, img_str = cv2.imencode('.jpg', images.astype(np.uint8))
        
        jpg_as_text.append(base64.b64encode(img_str).decode())    
        
        #print (images.shape)
#        cv2.imshow("img", images/255.)
#        cv2.waitKey(1) 
#        time.sleep(0.025)



    
    if target_fps > fps:
        target_fps = fps
    batch = []
    #frames = clip.iter_frames()
    for i in range(int(duration)):
        frm = clip.get_frame(i)
        frm = cv2.resize(frm, (w, h))
        #imgs(frm)
        if w*h > 1920*1080:
            scale = 1920/max(w, h)
            frm = cv2.resize(frm, (int(w*scale), int(h*scale)))
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        batch.append(frm)    
        if len(batch) == batch_size:
            yield batch
            batch = []        
    if len(batch) > 0:
        yield batch                
#    clip.close()
#    clip.__del__()
    
def crop_face(img, box, margin=1):
    x1, y1, x2, y2 = box
    size = int(max(x2-x1, y2-y1) * margin)
    center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
    x1, x2 = center_x-size//2, center_x+size//2
    y1, y2 = center_y-size//2, center_y+size//2
    face = Image.fromarray(img).crop([x1, y1, x2, y2])
    return np.asarray(face)


def fix_margins(faces, new_margin):
    fixed = []
    for face in faces:
        img = Image.fromarray(face)
        w, h = img.size
        sz = int(w/margin*new_margin)
        img = TF.center_crop(img, (sz, sz))
        
        fixed.append(np.asarray(img))
    return fixed


def cluster(faces):
    if margin != 1.2:
        faces = fix_margins(faces, 1.2)

    embeds = facenet.embedding(faces)

    dbscan = DBSCAN(eps=0.35, metric='cosine', min_samples=scan_fps*5)
    labels = dbscan.fit_predict(embeds)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    bad = {0: clusters.pop(-1, [])}
    if len(clusters) == 0 and len(bad[0]) >= scan_fps*5:
        return bad
    return clusters, embeds


def id_strategy(pred, t=0.8):
    pred = np.array(pred)
    fake = pred[pred >= t]
    real = pred[pred <= (1-t)]
    if len(fake) >= int(len(pred)*0.9):
        return np.mean(fake)
    if len(real) >= int(len(pred)*0.9):
        return np.mean(real)
    return np.mean(pred)


confident = lambda p: np.mean(np.abs(p-0.5)*2) >= 0.7
label_spread = lambda x: x-np.log10(x) if x >= 0.8 else x


def strategy(preds):
    #
    # If there is a fake id and we're confident,
    # return spreaded fake score, otherwise return
    # the original fake score.
    # If everyone is real and we're confident return
    # the minimum real score, otherwise return the
    # mean of all predictions.
    #
    preds = np.array(preds)
    p_max = np.max(preds)
    if p_max >= 0.8:
        if confident(preds):
            return label_spread(p_max)
        return p_max
    if confident(preds):
        return np.min(preds)
    return np.mean(preds)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])


def scan(file):
    frames = get_frames(file, batch_size, scan_fps)
    faces, preds = [], []
    face_cord = []
    all_frames.clear()
    all_preds.clear()
    for batch in frames:
        results = mtcnn.detect(batch)
        for i, res in enumerate(results):
            if res is None:
                continue
            boxes, probs, lands = res
            for j, box in enumerate(boxes):
                if probs[j] > 0.98:
                    face = crop_face(batch[i], box, margin)
                    face_cord.append(box)
#                    all_frames.append(box.astype(int).tolist())
                    all_frames.append(box)
                    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
                    face = cv2.resize(face, face_size)
                    faces.append(face)
    if len(faces) == 0:
        return None, []

    with torch.no_grad():
        n = batch_size
        splitted_faces = int(np.ceil(len(faces)/n))

        for i in range(splitted_faces):
            faces_proc = []
            #print (i*n, (i+1)*n, splitted_faces)
            for face in faces[i*n:(i+1)*n]:
                face = preprocess(face)
                faces_proc.append(face)

            x = torch.stack(faces_proc)
            with autocast():
                y = deepware(x.to(device))
            preds.append(y)

    preds = torch.sigmoid(torch.cat(preds, dim=0))[:,0].cpu().numpy()
    tidx = 0
    lnf = 0
    #out = cv2.VideoWriter(f'videos/{file}_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))#(h,w)
    #print (len(all_frames), fps)
    new_list = all_frames[:]
    temp_a = []
    for ivd in range(len(all_frames)):
        try:
            a = all_frames[ivd]
            b = all_frames[ivd+1]
            Z = b-a
            temp_a.append(Z/int(fps))
        except:
            temp_a.append(np.array([0,0,0,0]))

    all_frames.clear()
    #print (lnf, total, len(new_list), len(temp_a))
    Az = len(new_list)
    for ivd in range(total):
        if lnf < Az :
            new_list[lnf] = new_list[lnf] + temp_a[lnf]
            all_preds.append(str(preds[lnf]))     
            all_frames.append(new_list[lnf].astype(int).tolist())
            if tidx == int(fps):
                tidx = 0
                lnf += 1
            tidx += 1 
                   
    #all_frames.clear()
    
    
    #print (preds, len(preds), fps, (w,h))
    return list(preds), faces
#[768, 398, 895, 569], [784, 394, 920, 563]

def similarity(vector1, vector2):
    return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True), np.linalg.norm(vector2.T, axis=0, keepdims=True))
from sklearn.metrics.pairwise import cosine_similarity
def process(file):
    try:
        preds, faces = scan(file)
        if preds is None:
            return 0.01

        clust, emb = cluster(faces)
        if len(clust) == 0:
            return 0.5

        id_preds = defaultdict(list)

        for label, indices in clust.items():
            for idx in indices:
                id_preds[label].append(preds[idx])

        preds = [id_strategy(preds) for preds in id_preds.values()]
        if len(preds) == 0:
            return 0.5

        score = strategy(preds)
        
        VVV = np.average(emb, axis=0) 
#        N = emb[0:1,:]
#        SZ = similarity(VVV.reshape([1,VVV.shape[0]]), N)
#        print ("embeds", emb.shape, SZ)
        
        return np.clip(score, 0.01, 0.99), faces, VVV

    except Exception as e:
        print(e, file, file=sys.stderr)
        return 0.5


def init(models_dir, cfg_file, dev):
    print ("INIT", models_dir)
    global device, mtcnn, facenet, deepware, margin, face_size, all_frames, all_preds
    all_frames = []
    all_preds = []
    with open(cfg_file) as f:
        cfg = json.loads(f.read())

    arch = cfg['arch']
    margin = cfg['margin']
    face_size = (cfg['size'], cfg['size'])
    dev = 'cuda'
    print(f'margin: {margin}, size: {face_size}, device: {dev}, arch: {arch}')

    device = dev
    mtcnn = MTCNN(device)
    facenet = FaceNet(device)

    if os.path.isdir(models_dir):
        model_paths = glob.glob('%s/*.pt'%models_dir)
    else:
        model_paths = [models_dir]
    model_list = []
    assert len(model_paths) >= 1
    print('loading %d models...'%len(model_paths))

    for model_path in model_paths:
        b3_model = EffNet(arch)
        checkpoint = torch.load(model_path, map_location="cpu")
        b3_model.load_state_dict(checkpoint)
        del checkpoint
        model_list.append(b3_model)
    
    deepware = Ensemble(model_list).eval().to(device)




class DataBase():
    def __init__(self):
        self.client = Client('localhost', settings = { 'use_numpy' : True })
        
    def delete(self, x):
        self.client.execute(f'DROP TABLE IF EXISTS {x}')    
    def show_count_tables(self, x):
        start = time.time()
        LS = self.client.execute(f"SELECT count() FROM {x}")
        print (time.time()-start, LS)
        return LS
    def get_all_data(self, x):
        start = time.time()
        LS = self.client.execute(f"SELECT * FROM {x}")
        print (time.time()-start, len(LS)) 
        return LS


def deep_vector(x):
       t_arr = np.expand_dims(x, axis=0)
       processed_img = preprocess(t_arr)
       preds = model.predict(processed_img)
       return preds

def similarity(vector1, vector2):
        return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True),
                                                   np.linalg.norm(vector2.T, axis=0, keepdims=True))
                                                   
model = tf.keras.applications.VGG16(include_top=False, 
                                    weights='imagenet', 
                                    input_tensor=None, 
                                    input_shape=None, 
                                    pooling='max')
                                    
preprocess = tf.keras.applications.vgg16.preprocess_input

#a_ = ord('а')
#abc_ = ''.join([chr(i) for i in range(a_,a_+32)])
import string
abc_ = string.ascii_lowercase

class ImageWebSocket(tornado.websocket.WebSocketHandler):
    clients = set()
    myfile = 0
    namefile = 0
    
    # embedding data -------------->
    try:
        index = faiss.read_index("flat.index")
    except:
        index = faiss.IndexFlatL2(512)
        
        
    try:
        index2 = faiss.read_index("flat2.index")
    except:
        index2 = faiss.IndexFlatL2(512)
        
    # user data      -------------->
    D = DataBase()
    t_name = "face_id_table"
    embedding = 0
    gen_file = 0
    count_face = 0
    
    #----------------------->
    arr = []
    #----------------------->
    
    #init("weights", "config.json", "cuda")
    def check_origin(self, origin):
        return True

    def open(self):
        ImageWebSocket.clients.add(self)
        print("WebSocket opened from: " + self.request.remote_ip)
    def on_message(self, message):
        ms =  json.loads(message)
        # 104 клавиш клавиатуры
        if list(ms.keys())[0] == "KEYPRESS":
            if len(ms["KEYPRESS"]) != 0:
                Z = np.zeros((len(abc_), 1))
                for k in ms["KEYPRESS"]:
                    if str(k["key_name"]).lower() in abc_:
                        #print (k["key_code"], k["key_name"])
                        Z[abc_.index(k["key_name"]),:] = k["time_press"]
                self.arr.append(Z)
#                print (Z.shape, np.concatenate(self.arr, axis=1).shape)
        if list(ms.keys())[0] == "send_test":
            H = np.concatenate(self.arr, axis=1)
            arr_img = np.zeros((224, 224, 3))
            arr_img[0:H.shape[0],0:H.shape[1], 0] = H
            vector = deep_vector(arr_img)
            self.arr = []
            
            D, I = self.index2.search(np.reshape(vector, [1, 512]), 3) 
            if D[0][0] < 0.6:
                INFO = self.D.client.execute(f"""
                                                SELECT *
                                                FROM {self.t_name}
                                                WHERE {self.t_name}.ID = {I[0][0]} 
                                              """) 
                INFO = INFO[0][1]
            else:
                INFO = "Error"  
            """
            на шаге 0
            
            
            """                      
            print ("SEND_TEST", D, I, INFO)
            self.write_message(json.dumps({"switch":"SendTest", 
                                           "NameABC":INFO, 
                                           "coefficiens": float(D[0][0])}))
            
        if list(ms.keys())[0] == "save_test":
            H = np.concatenate(self.arr, axis=1)
            arr_img = np.zeros((224, 224, 3))
            arr_img[:H.shape[0], :H.shape[1], 0] = H
            self.arr = []
            
            vector = deep_vector(arr_img)
            # clickhouse
            _ID = int(self.D.show_count_tables(self.t_name)[0][0])
            R_N = ms['NameABC']
            self.D.client.execute(f"""INSERT INTO {self.t_name} 
                        (ID, User) 
                        VALUES ({_ID}, '{R_N}')""")
            self.index2.add(np.reshape(vector, [1, 512]))
            faiss.write_index(self.index2, "flat2.index")
            print ("SAVE_TEST", R_N, _ID)            
        
        
        if list(ms.keys())[0] == "Start":
            self.namefile = f'videos/{ms["Start"]["Name"]}'
            self.myfile = open(self.namefile, "wb")
            self.write_message(json.dumps({"switch":"MoreData"}))
        if list(ms.keys())[0] == "Upload":
            da = ms["Upload"]["Data"]
            da = da.split(",")[1]
            file_bytes = io.BytesIO(base64.b64decode(da)).read()
            self.myfile.write(file_bytes)
            self.write_message(json.dumps({"switch":"MoreData"}))
            
        if list(ms.keys())[0] == "Done":
            #self.myfile.close()
            preds, faces, self.embedding = process(self.namefile)
            #os.remove(self.namefile)
            print (preds*100, preds)
            if int(preds*100) > 20:
                textS = f"DEEPFAKE DETECTED ({round(preds*100)}%)"
                colorS = "rgb(221, 84, 84)"
            else:
                textS = f"NO DEEPFAKE DETECTED ({round(preds*100)}%)"
                colorS = "rgb(0, 189, 142)"
                
            if ms["Done"]["Type_Evt"] == "Enter":  
                D, I = self.index.search(np.reshape(self.embedding, [1, 512]), 3) 
                if D[0][0] < 0.6:
                    INFO = self.D.client.execute(f"""
                                                    SELECT *
                                                    FROM {self.t_name}
                                                    WHERE {self.t_name}.ID = {I[0][0]} 
                                                  """)
                    info_dict = {"UserName":INFO[0][1], "Password":INFO[0][2], 
                                 "ImageFace":INFO[0][3], "CountImg":INFO[0][4]}
                else:
                    info_dict = 0
                     
                self.write_message(json.dumps({"switch":"Done", "coord":all_frames, "fps":fps, "total":total, 
                                                "fps_audio": fps_audio , "channel_audio":channel_audio, 
                                                "resolution":resolution,
                                                "duration":duration,
                                                "all_preds":all_preds, "text":textS, "color":colorS, 
                                                "spectro":jpg_as_text, "info_dict":info_dict}))
            #-------------------------------------------------------------->
            if ms["Done"]["Type_Evt"] == "StartReg": 
                dict_face = {}
                self.gen_file = str(uuid.uuid4())[:12]
                os.mkdir(f"faces/{self.gen_file}")
                for ix, i in enumerate(faces):
                    cv2.imwrite(f"faces/{self.gen_file}/{ix}.jpg", i)
                    i_, img_str = cv2.imencode('.jpg', i)
                    dict_face[ix] = base64.b64encode(img_str).decode()
                    self.count_face = ix
                self.write_message(json.dumps({"switch":"DoneReg", "data":dict_face}))
        if list(ms.keys())[0] == "Register":
            _ID = int(self.D.show_count_tables(self.t_name)[0][0])
            R_N = ms['Register']['Name']
            R_P = str(ms['Register']['Pass'])
            R_Ph = ms['Register']['Phone']
            R_M = ms['Register']['Mail']
            self.D.client.execute(f"""INSERT INTO {self.t_name} 
                        (ID, User, Password, Image, CountImg, Phone, Mail) 
                        VALUES ({_ID}, '{R_N}', '{R_P}', '{self.gen_file}', {self.count_face},
                        '{R_Ph}', '{R_M}')""")
            self.index.add(np.reshape(self.embedding, [1, 512]))
            faiss.write_index(self.index, "flat.index")
            self.write_message(json.dumps({"switch":"Confirm"}))
            
    def on_close(self):
        ImageWebSocket.clients.remove(self)
        print("WebSocket closed from: " + self.request.remote_ip)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload_3.7.html", title="Нейронная сеть/Тренировка")
#        self.render("upload_3.4.html", title="Нейронная сеть/Тренировка")

app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/websocket", ImageWebSocket),
        (r"/faces/(.*)", tornado.web.StaticFileHandler, {'path':'./faces'}),
        (r"/static_file/(.*)", tornado.web.StaticFileHandler, {'path':'./static_file'}),
    ])
app.listen(8998)

tornado.ioloop.IOLoop.current().start()

#if __name__ == '__main__':
#    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#    
#https://gist.github.com/seriyps/3773703
#https://codepen.io/jamespeilow/pen/MWWMXPp

