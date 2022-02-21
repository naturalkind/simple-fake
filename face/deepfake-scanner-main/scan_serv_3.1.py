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


import face_alignment
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device='cuda',
                                  face_detector=face_detector)


#def get_frames(video, batch_size=10, target_fps=1):
#	vid = cv2.VideoCapture(video)
#	global total
#	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#	if total <= 0:
#		return None
#	global fps
#	fps = vid.get(cv2.CAP_PROP_FPS)
#	
#	if target_fps > fps:
#		target_fps = fps
#	nfrm = int(total/fps*target_fps)
#	idx = np.linspace(0, total, nfrm, endpoint=False, dtype=int)
#	batch = []
#	
#	global h, w
#	
#	for i in range(total):
#		ok = vid.grab()
#		ok, frm = vid.retrieve()
#		if i not in idx:
#			continue
#		if not ok:
#			continue
#		h, w = frm.shape[:2]
#		if w*h > 1920*1080:
#			scale = 1920/max(w, h)
#			frm = cv2.resize(frm, (int(w*scale), int(h*scale)))
#		frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#		batch.append(frm)
#		if len(batch) == batch_size:
#			yield batch
#			batch = []
#	if len(batch) > 0:
#		yield batch
#	vid.release()

do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)

def chunks(lst, count):
    start = 0
    for i in range(count):
          stop = start + len(lst[i::count, :])
          yield lst[start:stop, :]
          start = stop  


def get_frames(video, batch_size=10, target_fps=1):
	clip = VideoFileClip(video)
	
	
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
	
	_s = clip.audio.to_soundarray(nbytes=4)
	fig, ax = plt.subplots()
	
	cut_wave = list(chunks(_s, int(clip.duration)))
	for signal in cut_wave:
		melspec = do_melspec(y=signal[:,0], sr=clip.audio.fps, n_mels=128, fmax=4000)
		norm_melspec = pwr_to_db(melspec, ref=np.max)

		
		librosa.display.specshow(norm_melspec, y_axis='mel', fmax=4000, x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Mel spectrogram audio')
		fig.canvas.draw()
		image = np.array(fig.canvas.buffer_rgba())
		
		_, img_str = cv2.imencode('.jpg', image.astype(np.uint8))
		
		jpg_as_text.append(base64.b64encode(img_str).decode())	
		plt.clf()
	
	if target_fps > fps:
		target_fps = fps
	nfrm = int(total/fps*target_fps)
	idx = np.linspace(0, total, nfrm, endpoint=False, dtype=int)
	batch = []
	#frames = clip.iter_frames()
	for i, frm in enumerate(clip.iter_frames()):
		if i not in idx:
			continue
		h, w = np.array(frm).shape[:2]
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
	return clusters


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
#					all_frames.append(box.astype(int).tolist())
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
			#print (Z)
			temp_a.append(Z/int(fps))
		except:
			temp_a.append(np.array([0,0,0,0]))

	all_frames.clear()
	print (lnf, total, len(new_list), len(temp_a))
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


def process(file):
	try:
		preds, faces = scan(file)
		print ()
		if preds is None:
			return 0.5

		clust = cluster(faces)
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
		return np.clip(score, 0.01, 0.99)

	except Exception as e:
		print(e, file, file=sys.stderr)
		return 0.5


def init(models_dir, cfg_file, dev):
	global device, mtcnn, facenet, deepware, margin, face_size, all_frames, all_preds
	all_frames = []
	all_preds = []
	with open(cfg_file) as f:
		cfg = json.loads(f.read())

	arch = cfg['arch']
	margin = cfg['margin']
	face_size = (cfg['size'], cfg['size'])

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


def main():
	if len(sys.argv) != 5:
		print('usage: scan.py <scan_dir> <models_dir> <cfg_file> <device>')
		exit(1)
	  
	init("weights", "config.json", "cuda")

	if os.path.isdir(sys.argv[1]):
		files = glob.glob(sys.argv[1]+'/*')
	else:
		with open(sys.argv[1], 'r') as f:
			files = [l.strip() for l in f.readlines()]

	with ThreadPoolExecutor(max_workers=4) as ex:
		preds = list(tqdm(ex.map(process, files), total=len(files)))

	with open('result.csv', 'w') as f:
		print('filename,label', file=f)
		for i, file in enumerate(files):
			print('%s,%.4f'%(file, preds[i]), file=f)

class ImageWebSocket(tornado.websocket.WebSocketHandler):
	clients = set()
	myfile = 0
	namefile = 0
	init("weights", "config.json", "cuda")
	def check_origin(self, origin):
		return True

	def open(self):
		ImageWebSocket.clients.add(self)
		print("WebSocket opened from: " + self.request.remote_ip)
	def on_message(self, message):
		ms =  json.loads(message)
		#print ("MSG >>>", ms)
		if list(ms.keys())[0] == "Start":
			self.myfile = open(f'videos/{ms["Start"]["Name"]}', "wb")
			self.namefile = f'videos/{ms["Start"]["Name"]}'
			self.write_message(json.dumps({"MoreData":"MoreData"}))
		if list(ms.keys())[0] == "Upload":
			da = ms["Upload"]["Data"]
			da = da.split(",")[1]
			file_bytes = io.BytesIO(base64.b64decode(da)).read()
			self.myfile.write(file_bytes)
			
			self.write_message(json.dumps({"MoreData":"MoreData"}))
		if list(ms.keys())[0] == "Done":
			preds = process(self.namefile)
			print (preds*100, preds)
			if int(preds*100) > 20:
				textS = f"DEEPFAKE DETECTED ({round(preds*100)}%)"
				colorS = "rgb(221, 84, 84)"
			else:
				textS = f"NO DEEPFAKE DETECTED ({round(preds*100)}%)"
				colorS = "rgb(0, 189, 142)"
			self.write_message(json.dumps({"O":"Done", "coord":all_frames, "fps":fps, "total":total, 
											"fps_audio": fps_audio , "channel_audio":channel_audio, "resolution":resolution,
											"duration":duration,
											"all_preds":all_preds, "text":textS, "color":colorS, "spectro":jpg_as_text}))

#			self.cap = cv2.VideoCapture(f'videos/{self.namefile}_output.mp4')
#			while (self.cap.isOpened()):
#				ret, frame = self.cap.read()
#				if ret == False:
#					break
#				frame = cv2.putText(frame, textS, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colorS, 3, cv2.LINE_AA, False)
#				_, img_str = cv2.imencode('.jpg', frame.astype(np.uint8))
#				#imgs(frame)
#				#print (img_str)
#				#BS = img_str.tobytes()
#				jpg_as_text = base64.b64encode(img_str).decode()
#				#print (jpg_as_text)
#				self.write_message(json.dumps({"O":"O", "data":jpg_as_text}))	
#					
#			self.cap.release()			


	def on_close(self):
		ImageWebSocket.clients.remove(self)
		print("WebSocket closed from: " + self.request.remote_ip)


class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.render("upload_3.4.html", title="Нейронная сеть/Тренировка")

app = tornado.web.Application([
		(r"/", MainHandler),
		(r"/websocket", ImageWebSocket),
		(r"/(B24CYBER-2-768x768.png)", tornado.web.StaticFileHandler, {'path':'./'}),
		(r"/(b24cyber-cover-1.jpg)", tornado.web.StaticFileHandler, {'path':'./'}),
	])
app.listen(8800)

tornado.ioloop.IOLoop.current().start()

#if __name__ == '__main__':
#	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#	
#https://gist.github.com/seriyps/3773703
#https://codepen.io/jamespeilow/pen/MWWMXPp

