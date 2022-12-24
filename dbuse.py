from clickhouse_driver import Client
import threading
from dface import MTCNN, FaceNet
import numpy as np
import sys, cv2, os, time, json
import torchvision.transforms.functional as TF
import torch
import faiss
import pickle

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      #time.sleep(0.2)
      cv2.destroyAllWindows()
      
class DATA(object):
   def __init__(self):
       self.file = []

   def parseIMG(self, dir_name):
       path = f"{dir_name}/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f): 
                      if ".png" in file.lower(): 
                          self.file.append(os.path.join(r, file))
                      elif ".jpg" in file.lower(): 
                          self.file.append(os.path.join(r, file))
                      elif ".jpeg" in file.lower(): 
                          self.file.append(os.path.join(r, file))


class DataBase():
    def __init__(self):
        self.client = Client('localhost', settings = { 'use_numpy' : True })

    def createDB(self, x="face_id_table"):
        self.client.execute(f"""CREATE TABLE {x} 
                                (ID Int64,
                                 User String, 
                                 Password String, 
                                 Image String,
                                 CountImg Int64,
                                 Phone String,
                                 Mail String) 
                            ENGINE = MergeTree() ORDER BY User""")
                            
    #def add_data(self):

    def delete(self, x):
        self.client.execute(f'DROP TABLE IF EXISTS {x}')

    def show_tables(self):
        print (self.client.execute('SHOW TABLES'))
       
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
 
def similarity(vector1, vector2):
    return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True),
           np.linalg.norm(vector2.T, axis=0, keepdims=True)) 
        
def func_rec(ID):
##    print (len(arr_list), len(arr.file), len(G.keys()))
    while len(arr_list) != 0:
        for ix, i in enumerate(arr_list):
            G[ID] = [arr.file[ix]]
            del arr.file[ix]
            del arr_list[ix]
            for iix, ii in enumerate(arr_list):
                    KEF = similarity(i, ii) 
                    KEF = float(KEF[0][0])
                    if KEF > tresh:
    #                if KEF == 1.0:
    ##                    print (KEF, KEF>tresh, type(KEF), type(tresh))
                        G[ID].append(arr.file[iix])
                        del arr.file[iix]
                        del arr_list[iix] 
            ID += 1

def func_create_idx():
    index = faiss.IndexFlatL2(512)
    index_dict = {}
    
    arr = DATA()
    arr.parseIMG("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/MEDIADATA/face_img")
        

    tresh = .60
    G = {}

    arr_list = []
    s = time.time()
    id = 0
    for i in arr.file:
        #try:
            im = cv2.imread(i)
            face = cv2.resize(im, (160,160))
            face = TF.to_tensor(np.float32(face))
            face = (face - 127.5) / 128.0
            face = torch.reshape(face, (1,3,160,160))
            
            _vector = facenet.embedding_v2(face)
            
            index_dict.update({id: (i, _vector)})
            index.add(_vector)
            print (_vector.shape)#, a.shape)
            id += 1
    #--------------------->
    faiss.write_index(index, "flat.index")
    
    
    with open("face_vec.json", 'wb+') as f:
        pickle.dump(index_dict, f, True)
    f.close()
    
    
    #--------------------->
def test_():
    im = cv2.imread("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/MEDIADATA/faces_cut_v1/1/0_2c908073-63e_20180512_183621.jpg")
    face = cv2.resize(im, (160,160))
    imgs(face)
    face = TF.to_tensor(np.float32(face))
    face = (face - 127.5) / 128.0
    face = torch.reshape(face, (1,3,160,160))
    _vector = facenet.embedding_v2(face)
    print (_vector.shape)
#------------------------------------> 
    index = faiss.read_index("flat.index")
    
    with open("face_vec.json", 'rb') as f:
        index_dict = pickle.load(f)
    
    D, I = index.search(_vector, 7)  # Возвращает результат: Distances, Indices
    print(I, I.shape)
    print(D, D.shape)
    
    print (max(I[0,:]))
    
    info_1 = index_dict[max(I[0,:])]
    print (info_1[0])
    im = cv2.imread(info_1[0])
    imgs(im) 
    info_2 = index_dict[4082] 
    print (info_2[0]) 
    im = cv2.imread(info_2[0])
    imgs(im) 


if __name__ == "__main__":
#    facenet = FaceNet("cuda")
#    # Test idx
#    func_create_idx()
#    test_()
    
    D = DataBase()
#    print (D.client.execute(f"""
#                                    SELECT 
#                                    *
#                                    FROM face_id_table
#                                    WHERE face_id_table.ID = 5 
#                              """))

    
    D.delete("face_id_table")
    D.createDB()

    if os.path.exists("flat.index"):
        os.remove("flat.index")
        
    if os.path.exists("flat2.index"):
        os.remove("flat2.index")
#    D.client.execute(f"""INSERT INTO "face_id_table" 
#                        (ID, User, Password) 
#                        VALUES (5, 'Victor', '123')""")    
#    print (D.show_count_tables("face_id_table")[0][0])
#    print (D.get_all_data("face_id_table"))

#    _temp = D.client.execute(f"""   
#                            SELECT
#                              *
#                            FROM face_id_table
#                            """)
    #print (_temp)
    #print (D.client.execute("""SHOW CREATE TABLE face_id_table"""))
    
#------------------------------------>           
            
#            arr_list.append(_vector)
#    print (time.time()-s)
#    func_rec(0)

#    print (">>>>>>>>>", len(arr_list), len(G), len(arr.file))

#    with open('data_1.json', 'w') as fp:
#        json.dump(G, fp)   
#    


#        with open('data_1.json', 'r') as fp:
#            D = json.load(fp)
#            
#            name_file = "faces_cut"
#            try:
#                os.mkdir(f"{name_file}") 
#            except FileExistsError:
#                pass
#                
#            for U in list(D.keys()):
#                os.mkdir(f"{name_file}/{U}")
#                for UU in D[U]:
#                    i1 = open(UU, "rb").read()
#                    name = UU.split("/")[-1]
#                    i2 = open(f"{name_file}/{U}/{name}", "wb").write(i1)







