from channels.generic.websocket import AsyncJsonWebsocketConsumer
from keystroke.models import *
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async

import json
import datetime
import asyncio
import aioredis
import async_timeout
import re
import redis
import time
import uuid
import base64, io, os
import numpy as np

import pandas as pd
import tensorflow as tf
from collections import Counter
import requests
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# clickhouse
from clickhouse_driver import Client as ClientClickhouse
import faiss
class DataBase():
    def __init__(self):
        self.client = ClientClickhouse('localhost', settings = { 'use_numpy' : True })
 
    def createDB(self, x="test_table"):
    
        self.client.execute(f"""CREATE TABLE {x} 
                                (ID Int64,
                                 User String) 
                            ENGINE = MergeTree() ORDER BY User""")
    def delete(self, x):
        self.client.execute(f'DROP TABLE IF EXISTS {x}')    
    def show_count_tables(self, x):
        start = time.time()
        LS = self.client.execute(f"SELECT count() FROM {x}")
        print (time.time()-start, LS)
        return LS
    def show_tables(self):
        print (self.client.execute('SHOW TABLES'))        
    def get_all_data(self, x):
        start = time.time()
        LS = self.client.execute(f"SELECT * FROM {x}")
        print (time.time()-start, len(LS)) 
        return LS
        
clickhouse_table_name = "keypress_id_table"   
clickhouse_db = DataBase()
clickhouse_db.delete(clickhouse_table_name)
clickhouse_db.createDB(clickhouse_table_name)

alf_ = ord('а')
abc_ = ''.join([chr(i) for i in range(alf_, alf_+32)])
abc_ += " ,."




def time_pair(JS):
    time_a=[]
    for i in range(len(JS)-1):
        time_JS=[]
        pair=JS[i]['key_name']+JS[i+1]['key_name']
        t11= JS[i]['time_keydown']
        t12= JS[i]['time_keyup']
        t1=t12-t11
        t21= JS[i+1]['time_keydown']
        t22= JS[i+1]['time_keyup']
        t2=t22-t21
        time_JS.append(pair)
        time_JS.append(t21-t12)    
        time_a.append(time_JS)
    dataset = pd.DataFrame(time_a, columns=['pair', 'time'])
    return dataset
    
#def time_pair(JS,id,user):
#  time_a=[]
#  for i in range(len(JS)-1):
#    time_JS=[]
#    pair=JS[i]['key_name']+JS[i+1]['key_name']
#    t11= JS[i]['time_keydown']
#    t12= JS[i]['time_keyup']
#    t1=t12-t11
#    t21= JS[i+1]['time_keydown']
#    t22= JS[i+1]['time_keyup']
#    t2=t22-t21
#    time_JS.append(id)
#    time_JS.append(user)
#    time_JS.append(t1+t2)      
#    time_JS.append(t21-t12)    
#    time_JS.append(pair)
#    time_a.append(time_JS)
#  return time_a

#----------------------------------------------------->



#get_bootstrap
def get_bootstrap_cos(data_1, # числовые значения первой выборки
                      data_2, # числовые значения второй выборки
                      boot_it = 1000, # количество бутстрэп-подвыборок
                      statistic = np.mean, # интересующая нас статистика
                      bootstrap_conf_level = 0.95, # уровень значимости
                      pair_all=['а ']):
    #print(data_column_1, data_column_2)
    boot_len = 1  #max([len(data_column_1), len(data_column_2)])
    #boot_it=1000
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1=[]
        samples_2=[]
        cos_val=[[1]]
        for pair_b in pair_all:
            data_column_1 = data_1[data_1['pair']==pair_b]['time']#.values[0]#.astype("float")
            data_column_2 = data_2[data_2['pair']==pair_b]['time']#.values[0] #.astype("float").values
#            val_1 = data_column_1.values[0]
#            val_2 = data_column_2.values[0]
            
            if len(data_column_1)>2 and len(data_column_2)>2:
                val_1 =(np.random.choice(data_column_1, size=1, replace=True))
                val_2 =(np.random.choice(data_column_2, size=1, replace=True))
                #print("TWO--->", val_1, val_2)
                #time.sleep(2)
                samples_1.append(val_1[0])   
                samples_2.append(val_2[0])
        #print(samples_1)
        if len(samples_1)>2 and len(samples_2)>2:
            A=np.array(samples_1)
            B=np.array(samples_2)  
            A_med=statistic(A)
            B_med=statistic(B)                    
#            print(A)
#            print(B) 
            C=statistic(np.subtract(A,B))                  
            cos_val=C #cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
            #print('cos=',cos_val[0][0])
            boot_data.append(cos_val)
            
    pd_boot_data = pd.DataFrame(boot_data)
    p_value=100
    quants=[0,0]
    if len(boot_data)>1:
      left_quant = (1 - bootstrap_conf_level)/2
      right_quant = 1 - (1 - bootstrap_conf_level) / 2
      quants = pd_boot_data.quantile([left_quant, right_quant])
        
      p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
      )
      p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
      )
      p_value = min(p_1, p_2) * 2
        
      # Визуализация
#      _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
#      for bar in bars:
#          if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
#            bar.set_facecolor('red')
#          else: 
#            bar.set_facecolor('grey')
#            bar.set_edgecolor('black')
#    
#      plt.style.use('ggplot')
#      plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
#      plt.xlabel('boot_data')
#      plt.ylabel('frequency')
#      plt.title("Histogram of boot_data")
#      plt.show()
       
    return {"boot_data": boot_data, 
             "quants": quants, 
             "p_value": p_value}
#def def_boot
def def_boot_cos(data_1, data_2, pair_all, test_list_all):
    #print(id_1, id_2, pair_b)      
    test_list=[]
    booted_data=get_bootstrap_cos(data_1, 
                                  data_2, # числовые значения второй выборки
                                  boot_it = 100, # количество бутстрэп-подвыборок
                                  statistic = np.median, # интересующая нас статистика
                                  bootstrap_conf_level = 0.95, # уровень значимости
                                  pair_all=pair_all
                                  )
    test_list.append(len(pair_all))
    test_list.append(booted_data["p_value"])
    test_list.append(np.median(booted_data["boot_data"]))
    
    test_list_all.append(test_list)
#    print('p_value=',booted_data["p_value"])
#    print('median_cos=',np.median(booted_data["boot_data"]))
#    print('Количество общих пар', len(pair_all))  
    return test_list_all, booted_data["p_value"], np.median(booted_data["boot_data"]), len(pair_all)


#------------------------------------------------------>



combination = ["ст", "то", "но", "на", "по", "ен", "ни", "не", "ко", "ра", "ов", "ро", "го", "ал",
               "пр", "ли", "ре", "ос", "во", "ка", "ер", "от", "ол", "ор", "та", "ва", "ел", "ть",
               "ет", "ом", "те", "ло", "од", "ла", "ан", "ле", "ве", "де", "ри", "ес", "ат", "ог",
               "ль", "он", "ны", "за", "ти", "ит", "ск", "ил", "да", "ой", "ем", "ак", "ме", "ас",
               "ин", "об", "до", "че", "мо", "ся", "ки", "ми", "се", "тр", "же", "ам", "со", "аз",
               "нн", "ед", "ис", "ав", "им", "ви", "тв", "ар", "бы", "ма", "ие", "ру", "ег", "бо",
               "сл", "из", "ди", "чт", "вы", "вс", "ей", "ия", "пе", "ик", "ив", "сь", "ое", "их",
               "ча", "ну", "мы"] # 101   


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def gen_pd(T, post):
    list_data_all=[]
    for ih, h in enumerate(combination):
        idx = indices(T, h)
        if idx != []:
            for k in idx:
              list_data_line=[]
              t_up = post[k]["time_keyup"]
              t_dw = post[k+1]["time_keydown"]
              """
                t11= JS[i]['time_keydown']
                t12= JS[i]['time_keyup']
                t1=t12-t11
                t21= JS[i+1]['time_keydown']
                t22= JS[i+1]['time_keyup']
                t2=t22-t21              
              
              """
              
              list_data_line.append(h)
              list_data_line.append(t_dw-t_up)
              list_data_all.append(list_data_line)  
    dataset = pd.DataFrame(list_data_all, columns=['pair', 'time'])
#    dataset['time'][dataset['time'] < 0] = np.nan
    return dataset     

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def median_filter(data, filt_length=3):
    '''
    Computes a median filtered output of each [n_bins, n_channels] data sample
        and returns output w/ same shape, but median filtered
    NOTE: as TF doesnt have a median filter implemented, this had to be done in very hacky way...
    '''
    edges = filt_length// 2

    # convert to 4D, where data is in 3rd dim (e.g. data[0,0,:,0]
    exp_data = tf.expand_dims(tf.expand_dims(data, 0), -1)
    print ("median_filter", exp_data.shape)
    # get rolling window
    wins = tf.image.extract_patches(images=exp_data, sizes=[1, filt_length, 1, 1],
                       strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    # get median of each window
    wins = tf.math.top_k(wins, k=2)[0][0, :, :, edges]
    # Concat edges
    out = tf.concat((data[:edges, :], wins, data[-edges:, :]), 0)

    return out    

def deep_vector(x):
       t_arr = np.expand_dims(x, axis=0)
       processed_img = preprocess_idx(t_arr)
       preds = model_idx.predict(processed_img)
       return preds

def similarity(vector1, vector2):
        return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True),
                                                   np.linalg.norm(vector2.T, axis=0, keepdims=True))


model_idx = tf.keras.applications.VGG16(include_top=False, 
                                    weights='imagenet', 
                                    input_tensor=None, 
                                    input_shape=None, 
                                    pooling='max')
preprocess_idx = tf.keras.applications.vgg16.preprocess_input

class B_Handler(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = "main"
        self.sender_id = self.scope['user'].id
        self.room_group_name = f"{self.room_name}_{self.sender_id}"
        self.sender_name = self.scope['user']
        if str(self.scope['user']) != 'AnonymousUser':
            self.image_user = self.scope['user'].image_user
            self.path_data = self.scope['user'].path_data
            self.namefile = str()
            
#            # embedding data -------------->
#            if os.path.exists(f"media/data_image/{self.path_data}/keypress.index"):
#                os.remove(f"media/data_image/{self.path_data}/keypress.index")
            
#            try:
#                self.index = faiss.read_index(f"media/data_image/{self.path_data}/keypress.index")
#            except:
#                self.index = faiss.IndexFlatL2(len(abc_)*len(abc_))
            
            self.NIDX = 0
            self.previous_Z = 0
            self.control_text_ = ""
        #----------------------->
#        self.Z = np.zeros((len(abc_), 1))
#        self.Z_pad = np.zeros((len(abc_), 1))
        #----------------------->
            
        print ("CHANNEL_LAYERS", self.channel_name, self.room_group_name, self.scope['user']) #self.scope, 
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()


    async def disconnect(self, close_code):
        print("Disconnected", close_code)
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """
        Receive message from WebSocket.
        Get the event and send the appropriate event
        """
        
        response = json.loads(text_data)
        print (response, self.scope['user'])
        event = response.get("event", None)
        if self.scope['user'].is_authenticated:  
            # KEYSTROKE
            if event == "KEYPRESS":
                #print (event)
                pass
            if event == "send_test":
                """
                сохраняю с помощью orm django
                сохраняю данные в clickhouse
                из clickhouse делаю загрузку данных
                для дальнейшего анализа
                """
                T = response["text"].replace('\xa0', ' ').replace("\n\n", " ").replace("\n", " ").lower()
                value_pure = ""
                for op in response["KEYPRESS"]:
                    #print (op)
                    value_pure += op["key_name"]
                value_pure = value_pure.lower()
                print (value_pure == T)
                if value_pure == T:
                    if response["test"] != True:
                        post = Post()
                        post.pure_data = response["KEYPRESS"]
                        post.text = T
                        post.user_post = self.sender_name
                        post_async = sync_to_async(post.save)
                        await post_async()    
                        #---------------------------------------->
                        
                        div_temp = f"<div id='full_nameuser'>{self.sender_name.username}</div><div id='full_text'>{T}</div><br><table><tbody>"  
                        np_zeros = np.zeros((len(combination), 2)) #len(control_text), 
                        #print ("............", T, response["KEYPRESS"])

                        for ih, h in enumerate(combination):
                            idx = indices(T, h)
                            if idx != []:
                                #print (f"INDICES -> {h} <-------------------", idx)
                                temp_ls = []
                                for k in idx:
                                    temp_ls.append(response["KEYPRESS"][k+1]["time_keydown"]-response["KEYPRESS"][k]["time_keyup"])
                                np_zeros[ih, 0] = np.median(np.array(temp_ls))
                                np_zeros[ih, 1] = T.count(h)
                                div_temp += f"<tr><td>{h}</td><td>{T.count(h)}</td><td>{np.median(np.array(temp_ls))}</td></tr>"
        #                        print ("MDEIANA", np.median(np.array(temp_ls)))
                        div_temp += "</tbody></table>"                
                        #---------------------------------------->
                        now = datetime.datetime.now().strftime('%H:%M:%S')
                        _data={
                                "type": "wallpost",
                                "comment_text": T,
                                "post_id": post.id,
                                "user_id": self.sender_id,
                                "user_post": self.sender_name.username,
                                "timecomment":now,
                                "status" : "send_test",
                                "html": div_temp
                            }
                        await self.channel_layer.group_send(self.room_group_name, _data)  
                    else:
                        post = await database_sync_to_async(Post.objects.get)(id=response["id_post"])
                        T1 = post.text.lower().replace("\n", "")
                        
                        T2 = response["text"].lower().replace("\n", "")
                        #'pair', 'time'
                        #--------------->
                        if T1 == T2:
                            print (".................")
                            dt0 = time_pair(post.pure_data)
                            dt1 = time_pair(response["KEYPRESS"])
                            
                            P_data1 = dt0["time"].tolist()
                            P_data2 = dt1["time"].tolist()
                            #------------------------->
                            
                            #print (P_data1, P_data2)
                            zeros_1 = np.zeros((1, len(P_data1)))
                            zeros_2 = np.zeros((1, len(P_data2)))
                            for ix, k in enumerate(P_data2):
#                                if str(k["key_name"]).lower() in abc_:
                                    zeros_1[0, ix] = P_data1[ix]#["time_press"]
                                    zeros_2[0, ix] = P_data2[ix]#["time_press"]
                                    
                            zeros_1_m = np.float32(zeros_1) #median_filter(np.float32(zeros_1)) 
                            zeros_2_m = np.float32(zeros_2) #median_filter(np.float32(zeros_2))  
                            
                            arr_img_1 = np.zeros((224, 224, 3))
                            arr_img_1[0:zeros_1_m.shape[0], 0:zeros_1_m.shape[1], 0] = zeros_1_m
#                            vector1 = deep_vector(np.float32(arr_img_1))
                            vector1 = deep_vector(arr_img_1)
                            
                            arr_img_2 = np.zeros((224, 224, 3))
                            arr_img_2[0:zeros_2_m.shape[0], 0:zeros_2_m.shape[1], 0] = zeros_2_m
                            #vector2 = deep_vector(np.float32(arr_img_2))
                            vector2 = deep_vector(arr_img_2)
                            
                            sim = similarity(vector1, vector2)
                            CS = cosine_similarity(vector1, vector2) 
                            CD = cosine_distances(vector1, vector2)                             
                            print (CS, CD, sim)                      
#                        p1 = set(dt0['pair'].values)    
#                        p2 = set(dt1['pair'].values)
#                        pair_all = list(p1 & p2)
#                        print ("ONE----->", pair_all)
#                        test_list = []
#                        test_list, p_value, med_cos, len_pair = def_boot_cos(dt0, dt1, pair_all, test_list)
#                        #test_list_all, booted_data["p_value"], np.median(booted_data["boot_data"])
#                        
#                        #print (test_list)

#                        t_S = "" 
#                        for io, o in enumerate(test_list[0]):
#                            t_S += f"{o}, "
#                        div_temp = f"[{t_S}], p_value = {p_value}, median_cos = {med_cos}, Количество общих пар: {len_pair}"                            
#                        #---------------->
#                        _data={
#                                "type": "wallpost",
#                                "status" : "send_test_p",
#                                "html": div_temp
#                            }
#                        await self.channel_layer.group_send(self.room_group_name, _data) 

    async def wallpost(self, res):
        """ Receive message from room group """
        # Send message to WebSocket
        await self.send(text_data=json.dumps(res))
