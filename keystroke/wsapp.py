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
from collections import Counter
import requests
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm
from tqdm.auto import tqdm


## clickhouse
#from clickhouse_driver import Client as ClientClickhouse
#class DataBase():
#    def __init__(self):
#        self.client = ClientClickhouse('localhost', settings = { 'use_numpy' : True })
# 
#    def createDB(self, x="test_table"):
#    
#        self.client.execute(f"""CREATE TABLE {x} 
#                                (ID Int64,
#                                 User String) 
#                            ENGINE = MergeTree() ORDER BY User""")
#    def delete(self, x):
#        self.client.execute(f'DROP TABLE IF EXISTS {x}')    
#    def show_count_tables(self, x):
#        start = time.time()
#        LS = self.client.execute(f"SELECT count() FROM {x}")
#        print (time.time()-start, LS)
#        return LS
#    def show_tables(self):
#        print (self.client.execute('SHOW TABLES'))        
#    def get_all_data(self, x):
#        start = time.time()
#        LS = self.client.execute(f"SELECT * FROM {x}")
#        print (time.time()-start, len(LS)) 
#        return LS
#        
#clickhouse_table_name = "keypress_id_table"   
#clickhouse_db = DataBase()
#clickhouse_db.delete(clickhouse_table_name)
#clickhouse_db.createDB(clickhouse_table_name)


def get_bootstrap(data_column_1, # числовые значения первой выборки
                  data_column_2, # числовые значения второй выборки
                  boot_it = 1000, # количество бутстрэп-подвыборок
                  statistic = np.mean, # интересующая нас статистика
                  bootstrap_conf_level = 0.95 # уровень значимости
                  ):
    #print(data_column_1, data_column_2)
    boot_len = max([len(data_column_1), len(data_column_2)])
    #print(boot_len)
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            boot_len, 
            replace = True # параметр возвращения
        ).values
        samples_2 = data_column_2.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values

#        print(samples_1, len(samples_1), boot_len)
#        print(samples_2, len(samples_2), boot_len)
                
        boot_data.append(statistic(samples_1-samples_2)) 
    pd_boot_data = pd.DataFrame(boot_data)
        
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

    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


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
              list_data_line.append(h)
              list_data_line.append(t_dw-t_up)
              list_data_all.append(list_data_line)  
    dataset = pd.DataFrame(list_data_all, columns=['pair', 'time'])
    dataset['time'][dataset['time'] < 0] = np.nan
    return dataset     



class B_Handler(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = "main"
        self.sender_id = self.scope['user'].id
        self.room_group_name = self.room_name
        self.sender_name = self.scope['user']
        if str(self.scope['user']) != 'AnonymousUser':
            self.image_user = self.scope['user'].image_user
            self.path_data = self.scope['user'].path_data
            self.namefile = str()
            
#            # embedding data -------------->
#            if os.path.exists(f"media/data_image/{self.path_data}/keypress.index"):
#                os.remove(f"media/data_image/{self.path_data}/keypress.index")
#            
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
        #print (response, self.scope['user'])
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
                if response["test"] != True:
                    post = Post()
                    post.pure_data = response["KEYPRESS"]
                    post.text = response["text"]
                    post.user_post = self.sender_name
                    post_async = sync_to_async(post.save)
                    await post_async()    
    #                now = datetime.datetime.now().strftime('%H:%M:%S')
                    #---------------------------------------->
                    T = response["text"]
                    div_temp = f"<div id='full_nameuser'>{self.sender_name.username}</div><div id='full_text'>{T}</div><br><table><tbody>"  
                    np_zeros = np.zeros((len(combination), 2)) #len(control_text), 
                    for ih, h in enumerate(combination):
                        idx = indices(T, h)
                        if idx != []:
    #                        print (f"INDICES -> {h} <-------------------", idx)
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
                            "comment_text": response["text"],
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
#                    print ("ТЕСТИРОВАНИЕ", response, post)
                    T0 = post.text.lower().replace("\n", "")
                    T1 = response["text"].lower().replace("\n", "")
                    
                    dt0 = gen_pd(T0, post.pure_data)
                    dt1 = gen_pd(T1, response["KEYPRESS"])
                        
                    booted_data = get_bootstrap(dt0["time"].astype("float"), 
                                                dt1["time"].astype("float"), # числовые значения второй выборки
                                                boot_it = 1000, # количество бутстрэп-подвыборок
                                                statistic = np.median, # интересующая нас статистика
                                                bootstrap_conf_level = 0.95 # уровень значимости
                                                )                    
                    print (booted_data['p_value']) 
                    
                    
                    

    async def wallpost(self, res):
        """ Receive message from room group """
        # Send message to WebSocket
        await self.send(text_data=json.dumps(res))
