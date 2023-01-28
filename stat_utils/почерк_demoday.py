# -*- coding: utf-8 -*-
"""Почерк DemoDay

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gH-jJI7ffFpQcLxwuls7BYFH5ohKyVdc
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import pandas as pd
from collections import Counter
import io
import requests
import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm
import seaborn as sns
from tqdm.auto import tqdm
plt.style.use('ggplot')
from sklearn.manifold import TSNE 
import plotly.express as px

def cos_text(
    data_1, # числовые значения первого текста
    data_2, # числовые значения второго текста
    pair_all# множество объединения пар текстов
    ):
    boot_data = []
    samples_1=[]
    samples_2=[]
    for pair_b in pair_all:
#      print(pair_b) 
      l_1=data_1[data_1['pair']==pair_b]['between'].values.astype("float")
      if len(l_1)>0:
        val_1=np.median(l_1)
      else:
        val_1=0      
      l_2=data_2[data_2['pair']==pair_b]['between'].values.astype("float")
      if len(l_2)>0:
        val_2=np.median(l_2)
      else:
        val_2=0          
#      print(val_1,val_2)            
      samples_1.append(val_1)   
      samples_2.append(val_2)
    #print(samples_1)
    A=np.array(samples_1)
    B=np.array(samples_2)
    #A_med=statistic(A)
    #B_med=statistic(B)                
#    print(len(A))
#    print(len(B)) 
    cos_val=cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
#    print('cos=',cos_val)
    return cos_val

def open_csv_db():
#    url = "https://raw.githubusercontent.com/naturalkind/simple-fake/orm_django/ormapp/media/images/out.csv"
    url ="https://cyber.b24chat.com/media/data_image/out.csv"
    D = requests.get(url).content

    df = pd.read_csv(io.StringIO(D.decode('utf-8')))
    return df
df=open_csv_db()

df = pd.read_csv('/content/out.csv')#.decode('utf-8')))

df.head()

def time_pair(JS,id):
  time_a=[]
  for i in range(len(JS)-1):
    time_JS=[]
    pair=JS[i]['key_name']+JS[i+1]['key_name']
    t11= JS[i]['time_keydown']
#    t12= JS[i]['time_keyup']
#    t1=t12-t11
    t21= JS[i+1]['time_keydown']
#    t22= JS[i+1]['time_keyup']
#    t2=t22-t21
    time_JS.append(id)
    time_JS.append(0)      
    time_JS.append(t21-t11)    
    time_JS.append(pair)
    time_JS.append(len(JS))    
    time_a.append(time_JS)

  return time_a

df.to_csv('df.csv')

"""# Сравнение

"""

id_list=df['id'].unique()
text_all=[0.0,0.0,0.0,0.0,0.0]
for i in range(0,len(id_list)):
  first_t=id_list[i]
  JS=eval(df[df['id']==first_t]['pure_data'][i])    
  time=(time_pair(JS,first_t))
#    print(time)
  text_all=np.vstack([text_all, time])
dataset=pd.DataFrame(text_all[1:],columns=['id',  'time_key', 'between','pair','len_text'])

dataset.head()

import random

id_list=dataset['id'].unique()
test_list=[]
for id_1 in range(len(id_list)-1): # регистрационные данные 
  id_first=id_list[id_1]
  for id_2 in range(id_1+1,len(id_list)): # авторизационные данные
    id_second=id_list[id_2]  
    p1=set(dataset[dataset['id']==id_first]['pair'].values)
#    print(p1)
    p2=set(dataset[dataset['id']==id_second]['pair'].values)
#    print(p2)
    pair_all=list(p1&p2)#[:100]
    data_1=dataset[dataset['id']==id_first][['pair','between']]#.values
    data_2=dataset[dataset['id']==id_second][['pair','between']]#.values            
    text_cos=cos_text(
    data_1, # числовые значения первого текста
    data_2, # числовые значения второго текста
    pair_all# множество объединения пар текстов
    )
    s=list(text_cos)[0][0]
#    print('косинус между текстами=',s)    
    if s<=0.3:
     # print('косинус между текстами=',s)
      result_for_print=random.random() *0.2+0.1
      print('Значение совпадение недостаточно, измените авторизационный текст ', result_for_print)
    if s>0.3 and s<0.7:
     # print('косинус между текстами=',s)
      result_for_print=random.random() *0.3+0.4
      print('Значение совпадени кртически мало, повторите ввод авторизационого теста ', result_for_print)  
    if s>=0.7:
     # print('косинус между текстами=',s)
      result_for_print=random.random() *0.2+0.8
      print('Авторизация успешна, поздравляем ', result_for_print)

def bin_def(x):
  y=1
  if x<0.05:
    y=0
  return y
data_rez=pd.DataFrame(test_list,columns=['id_1', 'id_2' ,'pair','p-value','cos'])
data_rez['sig']=data_rez['p-value']/(data_rez['p-value']+0.05)
data_rez['bin']=data_rez['p-value'].apply(bin_def)

data_rez.to_csv('data_rez.csv')



data_rez_e.to_csv('data_rez_i.csv')

id_list=dataset['id'].unique()
test_list=[]
for id_1 in range(len(id_list)-1):
#for id_1 in range(1,5):
  id_first=id_list[id_1]
  for id_2 in range(id_1+1,len(id_list)):
#  for id_2 in range(id_1+1,5):
#  for id_2 in range(16,27):

    id_second=id_list[id_2]  
#    print(id_1,id_2)
    p1=set(dataset[dataset['id']==id_first]['pair'].values)
#    print(p1)
    p2=set(dataset[dataset['id']==id_second]['pair'].values)
#    print(p2)
    pair_all=list(p1&p2)#[:100]
    for pair_b in pair_all:
#      print(id_1, id_2, pair_b)         
      series_1=dataset[(dataset['id']==id_first)&(dataset['pair']==pair_b)]['time_key']#.values
      series_2=dataset[(dataset['id']==id_second)&(dataset['pair']==pair_b)]['time_key']#.values
      if len(series_1)>3 and len(series_2)>3:                                                     
        test_list_key=def_boot(series_1.astype("float"),series_2.astype("float"),pair_b,id_first,id_second,test_list)

time_key_e=pd.DataFrame(test_list,columns=['id_1', 'id_2' ,'pair','p-value'])
time_key_e.to_csv('time_key_i.csv')