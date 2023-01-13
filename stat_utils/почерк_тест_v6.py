# -*- coding: utf-8 -*-
"""Почерк тест v6

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T6Ht7i30x5UDYeQtjlhoM5AsYnur3j7H
"""

import numpy as np
from numpy import dot
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from numpy.linalg import norm
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

"""### новая функция бутстреп для косинуса угла между векторами"""

def get_bootstrap_cos(
    data_1, # числовые значения первой выборки
    data_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95, # уровень значимости
    pair_all=['а ']):
    #print(data_column_1,data_column_2)
    boot_len = 1  #max([len(data_column_1), len(data_column_2)])
    print(pair_all)
    #boot_it=1000
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
      samples_1=[]
      samples_2=[]
      cos_val=[[1]]
      for pair_b in pair_all:
        data_column_1=data_1[data_1['pair']==pair_b]['between'].values[0]#.astype("float")
        #print(data_column_1)
        data_column_2=data_2[data_2['pair']==pair_b]['between'].values[0] #.astype("float").values
        val_1=data_column_1[0]
        val_2=data_column_2[0]        
        if len(data_column_1)>1 and len(data_column_2)>1:
          val_1 =(np.random.choice(data_column_1, size=1, replace=True))
          val_2 =(np.random.choice(data_column_2, size=1, replace=True))
          samples_1.append(val_1[0])   
          samples_2.append(val_2[0])
      #print(samples_1)
      if len(samples_1)>2 and len(samples_2)>2:
        A=np.array(samples_1)
        B=np.array(samples_2)        
#        print(A)
#        print(B)                  
        cos_val=cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
        #print('cos=',cos_val[0][0])
        boot_data.append((cos_val[0][0]))
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
        
    # Визуализация
 # "  _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
 #   for bar in bars:
 #       if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
 #           bar.set_facecolor('red')
 #       else: 
 #           bar.set_facecolor('grey')
 #           bar.set_edgecolor('black')
 #   
 #   plt.style.use('ggplot')
 #   plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
 #   plt.xlabel('boot_data')
 #   plt.ylabel('frequency')
 #   plt.title("Histogram of boot_data")
 #   plt.show()
       
    return {"boot_data": boot_data, 
             "quants": quants, 
             "p_value": p_value}

"""## Старая функция бутстрап"""

def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
    ):
    #print(data_column_1,data_column_2)
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

#        print(samples_1, len(samples_1),boot_len)
#        print(samples_2,len(samples_2),boot_len)
                
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
        
    # Визуализация
 # "  _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
 #   for bar in bars:
 #       if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
 #           bar.set_facecolor('red')
 #       else: 
 #           bar.set_facecolor('grey')
 #           bar.set_edgecolor('black')
 #   
 #   plt.style.use('ggplot')
 #   plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
 #   plt.xlabel('boot_data')
 #   plt.ylabel('frequency')
 #   plt.title("Histogram of boot_data")
 #   plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}

"""## Обращение к базе"""

def open_csv_db():
#    url = "https://raw.githubusercontent.com/naturalkind/simple-fake/orm_django/ormapp/media/images/out.csv"
    url ="https://cyber.b24chat.com/media/data_image/out.csv"
    D = requests.get(url).content

    df = pd.read_csv(io.StringIO(D.decode('utf-8')))
    return df
df=open_csv_db()

df = pd.read_csv('/content/out.csv')#.decode('utf-8')))

def re_id(x):
  if x==1:
    y=1
  else:
    y=2
  return y

df['new_user_id']=df['user_post_id'].apply(re_id)

df.head()

"""## Новая функция обращения к бутстрап с косинусом"""

def def_boot_cos(data_1,data_2,pair_all,id_1,id_2,test_list_all):
  #print(id_1, id_2, pair_b)      
  test_list=[]
  booted_data=get_bootstrap_cos(data_1,data_2, # числовые значения второй выборки
      boot_it = 1000, # количество бутстрэп-подвыборок
      statistic = np.median, # интересующая нас статистика
      bootstrap_conf_level = 0.95, # уровень значимости
      pair_all=pair_all
      )
  test_list.append(id_1)
  test_list.append(id_2)
  test_list.append(len(pair_all))
  test_list.append(booted_data["p_value"])
  test_list.append(np.median(booted_data["boot_data"]))
    
  test_list_all.append(test_list)
  print('p_value=',booted_data["p_value"])
  print('median_cos=',np.median(booted_data["boot_data"]))
  print('Количество общих пар',len(pair_all))  
  return test_list_all

"""## Старая функция обращения к бутстрап"""

def def_boot(series_1,series_2,pair_b,id_1,id_2,test_list_all):
  print(id_1, id_2, pair_b)      
  test_list=[]
  booted_data=get_bootstrap(series_1, series_2, # числовые значения второй выборки
      boot_it = 1000, # количество бутстрэп-подвыборок
      statistic = np.median, # интересующая нас статистика
      bootstrap_conf_level = 0.95 # уровень значимости
      )
  test_list.append(id_1)
  test_list.append(id_2)
  test_list.append(pair_b)
  test_list.append(booted_data["p_value"])
  test_list_all.append(test_list)
  print('p_value=',booted_data["p_value"])
  return test_list_all

"""## Время"""

def time_pair(JS,id,user):
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
    time_JS.append(id)
    time_JS.append(user)
    time_JS.append(t1+t2)      
    time_JS.append(t21-t12)    
    time_JS.append(pair)
    time_a.append(time_JS)
  return time_a

df.to_csv('df.csv')

id_list=df['id'].unique()
#id_list=df['new_user_id'].unique()
text_all=[0.0,0.0,0.0,0.0,0.0]
for i in range(len(id_list)):
  first_t=id_list[i]
  JS=eval(df[df['id']==first_t]['pure_data'][i])
  user=(df[df['id']==first_t]['new_user_id'][i])  
  time=(time_pair(JS,first_t,user))
#    print(time)
  text_all=np.vstack([text_all, time])
dataset=pd.DataFrame(text_all[1:],columns=['id', 'user', 'time_key', 'between','pair'])

#from pandas.core.groupby import groupby
dataset_median=dataset.groupby(['id',	'pair'])['time_key','between'].median().reset_index()
dataset_median

dataset_median.to_excel('dataset_median_2.xls')

"""## НЕ ПОТЕРЯТЬ ОЧЕНЬ ВАЖНОЕ ПРЕОБРАЗОВАНИЕ. Все делается на данных расстояния между нажатиями, но если заменить "between" на "time_key" тоже нужно проверить"""

dataset_work=dataset.groupby(['id','pair'])['between'].agg(list).reset_index()
dataset_work.columns=['id',	'pair',	'between']
dataset_work.head()

dataset_work.to_csv('dataset_work.csv')

"""## Подготовка данных

## тест гипотез
"""

id_list=dataset_work['id'].unique()
test_list=[]
for id_1 in range(len(id_list)-1):
#for id_1 in range(1,5):
  id_first=id_list[id_1]
  for id_2 in range(id_1+1,len(id_list)):
#  for id_2 in range(id_1+1,5):
#  for id_2 in range(16,29):
    id_second=id_list[id_2]  
    print(id_first,id_second)
    p1=set(dataset_work[dataset_work['id']==id_first]['pair'].values)    
#    print(p1)
    p2=set(dataset_work[dataset_work['id']==id_second]['pair'].values)
#    print(p2)
    pair_all=list(p1&p2)#[:100]
    data_1=dataset_work[(dataset_work['id']==id_first)][['pair','between']]#.values
    data_2=dataset_work[(dataset_work['id']==id_second)][['pair','between']]#.values      
    test_list=def_boot_cos(data_1,data_2,pair_all,id_first,id_second,test_list)

test_list

data_rez=pd.DataFrame(test_list,columns=['id_1', 'id_2' ,'pair_count','p-value','median_cos'])
data_rez.to_csv('data_rez.csv')

data_rez_300.to_csv('data_rez_300.csv')

"""## функция расчета с группировкой по юзерам"""

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