# -*- coding: utf-8 -*-
"""
Почерк тест v4

"""
import numpy as np
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
        
#   Визуализация

#    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
#    for bar in bars:
#       if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
#           bar.set_facecolor('red')
#       else: 
#           bar.set_facecolor('grey')
#           bar.set_edgecolor('black')

#    plt.style.use('ggplot')
#    plt.vlines(quants, ymin=0, ymax=50, linestyle='--')
#    plt.xlabel('boot_data')
#    plt.ylabel('frequency')
#    plt.title("Histogram of boot_data")
#    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}

def open_csv_db():
#    url = "https://raw.githubusercontent.com/naturalkind/simple-fake/orm_django/ormapp/media/images/out.csv"
    url ="https://cyber.b24chat.com/media/data_image/out.csv"
    D = requests.get(url).content
    df = pd.read_csv(io.StringIO(D.decode('utf-8')))
    return df

#df = pd.read_csv('/content/out.csv')#.decode('utf-8')))    
df = open_csv_db()
print (df.head())

def def_boot(series_1, series_2, pair_b, id_1, id_2, test_list_all):
    test_list=[]
    booted_data = get_bootstrap(series_1, 
                                series_2, # числовые значения второй выборки
                                boot_it = 1000, # количество бутстрэп-подвыборок
                                statistic = np.median, # интересующая нас статистика
                                bootstrap_conf_level = 0.95 # уровень значимости
                                )
    test_list.append(id_1)
    test_list.append(id_2)
    test_list.append(pair_b)
    test_list.append(booted_data["p_value"])
    test_list_all.append(test_list)
    print(id_1, id_2, pair_b, 'p_value=', booted_data["p_value"])
    return test_list_all

def time_pair(JS,id):
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
        time_JS.append(t1+t2)      
        time_JS.append(t21-t12)    
        time_JS.append(pair)
        time_a.append(time_JS)
    return time_a

#df.to_csv('df.csv')

""" 
Сравнение

"""

id_list = df['id'].unique()
text_all = [0.0, 0.0, 0.0, 0.0]
for i in range(0,len(id_list)):
    first_t=id_list[i]
    JS=eval(df[df['id']==first_t]['pure_data'][i])  
    time=(time_pair(JS, first_t))
    text_all=np.vstack([text_all, time])
    #print(text_all)
dataset=pd.DataFrame(text_all[1:],columns=['id', 'time_key', 'between','pair'])

print (dataset.head())

data_count=dataset.groupby(['id','pair'])['time_key'].count().reset_index()

#data_count.to_csv('data_count.csv')

"""
Подготовка данных

"""

#from pandas.core.groupby import groupby
dataset_median=dataset.groupby(['id', 'pair'])['time_key','between'].median(numeric_only=False).reset_index()
print (dataset_median)

#dataset_median.to_excel('dataset_median_2.xls')

dataset_time=dataset.groupby(['id','pair'])['between'].agg(list).reset_index()
print (dataset_time)

#dataset_time.to_csv('dataset_time.csv')

"""
тест гипотез

"""

id_list=dataset['id'].unique()
test_list=[]
#for id_1 in range(len(id_list)-1):
for id_1 in range(1,5):
    id_first = id_list[id_1]
    #  for id_2 in range(id_1+1,len(id_list)):
    #  for id_2 in range(id_1+1,5):
    for id_2 in range(16,29):

        id_second = id_list[id_2]  
    #    print(id_1,id_2)
        p1=set(dataset[dataset['id']==id_first]['pair'].values)
#        print(p1)
        p2=set(dataset[dataset['id']==id_second]['pair'].values)
#        print(p2)
        pair_all=list(p1&p2)#[:100]
#        print (">>>>>>>>>>", pair_all)
        for pair_b in pair_all:
#      print(id_1, id_2, pair_b)         
            series_1=dataset[(dataset['id'] == id_first) & (dataset['pair'] == pair_b)]['between']#.values
            series_2=dataset[(dataset['id'] == id_second) & (dataset['pair'] == pair_b)]['between']#.values
            if len(series_1)>3 and len(series_2)>3:                                                     
                test_list= def_boot(series_1.astype("float"), 
                                    series_2.astype("float"), 
                                    pair_b, 
                                    id_first, 
                                    id_second, 
                                    test_list)

data_rez_e = pd.DataFrame(test_list, columns=['id_1', 'id_2' ,'pair','p-value'])
#data_rez.to_csv('data_rez.csv')

#data_rez_e.to_csv('data_rez_i.csv')

id_list = dataset['id'].unique()
test_list = []
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
            series_1=dataset[(dataset['id'] == id_first) & (dataset['pair'] == pair_b)]['time_key']#.values
            series_2=dataset[(dataset['id'] == id_second) & (dataset['pair'] == pair_b)]['time_key']#.values
            if len(series_1)>3 and len(series_2)>3:                                                     
                test_list_key = def_boot(series_1.astype("float"), 
                                         series_2.astype("float"),
                                         pair_b,
                                         id_first,
                                         id_second,
                                         test_list)

time_key_e = pd.DataFrame(test_list, columns=['id_1', 'id_2' ,'pair','p-value'])
print (time_key_e)
#time_key_e.to_csv('time_key_i.csv')
#['ис', 'ер', ' п', 'мо', 'ти', 'а ', 'та', 'от', 'о ', 'я ', 'ес', 'ет', 'по', 'ка', 'аз', 'оп', ' в', 'ус', 'за', 'сн', 'ст', 'ро', 'зм', 'и ', 'в ', 'то', 'т ', 'м ', 'си', 'на', 'не', ' н', ', ', 'дн', 'ор', 'ов', 'сп', 'ем', 'е ', 'ме', 'но', 'ри', 'ом', 'ос', 'я,', 'ат', 'ся', 'те', 'ви', 'ел', ' с', ' б', ' и', 'бо', ' у']
