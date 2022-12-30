from django.shortcuts import render, redirect, HttpResponseRedirect
from django.template.context_processors import csrf
from django.contrib.auth import authenticate, login
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required 
from django.contrib import auth
from django.http import HttpResponse, Http404, JsonResponse
from keystroke.models import *
import os
import uuid
import json
import numpy as np
import pandas as pd
from pandas.core.groupby import groupby
from scipy.stats import norm
import matplotlib.pyplot as plt
# обработка нажатий

alf_ = ord('а')
abc_ = ''.join([chr(i) for i in range(alf_, alf_+32)])
abc_ += " ,."
#abc_ = string.ascii_lowercase + " ,."
combination = ["ст", "то", "но", "на", "по", "ен", "ни", "не", "ко", "ра", "ов", "ро", "го", "ал",
               "пр", "ли", "ре", "ос", "во", "ка", "ер", "от", "ол", "ор", "та", "ва", "ел", "ть",
               "ет", "ом", "те", "ло", "од", "ла", "ан", "ле", "ве", "де", "ри", "ес", "ат", "ог",
               "ль", "он", "ны", "за", "ти", "ит", "ск", "ил", "да", "ой", "ем", "ак", "ме", "ас",
               "ин", "об", "до", "че", "мо", "ся", "ки", "ми", "се", "тр", "же", "ам", "со", "аз",
               "нн", "ед", "ис", "ав", "им", "ви", "тв", "ар", "бы", "ма", "ие", "ру", "ег", "бо",
               "сл", "из", "ди", "чт", "вы", "вс", "ей", "ия", "пе", "ик", "ив", "сь", "ое", "их",
               "ча", "ну", "мы"] # 101   

# 1000 - знаков для статистики


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def sort_col(df):
    cols = df.columns.tolist()[:]

    df_np = df.to_numpy()[:,:]

    print (df_np.shape, len(cols))

    G = []
    for u in range(df_np.shape[0]):
        S = np.sort(df_np[u])
        AS = np.argsort(df_np[u])
        #print (S[::-1], AS[::-1])
        new_cols = []
        for k in AS:
            new_cols.append(cols[k])
            #print (k, cols[k])
        G.append(S[::-1])

    AAA1 = np.array(G)
    #print (AAA1.shape)    

    dataset = pd.DataFrame(data=AAA1[0:,0:], index=[i for i in range(AAA1.shape[0])], columns=new_cols)
    dataset[new_cols] = dataset[new_cols].replace(['0', 0], np.nan)

    return dataset
    # сохранить
    #dataset1.to_csv(f'key_count.csv')
    #dataset.to_csv(f'key_median.csv')
    #print (dataset1)

        
def cratealldata():
    users = User.objects.all()
    all_user_m = []
    all_user_c = []
    for u in users:
        posts = Post.objects.filter(user_post=u)
        print (u, posts)
        list_post_m = []
        
        list_data_all=[]
        if len(posts)>0:
            for pp in posts:
                T = pp.text.lower().replace("\n", "")
                LS = pp.pure_data
                check_T = ""
                for o in LS:
                    check_T += o["key_name"]  
                #print (T==check_T, len(T)) 
                np_zeros1 = np.zeros((len(combination), 1)) 
                
                #list_data_all=[]
                for ih, h in enumerate(combination):
                    idx = indices(T, h)
                    if idx != []:
                        for k in idx:
                          list_data_line=[]
                          t_up = pp.pure_data[k]["time_keyup"]
                          t_dw = pp.pure_data[k+1]["time_keydown"]
                          list_data_line.append(h)
                          list_data_line.append(len(idx))
                          list_data_line.append(pp.pure_data[k]["key_name"])
                          list_data_line.append(pp.pure_data[k+1]["key_name"])
                          list_data_line.append(t_dw-t_up)
                          list_data_line.append(pp.pure_data[k+1]["time_keydown"])
                          list_data_line.append(pp.pure_data[k]["time_keydown"])
                          list_data_all.append(list_data_line)
                        
                #print ("END..........")
            dataset_l=pd.DataFrame(list_data_all,
                                   columns=['pair','pair_count','p_1','p_2','time','t_start','t_end'])
            #print (dataset_l)
    #            reset_index() - Сбросить индекс или его уровень.
    #            Сбросьте индекс DataFrame и используйте вместо него индекс по умолчанию. Если у DataFrame есть MultiIndex, этот метод может удалить один или несколько уровней.

            dataset_median=dataset_l.groupby('pair')['time'].median().reset_index()
            #print (dataset_median)

            #pd.unique
            #возвращает уникальные значения из входного массива, столбца или индекса DataFrame.
#            pair_name=dataset_l['pair'].unique()

            #dataset_l[dataset_l['pair']==pair_name[1]]['time']
            
            dataset_l = dataset_l.loc[dataset_l['pair_count'] > 10]
            pair_name=dataset_l['pair'].unique()
            print (pair_name)
            #print (pair_name)
#            print (dataset_l[dataset_l['pair']=="то"], dataset_l[dataset_l['pair']=="то"].shape)#['time']
#            dataset_l[dataset_l['pair']=="то"]['time'].hist(bins=100)
            for o in pair_name:
                #print (dataset_l[dataset_l['pair']==o])
                ax = dataset_l[dataset_l['pair']==o]['time'].hist(bins=100)  
                plt.title(o)
                plt.ylabel('quantity')
                plt.xlabel('time ms')
                if not os.path.exists(f"{u.username}"):
                    os.makedirs(f"{u.username}")
                plt.savefig(f'{u.username}/{o}.pdf')
                plt.clf()
                #plt.show()          
            
            
    
#cratealldata()
import csv
import sqlite3
def db_to_csv():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())
    
    cursor.execute("select * from keystroke_post")
    with open("out.csv", 'w', newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description]) 
        csv_writer.writerows(cursor)
    conn.close()

#db_to_csv()

def open_csv_db():
    df = pd.read_csv("out.csv")
    #print (df)
    grouped_df = df.groupby('user_post_id')
    for key, item in grouped_df:
        user_datas = grouped_df.get_group(key)
        #print(key, item)
        #print(user_datas, "\n")
        list_1 = user_datas["pure_data"].to_json() #pd.json_normalize(user_datas["pure_data"])
        list_2 = user_datas["text"].tolist()
        convet_js = json.loads(list_1)
        print (len(convet_js))
        list_data_all=[]
        for ix, i in enumerate(convet_js):
            posts = json.loads(convet_js[i])
            #print (_js)
            T = list_2[ix].replace("\n", "")
            check_T = ""
            if len(posts)>0:
                for pp in posts:
                    check_T += pp["key_name"]  
                    for ih, h in enumerate(combination):
                        idx = indices(T, h)
                        if idx != []:
                            for k in idx:
                              list_data_line=[]
                              t_up = posts[k]["time_keyup"]
                              t_dw = posts[k+1]["time_keydown"]
                              list_data_line.append(h)
                              list_data_line.append(len(idx))
                              list_data_line.append(posts[k]["key_name"])
                              list_data_line.append(posts[k+1]["key_name"])
                              list_data_line.append(t_dw-t_up)
                              list_data_line.append(posts[k+1]["time_keydown"])
                              list_data_line.append(posts[k]["time_keydown"])
                              list_data_all.append(list_data_line)                      
                    
        dataset_l=pd.DataFrame(list_data_all,
                   columns=['pair','pair_count','p_1','p_2','time','t_start','t_end'])
        
        print (dataset_l)        
        print (T==check_T, len(T)) 
        
        
        #dataset_l = dataset_l.loc[dataset_l['pair_count'] > 5]
        pair_name=dataset_l['pair'].unique()
        print (pair_name)
        for o in pair_name:
            ax = dataset_l[dataset_l['pair']==o]['time'].hist(bins=100)  
            plt.title(o)
            plt.ylabel('quantity')
            plt.xlabel('time ms')
#            if not os.path.exists(f"datas/user_id{key}"):
#                os.makedirs(f"datas/user_id{key}")
#            plt.savefig(f'datas/user_id{key}/{o}.pdf')
#            plt.clf()                   
      
            #print (list_2[ix], "------------------------------------>\n")
        
open_csv_db()       

#просто раньше использовали рекуррентные сети, которые были слабым звено в скорости, выполняя сигналы линейно, последовательно. Впринципе это ограничение убирается архитектурой фон неймана, общая концепция работы пк и мозга, но множество отдельно обученных сетей, контролируемые осп агентом, трудней контролировать и настравать отдельно. А так кодируем все в одно векторное пространство, и подключаем осп. Как вариант наш слепок работает с языковой моделью на частотах недостежимих физическому мозгу
#, ai sapiens



