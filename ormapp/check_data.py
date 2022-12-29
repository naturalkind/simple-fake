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

control_text = """повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим,
как будет изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в
которых работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
работает оператор. желательно, чтобы сначала работал в одних и тех же условиях."""
control_text = control_text.replace("\n", "")   
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
        
        np_zeros2 = np.zeros((len(combination), 1)) #len(control_text), 
        char_l = 0
        for pp in posts:
            T = pp.text.lower().replace("\n", "")
            LS = pp.pure_data
            check_T = ""
            for o in LS:
                check_T += o["key_name"]  
                char_l += 1          
            #print (T==check_T, len(T)) #check_T, T==check_T, 
            np_zeros1 = np.zeros((len(combination), 1)) #len(control_text), 
            for ih, h in enumerate(combination):
                idx = indices(T, h)
                if idx != []:
                    temp_ls = []
                    for k in idx:
                        temp_ls.append(pp.pure_data[k+1]["time_keydown"]-pp.pure_data[k]["time_keyup"])
                    MeDi = np.median(np.array(temp_ls))
                    if float(MeDi)>0:
                        np_zeros1[ih, 0] = MeDi
                    else:
                        print (pp.pure_data[k+1]["key_name"], 
                               pp.pure_data[k]["key_name"], 
                               MeDi,
                               pp.pure_data[k+1]["time_keydown"] - pp.pure_data[k]["time_keyup"],
                               pp.pure_data[k+1]["time_keydown"] - pp.pure_data[k]["time_keydown"])
                    np_zeros2[ih, 0] += T.count(h)
            list_post_m.append(np_zeros1)
            #print ("END..........")
          
        if len(posts) > 0:  
            all_user_c.append(np_zeros2)
        
        if len(list_post_m) > 1:
            m_post_user = np.median(np.array(list_post_m), axis=0)
            all_user_m.append(m_post_user)
            
        elif len(list_post_m) != 0:
            result = np.array(list_post_m).reshape((-1, 1))
            all_user_m.append(result)
        
        print (len(posts), len(list_post_m), char_l)
            
#            all_user_m.append(m_post_user)
            #print (m_post_user)
            #print (u, len(list_post_m), np.array(list_post_m).shape, m_post_user.shape)
    #print (np.array(all_user_m).shape)
    
    AAA = np.array(all_user_m)[:,:,0]
    dataset = pd.DataFrame(data=AAA[0:,0:], index=[i for i in range(AAA.shape[0])], columns=combination)
    dataset = sort_col(dataset)
    dataset[combination] = dataset[combination].replace(['0', 0], np.nan)
    dataset.to_csv(f'_count.csv')
    print(dataset)
    
    
    AAA1 = np.array(all_user_c)[:,:,0]
    dataset1 = pd.DataFrame(data=AAA1[0:,0:], index=[i for i in range(AAA1.shape[0])], columns=combination )
    dataset1 = sort_col(dataset1)
    dataset1[combination] = dataset1[combination].replace(['0', 0], np.nan)
    dataset.to_csv(f"_median.csv")
    print (dataset1)    
    
cratealldata()

