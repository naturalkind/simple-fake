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
import csv
import sqlite3

import io
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm

plt.style.use('ggplot')


class UserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username',)

# выбираю post с самым коротким текстом
#def gettext(request):
#    print ("GETTEXT", request.user.pk)
#    posts = list(Post.objects.filter(user_post__id=request.user.pk))
#    t_l = []
#    for p in posts:
#        t_l.append(len(p.text))
#    to_client = posts[t_l.index(min(t_l))]
#    return JsonResponse({"text":to_client.text, "id_post":to_client.id})

# выбираю любой текст
def gettext(request):
    json_data = json.loads(request.body)
    #print ("GETTEXT", request.user.pk, json_data)
    post = Post.objects.get(pk=json_data["post_id"])
    return JsonResponse({"text":post.text, "id_post":post.id})



def user_page(request, user_id):
    #print (user_id)
    posts = list(Post.objects.filter(user_post__id=user_id))
    args = {}
    args['username'] = auth.get_user(request) 
    args['posts'] = posts
    args.update(csrf(request))   
    return render(request, 'user_page.html', args)

def quit(request):
    auth.logout(request)
    return redirect("/")

#@login_required       
def mainpage(request):
    args = {}
    args['username'] = auth.get_user(request)
    args.update(csrf(request))
    if request.method == 'GET':
        return render(request, 'index.html', args)

def keystroke(request):
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
    return render(request, 'keystroke.html', args)

def registration(request):
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
    args['form'] = UserForm()
    #print (request, request.POST, request.method)
    if request.POST:
        newuser_form = UserForm(request.POST)
        #print (newuser_form.is_valid())
        error_str = f"<html><body>{newuser_form.errors}</body></html>"
        print ("REGISTER", request.POST)
        if newuser_form.is_valid():
#            path = str(uuid.uuid4())[:12]
#            if not os.path.exists(f"media/data_image/{path}"):
#                os.makedirs(f"media/data_image/{path}")
#            
#            new_author = newuser_form.save(commit=False)
#            new_author.path_data = path
#            
#            new_author.save()
#            newuser = auth.authenticate(username=newuser_form.cleaned_data['username'],
#                                        password=newuser_form.cleaned_data['password2'],
#                                        )
#            auth.login(request, newuser)
#            return redirect('/registrationend')
            request.session['registration'] = request.POST
            return HttpResponseRedirect('/registrationend')
        else:
            args['form'] = newuser_form
    return render(request, 'registration.html', args)

def registrationend(request):
    registration = request.session.get('registration')
    try:
        json_data = json.loads(request.body)
        if registration:
            newuser_form = UserForm(registration)
            if newuser_form.is_valid():
                path = str(uuid.uuid4())[:12]
                
                if not os.path.exists(f"media/data_image/{path}"):
                    os.makedirs(f"media/data_image/{path}")
                
                new_author = newuser_form.save(commit=False)
                new_author.path_data = path
                
                new_author.save()
                newuser = auth.authenticate(username=newuser_form.cleaned_data['username'],
                                            password=newuser_form.cleaned_data['password2'],
                                            )
                auth.login(request, newuser)
                
                T = json_data["text"].replace('\xa0', ' ').replace("\n\n", " ").replace("\n", " ").lower()
                post = Post()
                post.pure_data = json_data["KEYPRESS"]
                post.text = T
                post.user_post = newuser
                post.save()
                #return redirect('/')
                #print ("REGISTEREND", request.POST, registration, json_data)
    except:
        pass
    return render(request, 'createpost.html', registration)

def login(request):
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
    #print (request, request.POST)
    if request.POST:
        username = request.POST.get('username','')
        password = request.POST.get('password','')
        user = auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            args['login_error']= 'Пользователь не найден'
            return render(request, 'login.html',args)
    else:
        return render(request, 'login.html', args)

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


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

    
def post(request, post):
    if request.user.is_authenticated:
        
        try:
            post_id = Post.objects.get(id=post)
        except ObjectDoesNotExist:
            return HttpResponse("Больше не существует")        
        T = post_id.text.lower().replace("\n", "")
        #json_data = json.dumps(post_id.pure_data)
        print (post, request.user.username, post_id.pure_data)#, json_data)
        #print (T)
        div_temp = f"<div id='full_nameuser'>{post_id.user_post.username}</div><div id='full_text'>{T}</div><br><table><tbody>"       
        np_zeros = np.zeros((len(combination), 2)) 
        for ih, h in enumerate(combination):
            idx = indices(T, h)
            if idx != []:
                #print (f"INDICES -> {h} <-------------------", idx)
                temp_ls = []
                for k in idx:
                    temp_ls.append(post_id.pure_data[k+1]["time_keydown"]-post_id.pure_data[k]["time_keyup"])
                np_zeros[ih, 0] = np.median(np.array(temp_ls))
                np_zeros[ih, 1] = T.count(h)
                div_temp += f"<tr><td>{h}</td><td>{T.count(h)}</td><td>{np.median(np.array(temp_ls))}</td></tr>"
                #print ("MDEIANA", np.median(np.array(temp_ls)))
        div_temp += "</tbody></table>"
        return HttpResponse(div_temp)
        
#        return JsonResponse({"user":f'{request.user.username}',
#                             "data": post_id.pure_data,
#                             "text": post_id.text})
    else:
        return JsonResponse({"answer":'registration done'})
#        return JsonResponse({"answer":'registration done'})      

def alldata(request):
    post = Post.objects.all()
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
    args['post'] = post
    return render(request, 'posts.html', args)
    

# новая версия
def cratealldata(request):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor.execute("select * from keystroke_post")
#    if not os.path.exists(f"media/data_image/{request.user.path_data}"):
#        os.makedirs(f"media/data_image/{request.user.path_data}")
    with open(f"media/data_image/out.csv", 'w', newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description]) 
        csv_writer.writerows(cursor)
    conn.close()
    return JsonResponse({"answer":f'/media/data_image/out.csv'})
    
# предыдущая версия    
#def cratealldata(request):
#    users = User.objects.all()
#    all_user_m = []
#    all_user_c = []
#    for u in users:
#        posts = Post.objects.filter(user_post=u)
#        #print (u, posts)
#        list_post_m = []
#        
#        np_zeros2 = np.zeros((len(combination), 1)) #len(control_text), 
#        for pp in posts:
#            T = pp.text.lower().replace("\n", "")
#            np_zeros1 = np.zeros((len(combination), 1)) #len(control_text), 
#            for ih, h in enumerate(combination):
#                idx = indices(T, h)
#                if idx != []:
#                    temp_ls = []
#                    for k in idx:
#                        temp_ls.append(pp.pure_data[k+1]["time_keydown"]-pp.pure_data[k]["time_keyup"])
#                    np_zeros1[ih, 0] = np.median(np.array(temp_ls))
#                    np_zeros2[ih, 0] += T.count(h)
#            list_post_m.append(np_zeros1)
#          
#        if len(posts) > 0:  
#            all_user_c.append(np_zeros2)
#        
#        if len(list_post_m) > 1:
#            m_post_user = np.median(np.array(list_post_m), axis=0)
#            all_user_m.append(m_post_user)
#        elif len(list_post_m) != 0:
#            result = np.array(list_post_m).reshape((-1, 1))
#            all_user_m.append(result)
#            print (result.shape)            
##            all_user_m.append(m_post_user)
#            #print (m_post_user)
#            #print (u, len(list_post_m), np.array(list_post_m).shape, m_post_user.shape)
#    #print (np.array(all_user_m).shape)
#    
#    AAA1 = np.array(all_user_c)[:,:,0]
#    dataset1 = pd.DataFrame(data=AAA1[0:,0:], index=[i for i in range(AAA1.shape[0])], columns=combination )
#    #dataset1[combination] = dataset1[combination].replace(['0', 0], np.nan)
#    dataset1.to_csv(f'media/data_image/{request.user.path_data}/key_count.csv')
#    print (dataset1)
#    
#    
#    AAA = np.array(all_user_m)[:,:,0]
#    dataset = pd.DataFrame(data=AAA[0:,0:], index=[i for i in range(AAA.shape[0])], columns=combination)
#    #dataset[combination] = dataset[combination].replace(['0', 0], np.nan)
#    dataset.to_csv(f"media/data_image/{request.user.path_data}/key_median.csv")
#    print(dataset)
#    
#    return JsonResponse({"answer":f'/media/data_image/{request.user.path_data}/key_median.csv',
#                         "answer_count": f'/media/data_image/{request.user.path_data}/key_count.csv'})
#    
    
    
   

