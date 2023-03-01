from django.shortcuts import render, redirect, HttpResponseRedirect
from django.template.context_processors import csrf
from django.contrib.auth import authenticate, login
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required 
from django.contrib import auth
from django.http import HttpResponse, Http404, JsonResponse
from django.template.loader import render_to_string
from django.template import loader
from django.template import Template, Context
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
                post.status = "y"
                post.text = T
                post.user_post = newuser
                post.save()
                #return redirect('/')
                #print ("REGISTEREND", request.POST, registration, json_data)
    except:
        pass
    return render(request, 'createpost.html', registration)


#def login(request):
#    args = {}
#    args.update(csrf(request))
#    args['username'] = auth.get_user(request)
#    #print (request, request.POST)
#    if request.POST:
#        username = request.POST.get('username','')
#        password = request.POST.get('password','')
#        user = auth.authenticate(username=username,password=password)
#        if user is not None:
#            auth.login(request, user)
#            return redirect('/')
#        else:
#            args['login_error']= 'Пользователь не найден'
#            return render(request, 'login.html',args)
#    else:
#        return render(request, 'login.html', args)

# новая функция бутстреп для косинуса угла между векторами


# V4 обнавленная
def get_bootstrap(data_column_1, # числовые значения первой выборки
                  data_column_2, # числовые значения второй выборки
                  boot_it = 1000, # количество бутстрэп-подвыборок
                  statistic = np.mean, # интересующая нас статистика
                  bootstrap_conf_level = 0.95 # уровень значимости
                  ):
                  
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


def time_pair(JS, id):
  time_a=[]
  for i in range(len(JS)-1):
    time_JS=[]
    pair=JS[i]['key_name']+JS[i+1]['key_name']
    t11= JS[i]['time_keydown']
    t21= JS[i+1]['time_keydown']
    time_JS.append(id)
    time_JS.append(0)      
    time_JS.append(t21-t11)    
    time_JS.append(pair)
    time_a.append(time_JS)
  dataset = pd.DataFrame(time_a, columns=['id',  'time_key', 'time', 'pair'])
  return dataset

# вход новая версия
def login(request):
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
    
    #print (request, request.POST)
    if request.POST:
        username = request.POST.get('username','')
        password = request.POST.get('password','')
        user = auth.authenticate(username=username,password=password)
        print (user)
        if user is not None:
            request.session['login'] = request.POST
            return HttpResponseRedirect('/loginend')
        else:
            t = loader.get_template('login.html')
            template = Template('{%extends "' + "base.html" + '"%} ...'+t.template.source)
            context = Context(args)
            result = template.render(context)
            return HttpResponse(result)
            
    else:
        args['base'] = ""
        return render(request, 'login.html', args)


#1/(1+е(-ln(p-value/0.05)). 
#А после упрощающих преобразований x/(x+0.05)
def sigmoid(z):
    z = np.log((z/0.05))
    return 1/(1 + np.exp(-z))
    
   

def loginend(request):
    login = request.session.get('login')
    if request.body:
        json_data = json.loads(request.body)
        if login:
            username = login['username']
            password = login['password']
            user = auth.authenticate(username=username,password=password)
            if user is not None:
                #print ("......", login, json_data)
                # данные полученные для проверки
                T0 = json_data['text']
                dt0 = time_pair(json_data["KEYPRESS"], user.id)
                p0 = set(dt0['pair'].values)  

                posts = list(Post.objects.filter(status="y"))
#                for_all_data = {}
                test_list = []
                div_out = ""
                for post_ in posts:
                    T1 = post_.text
                    dt1 = time_pair(post_.pure_data, post_.user_post.id)
                    p1 = set(dt1['pair'].values)
                    pair_all = list(p0 & p1)
                    #print ("ONE----->", pair_all, len(T0), len(T1))
                    
                    for pair_b in pair_all:
                            series_1=dt0[dt0['pair'] == pair_b]['time']#.values
                            series_2=dt1[dt1['pair'] == pair_b]['time']#.values
                            if len(series_1) > 3 and len(series_2) > 3:
                                test_list = def_boot(series_1.astype("float"),
                                                     series_2.astype("float"),
                                                     pair_b,
                                                     user.id,
                                                     post_.user_post.id,
                                                     test_list)
                    
#                    sig = sigmoid(test_list[0][1])
#                    print ("TWO----->", test_list[0], sig)
#                    for_all_data.append([post_.user_post.username, sig])  
#                div_out += f"REQUEST USER: {user.username}, {user.id}"
#                
#                # Визуализация
                dataset_ = pd.DataFrame(test_list, columns=['id_1', 'id_2' ,'pair', 'value'])
                dataset_['sig']=dataset_['value']/(dataset_['value']+0.05)
                print (dataset_)
#                fig, ax = plt.subplots(figsize=(9,6))
#                g = sns.barplot(x='users_id', y='value', data=dataset_, ci=95, ax=ax)
#                ax.set_title("Histogram of p-value users")
#                dataset_["value"] = dataset_["value"].apply(lambda x: round(x, 4))
#                for index, data in enumerate(dataset_["value"].tolist()):
#                    plt.text(x = index-.25, y = data, s = f"{data}")
#                plt.tight_layout()

##                ax.bar_label(dataset_["value"].tolist()) matplotlib v3.4+
#                figdata = io.BytesIO()
#                fig.savefig(figdata, format='png')
#                figdata_png = base64.b64encode(figdata.getvalue()).decode()
#                div_out += f'<img id="bar_p" src="data:image/png;base64, {figdata_png}"/>'
##                plt.show()                
#                
#                # аутентификация
#                auth.login(request, user)    
#                return JsonResponse({"user":f'{user.username}',
#                                     "html": div_out})
    return render(request, 'createpost_log.html', login)

###----------------------------------------------------------------------->

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
    
"""    
расскажем как мы делали работу которая должна показать, что человек все еще можеть быть первым в познаниях всего.
доказать своим примером что мы, люди, еще на что то способны, очень важно. не время говорить - человеческой системе мышления конец.

"""  
