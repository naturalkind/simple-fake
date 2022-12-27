from django.shortcuts import render, redirect, HttpResponseRedirect
from django.template.context_processors import csrf
from django.contrib.auth import authenticate, login
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.forms import UserCreationForm 
from django.contrib import auth
from django.http import HttpResponse, Http404, JsonResponse
from keystroke.models import *
import os
import uuid
import json
# Create your views here.


#def mainpage(request):
#    if request.method == 'GET':
#        return render(request, 'index.html')
from django.contrib.auth.decorators import login_required


class UserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username',)

@login_required       
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
    print (request, request.POST, request.method)
    if request.POST:
        newuser_form = UserForm(request.POST)
        print (newuser_form.is_valid())
        error_str = f"<html><body>{newuser_form.errors}</body></html>"
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
            return redirect('/')
        else:
            args['form'] = newuser_form
    return render(request, 'registration.html', args)

def login(request):
    args = {}
    args.update(csrf(request))
    args['username'] = auth.get_user(request)
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


#def registration(request):
#    json_data = json.loads(request.body)
#    print (request.method)
#    if request.method == "POST":
#        print (json_data)
#        new_user = User()
#        new_user.username = json_data["Name"] 
#        new_user.password = json_data["Pass"]  
#        new_user.save()
#        login(request, new_user)     
#        return redirect('/')  
    
    
def post(request, post):
    if request.user.is_authenticated:
        print (post, request.user.username)
        return JsonResponse({"user":f'{request.user.username}'})
    else:
        return JsonResponse({"answer":'registration done'})
#        return JsonResponse({"answer":'registration done'})      
#def enter(request):
#    username = request.POST['username']
#    password = request.POST['password']
#    user = authenticate(request, username=username, password=password)
#    if user is not None:
#        login(request, user)
