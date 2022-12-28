from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from datetime import datetime
import base64
import re
import uuid, os

# Create your models here.

class User(AbstractUser):
    image_user = models.TextField(max_length=200, default="oneProf.png", verbose_name='Название картинки', blank=True)
    path_data = models.TextField(max_length=200, default="", verbose_name='Название каталога', blank=True)
    color = models.TextField(max_length=200, default="#507299", verbose_name='Цвет шрифта', blank=False)
    class Meta(AbstractUser.Meta):
        swappable = 'AUTH_USER_MODEL'
        
    def natural_key(self):
        return (self.image_user, self.path_data, self.id, self.username)
    
    def save(self, *args, **kwargs):
        if self.path_data == "":
            self.path_data = str(uuid.uuid4())[:12]
            if not os.path.exists(f"media/data_image/{self.path_data}"):
                os.makedirs(f"media/data_image/{self.path_data}")
        return super(User, self).save(*args, **kwargs)


class Post(models.Model):
    text = models.TextField(max_length=999999, default="", verbose_name='Текст', blank=True)
    pure_data = models.TextField(max_length=999999, default="", verbose_name='Данные', blank=True)
    date_post = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    user_post = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='us_post', default="", on_delete=models.CASCADE)

    def __unicode__(self):
            return u'name: %s , id: %s' % (self.text, self.id)



