from django.contrib import admin
from keystroke.models import *
# Register your models here.

class PostAdmin(admin.ModelAdmin):
    list_display = ('text', 'id', 'pure_data')
    search_fields = ('text', 'id', 'pure_data')
    fields = ('text', 'user_post', 'pure_data')


admin.site.register(Post, PostAdmin) 
admin.site.register(User)
