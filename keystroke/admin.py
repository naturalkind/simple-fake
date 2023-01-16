from django.contrib import admin
from keystroke.models import *
# Register your models here.

class PostAdmin(admin.ModelAdmin):
    list_display = ('text', 'id', 'user_post')
    search_fields = ('text', 'id', 'user_post')
    fields = ('text', 'user_post', 'pure_data', 'status')


admin.site.register(Post, PostAdmin) 
admin.site.register(User)
