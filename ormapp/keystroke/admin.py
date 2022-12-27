from django.contrib import admin
from keystroke.models import *
# Register your models here.

class PostAdmin(admin.ModelAdmin):
    list_display = ('text', 'id',)
    search_fields = ('text', 'id',)
    fields = ('text', 'user_post')


admin.site.register(Post, PostAdmin) 
admin.site.register(User)
