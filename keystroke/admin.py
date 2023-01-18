from django.contrib import admin
from keystroke.models import *
# Register your models here.

class PostAdmin(admin.ModelAdmin):
    list_display = ('text', 'id', 'user_post')
    search_fields = ('text', 'id', 'user_post')
    fields = ('text', 'user_post', 'pure_data', 'status', 'text_to_test')

class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'id')
    search_fields = ('username', 'id')


admin.site.register(Post, PostAdmin) 
#admin.site.register(User)
admin.site.register(User, UserAdmin)
