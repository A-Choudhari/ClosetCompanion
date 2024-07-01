from django.contrib import admin
from .models import User, Style, Closet, Watchlist
# Register your models here.

admin.site.register(User)
admin.site.register(Watchlist)
admin.site.register(Style)
admin.site.register(Closet)

