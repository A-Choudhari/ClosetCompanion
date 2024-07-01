from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    is_active = models.BooleanField(default=False)
    verification_code = models.CharField(max_length=6, blank=True)
    gender = models.IntegerField(default=0)


class Style(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="style_user")
    cloth_size = models.CharField(max_length=10, blank=True)
    cloth_style = models.CharField(max_length=1024, blank=True)
    cloth_item = models.CharField(max_length=1024, blank=True)

    def __str__(self):
        return f"{self.user} is a {self.cloth_item}"


class Closet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="closet_user")
    cloth_size = models.CharField(max_length=10, blank=True)
    cloth_item = models.CharField(max_length=1024, blank=True)
    cloth_color = models.CharField(max_length=1024, blank=True)
    cloth_category = models.CharField(max_length=1024, blank=True)
    image = models.ImageField(upload_to='images/')  # 'upload_to' specifies the upload directory


class Watchlist(models.Model):
    link = models.CharField(max_length=8096, blank=True)
    product_info = models.CharField(max_length=1024, blank=True)
    image = models.CharField(max_length=4096, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="watchlist_user")

