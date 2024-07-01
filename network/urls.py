
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("register", views.register, name="register"),
    path("verification/<int:user_id>", views.email_verification, name="verification"),
    path("new_password/<int:user_id>", views.new_password, name="new_password"),
    path("user_profile", views.user_profile, name="user_profile"),
    path("user_closet", views.user_closet, name="user_closet"),
    path("confirm_password/<int:user_id>", views.confirm_password, name="confirm_password"),
    path("email_address", views.email_address, name="email_address"),
    path("watchlist", views.watchlist, name="watchlist"),
    path("user_data", views.user_data, name="user_data"),
    path("ar/<str:image_url>", views.arFunc, name="ar"),
    path("arView/<str:image_url>", views.live_capture_view, name="arView"),
]
