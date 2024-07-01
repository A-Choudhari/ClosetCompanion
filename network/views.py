import json
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.http import StreamingHttpResponse
import threading
from PIL import Image
from .camera import VideoCamera
import numpy as np
from django.shortcuts import redirect
from django.core.paginator import Paginator
from .models import User, Style, Closet, Watchlist
import requests
from django.core.mail import send_mail, BadHeaderError
from django.core.exceptions import ObjectDoesNotExist
import secrets
from io import BytesIO
import sys
import string
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
import openai
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from keras import layers, models
#from keras import ImageDataGenerator
from .forms import ImageUploadForm
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import cv2
import cvlib as cv
import os
OpenposeDir = 'network/ildoonet-tf-pose-estimation'
pyopenpose_dir = os.path.join(OpenposeDir, 'build', 'python') # ex: '/content/openpose/build/python'
if pyopenpose_dir not in sys.path:
    sys.path.append(pyopenpose_dir)
#from openpose import pyopenpose as op


def index(request):
    return render(request, "network/index.html")


def user_data(request):
    if request.method == "GET":
        try:
            clothing = Style.objects.get(user=request.user)
            clothing_style = clothing.style
            clothing_item = clothing.item
        except:
            clothing_style = ""
            clothing_item = ""

        try:
            whatyouliked = Watchlist.objects.filter(user=request.user)
        except:
            whatyouliked = ""
        #weather = fetch_weather_for_location()
        #messages = [f"give 10 links to websites with clothing to wear in {weather} weather",
        #            f" similar to {clothing_style} as {clothing_item}"]
        #completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages={
        #"role": "user", "content": messages})
        #company_string = completion["choices"][0]["message"]
        #list_company = re.findall(r'\d+\.\s(.*?)\s', company_string)
        image_data = []
        list_company = ["macys", "amazon", "kohls", "jcpenny"]
        watchlist_items = Watchlist.objects.filter(user=request.user)
        for company in list_company:
            website_link = find_company_website(company + "men'stshirts")
            images = fetch_data(website_link, company)
            for image in images:
                for i in watchlist_items:
                    if image["file"] == i.image:
                        image["liked"] = True
                image_data.append(image)
        final_decison = rank_images(request, image_data)
        return JsonResponse(final_decison, safe=False)


def rank_images(request, images):
    try:
        train_data = Closet.objects.filter(user=request.user).image
    except:
        return images
    # Normalize pixel values to be between 0 and 1
    image_arrays = []
    for url in images:
        response = requests.get(url['file'])
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img) / 255.0
        image_arrays.append(np.array(img_array))
    # Define LeNet-5 model
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model with dummy labels (as labels are not provided)
    labels = np.random.randint(0, 2, size=(len(images)))  # Dummy labels (binary classification)
    training_data = np.array([item.get('image') / 255.0 for item in train_data])

    model.fit(training_data, labels, epochs=10)
    # Make predictions on the images
    predictions = model.predict(images)
    # Rank images based on predictions
    ranked_indices = np.argsort(predictions.flatten())[::-1]
    # Return the top-ranked images
    top_images = [images[i] for i in ranked_indices]
    return top_images


def find_company_website(company_name):
    webdriver_path = 'C:\Akshat\python\python projects\stockprediction_AI\chromedriver.exe'
    s = Service(executable_path=webdriver_path, args=['--headless'])
    driver = webdriver.Chrome(service=s)
    search_url = f"https://www.google.com/search?q={company_name}"
    # Navigate to the website
    driver.get(search_url)
    # Give the page some time to load

    # Get the page source using Selenium
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    items = soup.find_all('div', {'class': 'yuRUbf'})
    for each in items:
        try:
            a_tag = each.find('span').find('a')
            link = a_tag.get('href')
            return link
        except:
            continue


def watchlist(request):
    if request.method == "PUT":
        data = json.loads(request.body)
        if data.get("link") and data.get("product_info") and data.get("image") and data.get("liked"):
            link = data["link"]
            product_info = data["product_info"]
            image = data["image"]
            like = data["liked"]
            if like:
                new_item = Watchlist(user=request.user, link=link, product_info=product_info, image=image)
                new_item.save()
            else:
                query = Watchlist.objects.get(user=request.user, link=link, product_info=product_info, image=image)
                query.delete()

            return JsonResponse({"message": "Success"})
        else:
            return JsonResponse({"message": "Fail"})
    else:
        all_posts = Watchlist.objects.filter(user=request.user).reverse()
        p = Paginator(all_posts, 28)
        page_number = request.GET.get('page')
        page_obj = p.get_page(page_number)
        return render(request, "network/watchlist.html", {"page_obj": page_obj})


def fetch_data(url, website_name):
    webdriver_path = 'C:\Akshat\python\python projects\stockprediction_AI\chromedriver.exe'
    s = Service(webdriver_path, args=['--headless'])
    driver = webdriver.Chrome(service=s)
    try:
        driver.get(url)
    except:
        return []

    for _ in range(5):  # Adjust the number of steps as needed
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight/5);")
    # Get the page source using Selenium
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    # Find all image tags
    url_paths = []
    li_with_a_tags = soup.find_all('li')  # Adjust the class name as needed
    driver.quit()
    for li in li_with_a_tags:
        try:
            image = li.find('img')
            a = li.find('a')
            if image and a:
                image_url = image.get('src')
                a_link = a.get('href')
            else:
                continue
            #print(f"A_Tag:{a}   Image:{image}")
            img_src = [d["file"] for d in url_paths]
            if ("svg" in image_url) or ("gif" in image_url) or ("png" in image_url) or (image_url in img_src):
                continue
            product_info = li.get_text()
            if product_info:
                product_info = product_info.replace("\n", "").replace("\t", "")
                index_item = product_info.rindex("$")
                if index_item != -1:
                    if len(product_info) <= (index_item + 7):
                        combined_text = product_info
                    elif product_info[index_item + 6].isdigit():
                        combined_text = product_info[0: index_item + 7].replace("$", "\n$")
                    else:
                        combined_text = product_info[0: index_item + 6].replace("$", "\n$")
            else:
                combined_text = ""
            if not a_link.startswith(('http', 'https')):
                if website_name not in a_link[:len(a_link) // 2]:
                    link = f"https://www.{website_name}.com{a_link}"
                else:
                    link = f"https:{a_link}"
            else:
                link = a_link
            url_paths.append({"file": image_url, "link": link, "product_info": combined_text, "liked": False})
        except Exception as e:
            print(f"Error: {e}")
            continue
    return url_paths


def login_view(request):
    if request.method == "POST":
        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "network/login.html", {
                "message": "Invalid username and/or password."
            })
    else:
        return render(request, "network/login.html")


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        user_info = User.objects.all()
        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "network/register.html", {
                "message": "Passwords must match."
            })
        for user in user_info:
            if email == user.email:
                return render(request, "network/register.html", {
                    "message": "Email is already used"})
        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "network/register.html", {
                "message": "Username already taken."
            })

        return HttpResponseRedirect(reverse("verification", args=[User.objects.get(username=username, email=email).pk]))
    else:
        return render(request, "network/register.html")


def email_address(request):
    if request.method == "POST":
        email = request.POST["email_address"]
        try:
            user = User.objects.get(email=email)
            return HttpResponseRedirect(reverse("new_password", args=[User.objects.get(email=email).pk]))
        except IntegrityError:
            return render(request, "network/email_address.html", {"message": "Email not recognized"})
    else:
        return render(request, "network/email_address.html")


def email_verification(request, user_id):
    if request.method == "POST":
        user_code = ""
        user_verification = User.objects.get(pk=user_id)
        i = 1
        while i < 7:
            user_code += (request.POST[f"email_verification_code_{i}"])
            i += 1
        if user_verification.verification_code.lower() == user_code.lower():
            user_verification.is_active = True
            user_verification.verification_code = ""
            user_verification.save()
            login(request, user_verification)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "network/email_verification.html", {"message": "Verification Code was incorrect",
                                                                       "id": user_verification.pk})
    else:
        user_email = User.objects.get(pk=user_id)
        verification_code = generate_verification_code()
        try:
            message = (f'Dear User,\n\n'
                       f'We\'re thrilled to have you as a member of Your Website Name! '
                       f'To complete your registration and ensure the security of your account, '
                       f'please verify your email address by copying the code below:\n\n'
                       f'{verification_code}\n\n'
                       f'If you didn\'t request this email or if you have any concerns, please disregard it. '
                       f'Your account\'s security is essential to us, and no action will be taken without your verification.\n\n'
                       f'Once your email is verified, you\'ll be able to enjoy all the features and benefits of our website.\n\n'
                       f'Thank you for choosing Fashion! If you have any questions or need assistance, '
                       f'don\'t hesitate to contact our support team at support@yourwebsitename.com.\n\n'
                       f'Best regards,\n'
                       f'The Fashion Team')

            subject = "Email Verification"
            recipient_list = [user_email.email]
            from_email = "akshatc413@gmail.com"
            send_mail(subject, message, from_email, recipient_list)
            user_email.verification_code = verification_code
            user_email.save()
        except BadHeaderError:
            return HttpResponse('Invalid header found in the email.')
        return render(request, "network/email_verification.html", {"id": user_email.pk})


def new_password(request, user_id):
    if request.method == "POST":
        user_code = ""
        user_verification = User.objects.get(pk=user_id)
        i = 1
        while i < 7:
            user_code += (request.POST[f"password_verification_code_{i}"])
            i += 1
        if user_verification.verification_code.lower() == user_code.lower():
            return HttpResponseRedirect(reverse("confirm_password", args=[user_id]))
        else:
            return render(request, "network/password_verification.html", {"message": "Verification Code was incorrect"})
    else:
        user_email = User.objects.get(pk=user_id)
        verification_code = generate_verification_code()
        try:
            message = (f'Dear User,\n\n'
                       f'We\'re thrilled to have you as a member of Your Website Name! '
                       f'To complete your registration and ensure the security of your account, '
                       f'please verify your email address by copying the code below:\n\n'
                       f'{verification_code}\n\n'
                       f'If you didn\'t request this email or if you have any concerns, please disregard it. '
                       f'Your account\'s security is essential to us, and no action will be taken without your verification.\n\n'
                       f'Once your email is verified, you\'ll be able to enjoy all the features and benefits of our website.\n\n'
                       f'Thank you for choosing Fashion! If you have any questions or need assistance, '
                       f'don\'t hesitate to contact our support team at support@yourwebsitename.com.\n\n'
                       f'Best regards,\n'
                       f'The Fashion Team')

            subject = "Email Verification"
            recipient_list = [user_email.email]
            from_email = "akshatc413@gmail.com"
            send_mail(subject, message, from_email, recipient_list)
            user_email.verification_code = verification_code
            user_email.save()
        except BadHeaderError:
            return HttpResponse('Invalid header found in the email.')
        return render(request, "network/password_verification.html", {"user": user_id})


def confirm_password(request, user_id):
    if request.method == "POST":

        password = request.POST["new_password"]
        retype_password = request.POST["retype_new_password"]
        user = User.objects.get(pk=user_id)
        login(request, user)
        dict = {'old_password':request.user.password,'csrfmiddlewaretoken':request.POST['csrfmiddlewaretoken'],
                'new_password1':password, 'new_password2':retype_password}
        if password != retype_password:
            return render(request, "network/type_password.html", {"error": "Passwords Do Not Match"})

            # Attempt to create new user
        try:
            form = PasswordChangeForm(request.user, dict)
            if form.is_valid():
                user = form.save()
                update_session_auth_hash(request, user)  # Important for security
                return HttpResponseRedirect('index')
        except IntegrityError:
            return render(request, "network/register.html", {
                "message": "Password not accepted"
            })

        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "network/type_password.html", {"user": user_id})


def generate_verification_code(length=6):
    # Define the characters to use for the verification code (letters + digits)
    characters = string.ascii_letters + string.digits

    # Generate a random verification code of the specified length
    verification_code = ''.join(secrets.choice(characters) for _ in range(length))

    return verification_code


def user_closet(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save(commit=False)
            form.instance.user = request.user
        return HttpResponseRedirect(reverse("display_listing"))
    else:
        form = ImageUploadForm()
        return render(request, 'network/user_create.html', {'form': form})


def user_profile(request):
    if request.method == "POST":
        clothing_items = request.POST["clothing_items"]
        favorite_styles = request.POST["favorite_styles"]
        clothing_size = request.POST["clothing_size"]
        user_data = Style.objects.get(user=request.user)
        if user_data is not None:
            user_data.cloth_size = clothing_size
            user_data.cloth_item = clothing_items
            user_data.cloth_style = favorite_styles
            user_data.save()
        else:
            new_info = Style(cloth_size=clothing_size, cloth_item=clothing_items, cloth_style=favorite_styles)
            new_info.save()
    else:
        try:
            data = Style.objects.get(user=request.user)
            # Handle the found object here
        except ObjectDoesNotExist:
            # Handle the case where the object doesn't exist
            data = None
        return render(request, "network/user_profile.html", {"data":data})


def sort(data):
    list = []
    phrase = ""
    for character in data:
        if character == ",":
            list.append(phrase)
            phrase = ""
        else:
            phrase = phrase + character
    return list


def get_weather(api_key, latitude, longitude):
    # Define the API endpoint URL with your API key
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": "metric",  # You can use "imperial" for Fahrenheit
    }

    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        data = response.json()

        if data["cod"] == 200:
            # Weather data is available
            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            return temperature
        else:
            return "Weather data not available."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def fetch_weather_for_location():
    api_key = "2118f91ed006b3cda9fb5b2a0b70d1c2"
    location = get_user_location()
    # Split the location string into latitude and longitude
    latitude, longitude = location.split(",")

    # Convert latitude and longitude to float (if needed)
    latitude = float(latitude)
    longitude = float(longitude)

    weather_data = get_weather(api_key, latitude, longitude)
    return weather_data


def get_user_location():
    try:
        # Send a GET request to ipinfo.io to get user location based on their IP address
        response = requests.get("https://ipinfo.io")
        # Parse the JSON response
        data = response.json()

        # Extract location data
        location = data.get("loc", "Unknown")
        return location
    except Exception as e:
        return f"An error occurred: {str(e)}"


def arFunc(request, image_url):

    return render(request, "network/arImplementation.html", {"image_url": image_url})


# Webcam capture function
def webcam_stream(camera):
    while True:
        frame = camera.get_frame()  # Read a frame from the camera
        #processed_frame = process_frame_with_openpose(frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # Yield the frame as a multipart response
        if cv2.waitKey(0) & 0xFF==ord(' '):
            break
    cv2.destroyAllWindows()  # Release the camera when done


def live_capture_view(request, image_url):
    try:
        return StreamingHttpResponse(webcam_stream(VideoCamera(image_url)), content_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print("Stream closed:", e)
