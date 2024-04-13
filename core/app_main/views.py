from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PictureForm
from .models import Picture
from pathlib import Path
import os


# Create your views here.
def main(request):
    return render(request, "app_main/index.html", context={"title": "app_main"})

def upload(request):
    if request.method == "POST":
        form = PictureForm(request.POST, request.FILES, instance=Picture())
        if form.is_valid():

            original_filepath = form.instance.path
            # for filename, file in request.FILES.items():
            #     print(request.FILES[filename].name)
            form.save()
            saved_filepath = str(form.instance.path)
            
            saved_filepath = os.path.join(str(settings.MEDIA_ROOT), saved_filepath)
            print("saved_filepath:", saved_filepath)

            return redirect(to="app_main:pictures")

    form = PictureForm(instance=Picture()) # bind Picture-model to PictureForm
    return render(request, "app_main/upload.html", context={"title": "upload image", "form": form})

def pictures(request):
    imgs = Picture.objects.all()
    return render(request, "app_main/pictures.html", context={"title": "pictures output", "pictures": imgs, "media":settings.MEDIA_URL})
