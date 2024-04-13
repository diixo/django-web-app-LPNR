from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PictureForm
from .models import Picture

# Create your views here.
def main(request):
    return render(request, "app_main/index.html", context={"title": "app_main"})

def upload(request):
    if request.method == "POST":
        form = PictureForm(request.POST, request.FILES, instance=Picture())
        if form.is_valid():
            form.save()
            return redirect(to="app_main:pictures")

    form = PictureForm(instance=Picture()) # bind Picture-model to PictureForm
    return render(request, "app_main/upload.html", context={"title": "upload image", "form": form})

def pictures(request):
    imgs = Picture.objects.all()
    return render(request, "app_main/pictures.html", context={"title": "pictures output", "pictures": imgs, "media":settings.MEDIA_URL})
