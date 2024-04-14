from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PictureForm
from .models import Picture
from pathlib import Path
from .app_ua import extract_plate, load_ml_model, segment_to_contours, predict_result
import os
import cv2


def recognize_plate_number(filepath: str):

    original = cv2.imread(filepath)
    output_img, plate = extract_plate(original)

    model = load_ml_model("ua-license-plate-recognition-model-37v2.h5")
    chars, img_gray, char_rects = segment_to_contours(plate)

    predicted_str = predict_result(chars, model)
    predicted_str = str.replace(predicted_str, '#', '')
    print(predicted_str)

    path = Path(filepath)
    #print(path.stem, path.suffix)
########################################################################


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
            
            path_str = str(os.path.join(settings.MEDIA_ROOT, saved_filepath))
            #print(path_str)
            recognize_plate_number(path_str)

            return redirect(to="app_main:pictures")

    form = PictureForm(instance=Picture()) # bind Picture-model to PictureForm
    return render(request, "app_main/upload.html", context={"title": "upload image", "form": form})

def pictures(request):
    imgs = Picture.objects.all()
    return render(request, "app_main/pictures.html", context={"title": "pictures output", "pictures": imgs, "media":settings.MEDIA_URL})
