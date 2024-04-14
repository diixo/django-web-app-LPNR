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

    #####################
    path = Path(filepath)

    # save results:
    width = original.shape[1]
    height = original.shape[0]
    ratio = width / height
    new_height = 540

    output_img = cv2.resize(output_img, (int(new_height*ratio), new_height), interpolation=cv2.INTER_LINEAR)

    output_files = [ path.stem + "-1.jpg", path.stem + "-2.jpg", path.stem + "-3.jpg", path.stem + "-4.jpg"]

    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, output_files[0]), output_img)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, output_files[1]), plate)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, output_files[2]), img_gray)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, output_files[3]), char_rects)

    return predicted_str.upper(), output_files
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

            number_str, imgs = recognize_plate_number(path_str)

            return render(request, "app_main/upload.html", context = {
                "title": "upload image", 
                "form": PictureForm(instance=Picture()), 
                "imgs": imgs, "media":settings.MEDIA_URL, "plate_number":number_str
                })

    form = PictureForm(instance=Picture()) # bind Picture-model to PictureForm
    return render(request, "app_main/upload.html", context={"title": "upload image", "form": form, "plate_number":""})

def pictures(request):
    imgs = Picture.objects.all()
    return render(request, "app_main/pictures.html", 
        context={"title": "pictures output", "pictures": imgs, "media":settings.MEDIA_URL})
