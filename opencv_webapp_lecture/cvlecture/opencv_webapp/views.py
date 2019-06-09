from django.shortcuts import render

# Create your views here.
from django.shortcuts import redirect

from .Background_subtraction import BGsubtractor
from .main import pixelize
from .main import pixelizes
from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from django.conf import settings
from .models import ImageUploadModel

def first_page(request):
    return render(request, 'opencv_webapp/first_page.html', {})

def uimage(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            myfile = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            return render(request, 'opencv_webapp/uimage.html', {'form': form, 'uploaded_file_url': uploaded_file_url})

    else:
        form = UploadImageForm()
        return render(request, 'opencv_webapp/uimage.html', {'form': form})

def dface(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.save()

            # db delete
            if ImageUploadModel.objects.all().count() > 100:
                obs = ImageUploadModel.objects.all().first()
                if obs:
                    obs.delete()

            imageURL = settings.MEDIA_URL + form.instance.document.name
            a = BGsubtractor()
            img = a.run(settings.MEDIA_ROOT_URL + imageURL)  # a.bg_result is result image
            pixelize(img, settings.MEDIA_ROOT_URL + imageURL)

        return render(request, 'opencv_webapp/dface.html', {'form': form, 'post': post})
    else:
        form = ImageUploadForm()
    return render(request, 'opencv_webapp/dface.html', {'form': form})

def pixelization(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.save()

            # db delete
            if ImageUploadModel.objects.all().count() > 100:
                obs = ImageUploadModel.objects.all().first()
                if obs:
                    obs.delete()

            imageURL = settings.MEDIA_URL + form.instance.document.name
            pixelizes(settings.MEDIA_ROOT_URL + imageURL)

        return render(request, 'opencv_webapp/pixelization.html', {'form': form, 'post': post})
    else:
        form = ImageUploadForm()
        return render(request, 'opencv_webapp/pixelization.html', {'form': form})




