from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from uploads.core.algorithm import run
import os


def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES.get('myfile', False):
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        if fs.exists(myfile.name):
            os.remove(os.path.join(settings.MEDIA_ROOT, myfile.name))
        filename = fs.save(myfile.name, myfile)
        img = fs.open(filename)
        assert(filename == myfile.name)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url,
        })
    return render(request, 'core/simple_upload.html')


def run_replacement(request):
    unit_addr = os.path.join(settings.MEDIA_ROOT, 'load/l_unit.txt')
    skirt_addr = os.path.join(settings.MEDIA_ROOT, 'load/l_skirt.txt')
    fs = FileSystemStorage()
    if request.method == 'POST' and fs.exists(unit_addr) and fs.exists(skirt_addr):
        run()
        return render(request, 'core/simple_upload.html', {
            'replaced_skirt': os.path.join(settings.MEDIA_ROOT, 'result/gebing_replaced.jpg')
        })
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })