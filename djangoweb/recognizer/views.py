from django.shortcuts import render

from django.shortcuts import render
from django.http import JsonResponse
import source_files/models/paligemma.py

def index(request):
    return render(request, 'index.html')

def recognize(request):
    if request.method == 'POST':
        image = request.FILES['image']
        result = your_model_module.predict(image)  # voláš tvoj model
        return JsonResponse({'result': result})
