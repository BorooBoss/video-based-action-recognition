from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from source_files.models import load_model

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def recognize(request):
    if request.method == 'POST':
        image = request.FILES["image"]
        model_name = request.POST.get("model", "paligemma")
        prompt = request.POST.get("prompt", "describe\n")

        # Ulož dočasne uploadnutý obrázok
        tmp_path = "/tmp/uploaded_image.jpg"
        with open(tmp_path, "wb") as f:
            for chunk in image.chunks():
                f.write(chunk)

        model = load_model(model_name)
        result = model.predict(tmp_path, prompt)

        return JsonResponse({"model": model_name, "prompt": prompt, "result": result})

    return JsonResponse({"error": "Invalid request"}, status=400)