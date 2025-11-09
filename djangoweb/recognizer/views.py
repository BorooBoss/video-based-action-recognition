from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from source_files.models import load_model
from source_files.models import paligemma, florence
import os


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def recognize(request):
        
    if request.method == 'POST':
        image = request.FILES.get("image")
        model_name = request.POST.get("model")
        prompt = request.POST.get("prompt", "describe the image")
        print(model_name,"\n")

        if not image or not model_name:
            return JsonResponse({"error": "Missing image or model name"}, status=400)
        # Ulož dočasne uploadnutý obrázok
        tmp_path = "/tmp/uploaded_image.jpg"
        with open(tmp_path, "wb") as f:
            for chunk in image.chunks():
                f.write(chunk)
                
        try:
            # Vyber správny model a zavolaj predict s model_id
            if "paligemma" in model_name.lower():
                result = paligemma.predict(tmp_path, prompt, model_id=model_name)
            elif "florence" in model_name.lower():
                result = florence.predict(tmp_path, prompt, model_id=model_name)
                if isinstance(result, dict):
                    result = result[prompt]
            else:
                return JsonResponse({"error": "Unknown model type"}, status=400)
            print(model_name,"\n")
            # Vyčisti temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            return JsonResponse({
                "model": model_name, 
                "prompt": prompt, 
                "result": result
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        

    #     
    #     tmp_path = "/tmp/uploaded_image.jpg"
    #     with open(tmp_path, "wb") as f:
    #         for chunk in image.chunks():
    #             f.write(chunk)

    #     model = load_model(model_name)
    #     result = model.predict(tmp_path, prompt)

    #     return JsonResponse({"model": model_name, "prompt": prompt, "result": result})

    return JsonResponse({"error": "Invalid request"}, status=400)