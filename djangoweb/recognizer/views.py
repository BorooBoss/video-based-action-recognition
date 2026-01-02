from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from source_files.models import paligemma, florence
from source_files import draw_objects, user_input
import os, base64, subprocess

def call_qwen(image_path, prompt): #FROM qwen_env -> ai_env
    result = subprocess.run(
        [
            "/home/borooboss11/miniconda3/envs/qwen_env/bin/python",
            "/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/source_files/models/run_qwen.py",
            "--image", image_path,
            "--prompt", prompt,
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def call_internvl(image_path, prompt): #FROM invernvl_env -> ai_env
    result = subprocess.run(
        [
            "/home/borooboss11/miniconda3/envs/internvl_env/bin/python",
            "/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/source_files/models/run_internvl.py",
            "--image", image_path,
            "--prompt", prompt,
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def recognize(request):
    if request.method == 'POST':
        # Načítaj VŠETKY zaškrtnuté prompty z frontendu
        selected_prompts = request.POST.getlist("selected_prompts[]")  # ["DETECT", "VQA"]
        prompt_inputs = {}  # {"VQA": "What is this?", "DETECT": "person car"}

        # Pre každý prompt načítaj jeho input (ak má)
        for prompt_name in selected_prompts:
            input_value = request.POST.get(f"prompt_input_{prompt_name}", "").strip()
            prompt_inputs[prompt_name] = input_value

        ui = user_input.UserInput()
        ui.image = request.FILES.get("image")
        ui.model_name = request.POST.get("model")

        if not ui.image or not ui.model_name:
            return JsonResponse({"error": "Missing image or model name"}, status=400)

        # SAVE IMAGE
        tmp_path = "/tmp/uploaded_image.jpg"
        with open(tmp_path, "wb") as f:
            for chunk in ui.image.chunks():
                f.write(chunk)

        try:
            results = []  # Tu uložíme výsledky všetkých promptov

            # PREJDI VŠETKY VYBRANÉ PROMPTY
            for prompt_name in selected_prompts:
                ui.prompt_type = prompt_name
                ui.addition = prompt_inputs.get(prompt_name, "")
                ui.set_base_prompt()

                print(f"\n=== Processing prompt: {prompt_name} ===")
                print(f"Base prompt: {ui.base_prompt}")
                print(f"Full prompt: {ui.full_prompt}")

                raw_result = None

                # CALL MODEL
                if "paligemma" in ui.model_name.lower():
                    raw_result = paligemma.predict(tmp_path, ui.full_prompt, model_id=ui.model_name,
                                                   base_prompt=ui.base_prompt)
                    result = raw_result

                elif "florence" in ui.model_name.lower():
                    raw_result = florence.predict(tmp_path, ui.full_prompt, model_id=ui.model_name,
                                                  base_prompt=ui.base_prompt)
                    if isinstance(raw_result, dict):
                        result = raw_result.get(ui.full_prompt, raw_result)
                    else:
                        result = raw_result

                elif "qwen" in ui.model_name.lower():
                    result = call_qwen(tmp_path, ui.full_prompt)
                    raw_result = result

                elif "internvl" in ui.model_name.lower():
                    result = call_internvl(tmp_path, ui.full_prompt)
                    raw_result = result

                else:
                    continue

                # SPRACUJ DETEKCIE (ak je to detekčný prompt)
                annotated_image_base64 = None
                detections_dict = None

                if isinstance(raw_result, list) and len(raw_result) > 0:
                    detections_dict = raw_result
                elif isinstance(raw_result, dict):
                    if "<OD>" in raw_result:
                        detections_dict = raw_result["<OD>"]
                    elif ui.full_prompt in raw_result and isinstance(raw_result[ui.full_prompt], dict):
                        detections_dict = raw_result[ui.full_prompt]

                if detections_dict:
                    output_image_path = f"/tmp/out_{prompt_name}.jpg"

                    if "florence" in ui.model_name.lower():
                        draw_objects.draw_boxes_florence(tmp_path, detections_dict, output_image_path)

                    if "paligemma" in ui.model_name.lower():
                        draw_objects.draw_boxes_paligemma(tmp_path, result, output_image_path)

                    if os.path.exists(output_image_path):
                        with open(output_image_path, "rb") as img_file:
                            annotated_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        os.remove(output_image_path)

                # Ulož výsledok tohto promptu
                results.append({
                    "prompt_name": prompt_name,
                    "prompt_code": ui.base_prompt,
                    "input": ui.addition,
                    "result": result,
                    "annotated_image": f"data:image/jpeg;base64,{annotated_image_base64}" if annotated_image_base64 else None
                })

            # CLEAR temp
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # Vráť VŠETKY výsledky
            return JsonResponse({
                "model": ui.model_name,
                "results": results
            })

        except Exception as e:
            import traceback
            print("ERROR occurred:")
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)