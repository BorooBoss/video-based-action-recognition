from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from source_files.models import load_model
from source_files.models import paligemma, florence
from source_files import draw_objects, user_input
import os
import base64
import subprocess
import json

def call_qwen(image_path, prompt):
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

    # DEBUG
    print("=== SUBPROCESS STDOUT ===")
    print(result.stdout)
    print("=== SUBPROCESS STDERR ===")
    print(result.stderr)

    # Vráť len text (už nie JSON)
    return result.stdout.strip()


def call_internvl(image_path, prompt):
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

    # DEBUG
    print("=== INTERNVL STDOUT ===")
    print(result.stdout)
    print("=== INTERNVL STDERR ===")
    print(result.stderr)

    # Vráť len text (už nie JSON)
    return result.stdout.strip()

def index(request):
    return render(request, 'index.html')


@csrf_exempt
def recognize(request):
    if request.method == 'POST':
        ui = user_input.UserInput()

        ui.image = request.FILES.get("image")
        ui.model_name = request.POST.get("model")
        ui.prompt_type = request.POST.get("prompt_type", "describe the image")
        ui.addition = request.POST.get("addition", "").strip()
        ui.set_base_prompt()

        print(f"Model: {ui.model_name}")
        print(f"Prompt type: {ui.prompt_type}")
        print(f"Base prompt: {ui.base_prompt}")
        print(f"Final prompt: {ui.full_prompt}")

        if not ui.image or not ui.model_name:
            return JsonResponse({"error": "Missing image or model name"}, status=400)
        # SAVE IMAGE
        tmp_path = "/tmp/uploaded_image.jpg"
        with open(tmp_path, "wb") as f:
            for chunk in ui.image.chunks():
                f.write(chunk)
        #   TU
        try:
            raw_result = None

            if "paligemma" in ui.model_name.lower():
                raw_result = paligemma.predict(tmp_path, ui.full_prompt, model_id=ui.model_name, base_prompt=ui.base_prompt)
                result = raw_result

            elif "florence" in ui.model_name.lower():
                raw_result = florence.predict(tmp_path, ui.full_prompt, model_id=ui.model_name, base_prompt=ui.base_prompt)
                if isinstance(raw_result, dict):
                    result = raw_result.get(ui.full_prompt, raw_result)
                else:
                    result = raw_result

            elif "qwen" in ui.model_name.lower():
                result = call_qwen(tmp_path, ui.full_prompt)

            elif "internvl" in ui.model_name.lower():
                result = call_internvl(tmp_path, ui.full_prompt)


            else:
                return JsonResponse({"error": "Unknown model type"}, status=400)

            print(f"DEBUG: raw_result type = {type(raw_result)}")
            print(f"DEBUG: raw_result = {raw_result}")
            print(ui.model_name, "\n")

            # DETECTION HANDLING
            detections_dict = None
            annotated_image_base64 = None
            output_image_path = None

            if isinstance(raw_result, list) and len(raw_result) > 0:
                # PaliGemma format - list of dicts with 'label' and 'bbox'
                print("DEBUG: Processing PaliGemma list format")
                # Konvertuj do formátu kompatibilného s draw_boxes_paligemma
                detections_dict = raw_result  # alebo si môžeš vytvoriť wrapper dict ak chceš

            elif isinstance(raw_result, dict):
                print("DEBUG: Processing Florence dict format")
                # Florence format
                if "<OD>" in raw_result:
                    detections_dict = raw_result["<OD>"]
                    print("DEBUG: Found <OD> in raw_result")
                elif ui.full_prompt in raw_result and isinstance(raw_result[ui.full_prompt], dict):
                    detections_dict = raw_result[ui.full_prompt]
                    print(f"DEBUG: Found {ui.full_prompt} in raw_result")

            if detections_dict:
                output_image_path = "/tmp/out.jpg"
                print("DEBUG: DETECTIONS:", detections_dict)
                #print("JE TO PALIGEMMA ALE SOM TU AJ TAK")
                if "florence" in ui.model_name.lower():
                    # Draw boxes
                    draw_objects.draw_boxes_florence(tmp_path, detections_dict, output_image_path)
                    print(f"DEBUG: Bounding boxes drawn!")

                if "paligemma" in ui.model_name.lower():
                    draw_objects.draw_boxes_paligemma(tmp_path, result, output_image_path) #image and coords
                # Check if output file exists
                if os.path.exists(output_image_path):
                    print(f"DEBUG: Output image exists at {output_image_path}")

                    # Convert to base64
                    with open(output_image_path, "rb") as img_file:
                        annotated_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                    print(f"DEBUG: Base64 created, length: {len(annotated_image_base64)}")
                else:
                    print(f"ERROR: Output image does NOT exist at {output_image_path}")

            # CLEAR TEMP
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"DEBUG: Removed temp input image: {tmp_path}")

            if output_image_path and os.path.exists(output_image_path):
                os.remove(output_image_path)
                print(f"DEBUG: Removed temp output image: {output_image_path}")

            # Prepare response
            response_data = {
                "model": ui.model_name,
                "prompt": ui.full_prompt,
                "result": result
            }

            # Add annotated image
            if annotated_image_base64:
                print(f"DEBUG: Adding annotated_image to response")
                response_data["annotated_image"] = f"data:image/jpeg;base64,{annotated_image_base64}"
            else:
                print("DEBUG: No annotated_image to add")

            return JsonResponse(response_data)

        except Exception as e:
            import traceback
            print("ERROR occurred:")
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)