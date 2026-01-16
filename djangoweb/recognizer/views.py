import json, os, base64, subprocess, tempfile
import shutil

import cv2
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse

from mysite import settings
from source_files.models import florence
from source_files import draw_objects, user_input
from source_files.video.ffmpeg_convert import convert_to_mp4
from source_files.vision_adapter import normalize_output

TEMP_FRAMES_DIR = os.path.join(settings.BASE_DIR, 'temp_frames')


def ensure_temp_frames_dir():
    if not os.path.exists(TEMP_FRAMES_DIR):
        os.makedirs(TEMP_FRAMES_DIR)
    return TEMP_FRAMES_DIR


def clear_temp_frames():
    if os.path.exists(TEMP_FRAMES_DIR):
        shutil.rmtree(TEMP_FRAMES_DIR)
    ensure_temp_frames_dir()


def video_to_frames(video_path, output_folder, every_n_seconds=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    os.makedirs(output_folder, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback

    frame_interval = int(fps * every_n_seconds)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frames = []
    idx, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            fname = f"frame_{saved:05d}.jpg"
            path = os.path.join(output_folder, fname)
            cv2.imwrite(path, frame)

            frames.append({
                "file": fname,
                "time": round(idx / fps, 2)
            })
            saved += 1

        idx += 1

    cap.release()
    return frames


@csrf_exempt
def video_frames(request):
    """Spracuje video a vráti zoznam snímkov"""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    video = request.FILES.get("video")
    if not video:
        return JsonResponse({"error": "No video"}, status=400)

    # Vyčisti staré snímky
    clear_temp_frames()
    frames_dir = ensure_temp_frames_dir()

    # Ulož video dočasne
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        for chunk in video.chunks():
            tmp_video.write(chunk)
        tmp_video.close()

        # Konvertuj video na snímky (každú 1 sekundu)
        every_n_seconds = float(request.POST.get('every_n_seconds', 1))
        frames = video_to_frames(tmp_video.name, frames_dir, every_n_seconds)

        # Vráť zoznam snímkov
        return JsonResponse({
            "frames": frames,
            "base_url": "/recognizer/frame/",
            "total": len(frames)
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp_video.name)


@csrf_exempt
def get_frame(request, filename):
    """Vráti konkrétny snímok"""
    frame_path = os.path.join(TEMP_FRAMES_DIR, filename)

    if not os.path.exists(frame_path):
        return JsonResponse({"error": "Frame not found"}, status=404)

    return FileResponse(open(frame_path, 'rb'), content_type='image/jpeg')


@csrf_exempt
def clear_frames(request):
    """Vymaže všetky dočasné snímky"""
    if request.method == "POST":
        clear_temp_frames()
        return JsonResponse({"status": "cleared"})
    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def convert_video(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    video = request.FILES.get("video")
    if not video:
        return JsonResponse({"error": "No video"}, status=400)

    tmp_in = tempfile.NamedTemporaryFile(delete=False)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        for chunk in video.chunks():
            tmp_in.write(chunk)
        tmp_in.close()
        tmp_out.close()

        convert_to_mp4(tmp_in.name, tmp_out.name)

        with open(tmp_out.name, "rb") as f:
            data = f.read()

        return HttpResponse(data, content_type="video/mp4")

    finally:
        os.unlink(tmp_in.name)
        os.unlink(tmp_out.name)

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
    return result.stdout.strip()


def call_paligemma2(image_path, prompt, model_id):
    result = subprocess.run(
        [
            "/home/borooboss11/miniconda3/envs/paligemma2_env/bin/python",
            "/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/source_files/models/run_paligemma2.py",
            "--image", image_path,
            "--prompt", prompt,
            "--model_id", model_id
        ],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return json.loads(result.stdout)


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def recognize(request):
    if request.method == 'POST':
        # nacitaj vsetky vybrate prompty
        selected_prompts = request.POST.getlist("selected_prompts[]")
        prompt_inputs = {}

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
            results = []

            # PREJDI VŠETKY VYBRANÉ PROMPTY
            for prompt_name in selected_prompts:
                ui.prompt_type = prompt_name
                ui.addition = prompt_inputs.get(prompt_name, "")
                ui.set_base_prompt()

                print(f"\n=== Processing prompt: {prompt_name} ===")
                print(f"Base prompt: {ui.base_prompt}")
                print(f"Full prompt: {ui.full_prompt}")
                print(f"Model name: {ui.model_name}")

                raw_result = None

                # CALL MODEL
                if "paligemma" in ui.model_name.lower():
                    print("som v call")
                    raw_result = call_paligemma2(tmp_path, ui.full_prompt, ui.model_name)

                    if ui.base_prompt == "detect":
                        result = normalize_output(raw_result, "paligemma")
                    else:
                        result = raw_result

                    raw_result = result

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