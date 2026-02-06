import os, base64, subprocess, tempfile
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse

from source_files.models import florence, run_paligemma2
from recognizer import subprocess
from source_files import draw_objects, user_input
from source_files.video.ffmpeg_convert import convert_to_mp4
from source_files.vision_adapter import normalize_output
from source_files.video.frames import clear_temp_frames, ensure_temp_frames_dir, video_to_frames, TEMP_FRAMES_DIR


@csrf_exempt #load video and return temp frames
def video_frames(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    video = request.FILES.get("video")
    if not video:
        return JsonResponse({"error": "No video"}, status=400)

    clear_temp_frames()
    frames_dir = ensure_temp_frames_dir()

    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        for chunk in video.chunks():
            tmp_video.write(chunk)
        tmp_video.close()

        every_n_seconds = float(request.POST.get('every_n_seconds', 1))
        frames = video_to_frames(tmp_video.name, frames_dir, every_n_seconds)


        return JsonResponse({
            "frames": frames,
            "base_url": "/recognizer/frame/",
            "total": len(frames)
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp_video.name)


@csrf_exempt #return exact frame
def get_frame(request, filename):
    frame_path = os.path.join(TEMP_FRAMES_DIR, filename)

    if not os.path.exists(frame_path):
        return JsonResponse({"error": "Frame not found"}, status=404)

    return FileResponse(open(frame_path, 'rb'), content_type='image/jpeg')


@csrf_exempt #remove temp frames from sever
def clear_frames(request):
    if request.method == "POST":
        clear_temp_frames()
        return JsonResponse({"status": "cleared"})
    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt #convert videos to MP4
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

def index(request):
    return render(request, 'index.html')


@csrf_exempt
def recognize(request):
    if request.method == 'POST':
        selected_prompts = request.POST.getlist("selected_prompts[]")
        print(f"DEBUG: selected_prompts = {selected_prompts}")
        prompt_inputs = {p: request.POST.get(f"prompt_input_{p}", "").strip() for p in selected_prompts}

        ui = user_input.UserInput()
        ui.model_name = request.POST.get("model")
        uploaded_file = request.FILES.get("image")

        if not ui.model_name:
            return JsonResponse({"error": "Missing model name"}, status=400)
        #was video uploaded?
        is_video = uploaded_file and uploaded_file.content_type.startswith('video/')

        frames_to_process = []
        if is_video:
            #get array of all .jpg temp_frames
            frame_files = sorted([f for f in os.listdir(TEMP_FRAMES_DIR) if f.endswith('.jpg') and not f.startswith('annotated_')])
            for f in frame_files:
                frames_to_process.append(os.path.join(TEMP_FRAMES_DIR, f))
        else:
            #one image
            tmp_path = "/tmp/uploaded_image.jpg"
            with open(tmp_path, "wb") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            frames_to_process = [tmp_path]

        try:
            video_results = []

            #first loop - iterates each frame
            for current_image_path in frames_to_process:
                frame_name = os.path.basename(current_image_path)
                current_frame_results = []
                #second loop - iterates each prompt (detect/caption/vqa)
                for prompt_name in selected_prompts:
                    ui.prompt_type = prompt_name
                    raw_input = prompt_inputs.get(prompt_name, "")
                    sub_inputs = ui.split_if_needed(prompt_name, raw_input)

                    out_path = None
                    annotated_image_base64 = None
                    annotated_frame_url = None
                    all_results = []
                    all_detections = []

                    if not is_video:
                        out_path = f"/tmp/out_{prompt_name}_{frame_name}"

                    #third loop - iterates multiple prompts of one type (detect person; gun)
                    for sub_input in sub_inputs:
                        ui.prompt_input = sub_input
                        ui.set_base_prompt()

                        print(f"\n=== Processing prompt: {prompt_name} ===")
                        print(f"Base prompt: {ui.base_prompt}")
                        print(f"Full prompt: {ui.full_prompt}")
                        print(f"Model name: {ui.model_name}")

                        result = None #USED FOR DESCRIBE/VQA PROMPTS
                        raw_result = None #USED FOR DETECT PROMPTS NORMALIZATION

                        #choose model with current_image_path
                        if "paligemma" in ui.model_name.lower():
                            raw_result = run_paligemma2.predict(current_image_path, ui.full_prompt, ui.model_name)
                            #raw_result = subprocess.call_paligemma2(current_image_path, ui.full_prompt, ui.model_name)
                            if ui.base_prompt == "detect":
                                result = normalize_output(raw_result, "paligemma")
                            else:
                                result = raw_result
                            raw_result = result

                        elif "florence" in ui.model_name.lower():
                            raw_result = florence.predict(current_image_path, ui.full_prompt, model_id=ui.model_name,base_prompt=ui.base_prompt)
                            if ui.base_prompt == "detect":
                                result = normalize_output(raw_result, "florence")
                                raw_result = result
                            else:
                                result = raw_result

                        elif "qwen" in ui.model_name.lower():
                            raw_result = subprocess.call_qwen(current_image_path, ui.full_prompt)
                            result = raw_result

                        elif "internvl" in ui.model_name.lower():
                            raw_result = subprocess.call_internvl(current_image_path, ui.full_prompt)
                            result = raw_result
                        else:
                            continue

                        detections_dict = None


                        if isinstance(raw_result, list) and len(raw_result) > 0:
                            detections_dict = raw_result
                        elif isinstance(raw_result, dict):
                            if "<OD>" in raw_result:
                                detections_dict = raw_result["<OD>"]
                            elif ui.full_prompt in raw_result and isinstance(raw_result[ui.full_prompt], dict):
                                detections_dict = raw_result[ui.full_prompt]

                        if detections_dict:
                            if isinstance(detections_dict, list):
                                all_detections.extend(detections_dict)
                            elif isinstance(detections_dict, dict):
                                all_detections.append(detections_dict)


                        #remove florence prompt <OD>/<VQA> from origin output
                        if isinstance(raw_result, dict) and len(raw_result) == 1:
                            result = list(raw_result.values())[0]
                        elif isinstance(raw_result, dict) and ui.full_prompt in raw_result:
                            result = raw_result[ui.full_prompt]
                        else:
                            result = raw_result
                        all_results.append(result)

                    if all_detections:
                        #for video: save into TEMP_FRAMES_DIR, for one frame: /tmp
                        if is_video:
                            base_name = frame_name.replace('.jpg', '')
                            annotated_filename = f"annotated_{base_name}.jpg"
                            out_path = os.path.join(TEMP_FRAMES_DIR, annotated_filename)

                        if "florence" in ui.model_name.lower():
                            draw_objects.draw_boxes_florence(current_image_path, all_detections, out_path)
                        elif "paligemma" in ui.model_name.lower():
                            draw_objects.draw_boxes_paligemma(current_image_path, all_detections, out_path)

                    if out_path and os.path.exists(out_path):
                        if not is_video:
                            with open(out_path, "rb") as img_file:
                                annotated_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                            os.remove(out_path)
                        else:
                            annotated_frame_url = f"/recognizer/frame/{annotated_filename}"

                    current_frame_results.append({
                        "prompt_name": prompt_name,
                        "prompt_code": ui.base_prompt,
                        "input": raw_input,
                        "result": all_results if len(all_results) > 1 else (all_results[0] if all_results else None),
                        "annotated_image": f"data:image/jpeg;base64,{annotated_image_base64}" if annotated_image_base64 else None,
                        "annotated_frame_url": annotated_frame_url
                    })
                #save result for one exact frame
                #should work like frame_00005.jpg -> 00005
                timestamp_val = frame_name.split('_')[-1].replace('.jpg', '') if '_' in frame_name else "0"

                video_results.append({
                    "frame": frame_name,
                    "timestamp": timestamp_val,
                    "analysis": current_frame_results
                })

            if not is_video and os.path.exists(tmp_path):
                os.remove(tmp_path)

            return JsonResponse({
                "model": ui.model_name,
                "is_video": is_video,
                "results": video_results #return array of results for every screen
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)