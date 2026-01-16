import os
import cv2 as OpenCV
import json


import cv2
import os

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
            fname = f"{video_name}_{saved:05d}.jpg"
            path = os.path.join(output_folder, fname)
            cv2.imwrite(path, frame)

            frames.append({
                "file": fname,
                "index": saved,  #poradie frame-u
                "frame_idx": idx, #index FRAME-U VIDEA
                "time": round(idx / fps, 2)
            })
            saved += 1

        idx += 1

    cap.release()
    return frames

def process_videos(input_path, output_root, every_n_seconds=1, by_seconds=True):

    # case for one video
    if os.path.isfile(input_path):
        print("Processing SINGLE VIDEO:", input_path)
        os.makedirs(output_root, exist_ok=True)

        video_to_frames(
            video_path=input_path,
            output_folder=output_root,
            every_n_seconds=every_n_seconds,
            by_seconds=by_seconds
        )
        return

    # case for folder with videos
    elif os.path.isdir(input_path):
        print("Input is a directory — scanning for videos...")
        os.makedirs(output_root, exist_ok=True)

        videos = [f for f in os.listdir(input_path)
                  if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        print(f"Found {len(videos)} videos")

        for video_file in videos:
            full_path = os.path.join(input_path, video_file)
            print(f"Processing {video_file}...")

            video_to_frames(
                video_path=full_path,
                output_folder=output_root,
                every_n_seconds=every_n_seconds,
                by_seconds=by_seconds
            )
    else:
        print("ERROR: input_path is neither a file nor a folder")

    #call
#process_videos(
    input_path=r"/mnt/c/Users/boris/Desktop/5.semester/bp/datasets/smart-city_cctv_dataset/SCVD_converted/Train/Weaponized/w001_converted.avi",
    output_root=r"/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/frames",
    every_n_seconds=1
#)

