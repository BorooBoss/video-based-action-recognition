import shutil
import cv2
import os
from mysite import settings

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
                "index": saved,
                "time_sec": round(idx / fps, 2),
                "frame_idx": idx
            })
            saved += 1

        idx += 1

    cap.release()
    return frames

