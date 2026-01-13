import os
import cv2 as OpenCV
import json


def video_to_frames(video_path, output_folder, every_n_seconds=1, by_seconds=True):
    video = OpenCV.VideoCapture(video_path)
    if not video.isOpened():
        print("Failed to open video:", video_path)
        return

    os.makedirs(output_folder, exist_ok=True) #creates folder if missing

    fps = video.get(OpenCV.CAP_PROP_FPS)
    total_frames = int(video.get(OpenCV.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(video_path)} → {total_frames} frames, {fps:.2f} FPS")

    # how often save frames
    frame_interval = int(fps * every_n_seconds) if by_seconds else every_n_seconds

    # video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count, saved_count = 0, 0
    metadata = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # check if the frame is the one we want
        if frame_count % frame_interval == 0:
            filename = os.path.join(f"{video_name}_frame_{saved_count:05d}.jpg")
            OpenCV.imwrite(filename, frame)
            print("hhhhhhhhkhh")

            time_sec = frame_count / fps
            metadata.append({
                "frame": filename,
                "time_sec": round(time_sec, 2),
                "frame_idx": frame_count
            })
            saved_count += 1

        frame_count += 1

    with open(os.path.join(output_folder, f"{video_name}_frames.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    video.release()
    print(f"Saved {saved_count} frames to {output_folder}\n")


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

