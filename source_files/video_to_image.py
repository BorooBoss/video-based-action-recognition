import os
import cv2 as OpenCV

def video_to_frames(video_path, output_folder, every_n_seconds=1, by_seconds=True):
    video = OpenCV.VideoCapture(video_path)
    if not video.isOpened():
        print("Failed to open video from path ", video_path)
        return
    
    os.makedirs(output_folder, exist_ok=True) #creates folder if does not exist

    fps = video.get(OpenCV.CAP_PROP_FPS)
    total_frames = int(video.get(OpenCV.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} total frames, {fps:.2f} FPS")

    #how often save frames
    if by_seconds:
        frame_interval = int(fps * every_n_seconds)
    else :
        frame_interval = every_n_seconds
    
    frame_count, saved_count = 0, 0
    while True:
        ret, frame = video.read() 
        if not ret:
            break

        #check if the frame is the one we want
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            OpenCV.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Done. Saved {saved_count} images.")




#EXAMPLE

video_to_frames(
    video_path="/mnt/c/Users/boris/Desktop/5.semester/bp/crcv.ucf.crime/Anomaly-Videos-Part-1/Abuse/Abuse023_x264.mp4",
    output_folder="/mnt/c/Users/boris/Desktop/5.semester/bp/source_files/frames",
    every_n_seconds=1,           
    by_seconds=True              # true - by seconds, false - by frames
)        