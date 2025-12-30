import os
import cv2
import random
import albumentations as A
from tqdm import tqdm

IMAGES_DIR = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/images"
LABELS_DIR = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/labels"
OUTPUT_IMAGES = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/images_aug"
OUTPUT_LABELS = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/labels_aug"

TARGET_IMAGE_COUNT = 8000
WEAPON_CLASS_ID = 0         # edit if weapon class is not zero

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

transform = A.Compose([
    A.RandomResizedCrop( #CROP
        size=(640, 640),
        scale=(0.6, 1.0), #60-100% of original
        ratio=(0.75, 1.33),
        p=0.5 #chance
    ),

    A.Rotate(limit=15, p=0.5), #ROTATE
    A.Affine(shear=(-15, 15), p=0.5), #SHEAR

    A.OneOf([
        A.MotionBlur(blur_limit=5), #BLUR
        A.GaussianBlur(blur_limit=5),
    ], p=0.3),

    A.GaussNoise(var_limit=(10, 50), p=0.3), #NOISE

    A.OneOf([
        A.ToGray(), #GREYSCALE OR COLOR EDIT
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.4,
            hue=0.1
        ),
    ], p=0.4),

], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3
))

# -----------------------------------------------------

def load_yolo_labels(path):
    boxes = []
    classes = []
    with open(path, "r") as f:
        for line in f:
            c, x, y, w, h = map(float, line.split())
            boxes.append([x, y, w, h])
            classes.append(int(c))
    return boxes, classes

def save_yolo_labels(path, boxes, classes):
    with open(path, "w") as f:
        for box, cls in zip(boxes, classes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))]

generated = 0
idx = 0

with tqdm(total=TARGET_IMAGE_COUNT) as pbar:
    while generated < TARGET_IMAGE_COUNT:
        img_name = random.choice(image_files)
        img_path = os.path.join(IMAGES_DIR, img_name)
        lbl_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        boxes, classes = load_yolo_labels(lbl_path)

        if WEAPON_CLASS_ID not in classes:
            continue

        augmented = transform(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        if len(augmented["bboxes"]) == 0:
            continue

        out_img_name = f"aug_{idx:06d}.jpg"
        out_lbl_name = f"aug_{idx:06d}.txt"

        cv2.imwrite(os.path.join(OUTPUT_IMAGES, out_img_name), augmented["image"])
        save_yolo_labels(
            os.path.join(OUTPUT_LABELS, out_lbl_name),
            augmented["bboxes"],
            augmented["class_labels"]
        )

        generated += 1
        idx += 1
        pbar.update(1)
