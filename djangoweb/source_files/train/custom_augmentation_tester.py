#preview augmented results
import cv2, os, random
from matplotlib import pyplot as plt

IMAGES = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/images_aug"
LABELS = r"/mnt/c/Users/boris/Desktop/5.semester/bp/weapon_aug/train/labels_aug"

samples = random.sample(os.listdir(IMAGES), 20)

for img_name in samples:
    img = cv2.imread(os.path.join(IMAGES, img_name))
    h, w, _ = img.shape

    label_path = os.path.join(LABELS, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)  # BGR for OpenCV
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    #convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #display random images for correction
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(img_name)
    plt.show()
