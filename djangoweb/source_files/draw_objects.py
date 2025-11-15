from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image_path, detections, output_path):
    """
    detections = {
        "bboxes": [[x1, y1, x2, y2], ...],
        "labels": ["elephant", "giraffe", ...]
    }
    """

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for bbox, label in zip(detections["bboxes"], detections["labels"]):
        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1 - 10), label, fill="red")

    img.save(output_path)
