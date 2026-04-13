import os
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "weapon": "#FF4444",
    "person": "#44FF44",
}
DEFAULT_COLOR = "#FF8C00"


def draw_boxes_florence_weapon(image_path, detections, output_path):
    """
    Kreslí bounding boxy pre finetuned Florence weapon model.

    detections = [{"label": "weapon", "bbox": [x1, y1, x2, y2]}, ...]
    bbox hodnoty sú v pixeloch (po konverzii cez florence_weapon_adapter).
    """
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Skús načítať font, fallback na default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for obj in detections:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        label = obj.get("label", "object")
        color = COLORS.get(label.lower(), DEFAULT_COLOR)

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label pozadie
        text_y = max(0, y1 - 20)
        try:
            bbox_text = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(bbox_text, fill=color)
        except Exception:
            draw.rectangle([x1, text_y, x1 + len(label) * 8, text_y + 16], fill=color)

        draw.text((x1, text_y), label, fill="white", font=font)

    img.save(output_path)
    return img