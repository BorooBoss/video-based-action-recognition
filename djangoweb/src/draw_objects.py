import os
from PIL import Image, ImageDraw, ImageFont

# Konfigurácia farieb (globálna, aby si ju mal na jednom mieste)
COLORS = {
    "weapon": "#FF4444",
    "person": "#44FF44",
    "knife": "#FF4444",
}
DEFAULT_COLOR = "#FF8C00"

def draw_boxes_florence(image_path, detections, output_path):
    """Kreslenie pre Florence (predpokladá súradnice v pixeloch [x1, y1, x2, y2])"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for obj in detections:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        label = obj.get("label", "object")
        color = COLORS.get(label.lower(), DEFAULT_COLOR)

        # Hrubší box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Pozadie pre label
        text_y = max(0, y1 - 25)
        try:
            bbox_text = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(bbox_text, fill=color)
        except:
            draw.rectangle([x1, text_y, x1 + len(label) * 10, text_y + 20], fill=color)

        draw.text((x1 + 2, text_y), label, fill="white", font=font)

    img.save(output_path)
    return img

def draw_boxes_paligemma(image_path, coords_and_labels, output_path):
    """Kreslenie pre PaliGemma (normalizované [y1, x1, y2, x2] -> pixely)"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for obj in coords_and_labels:
        # Prepočet z normalizovaných súradníc (0-1) na pixely
        # PaliGemma formát: [y1, x1, y2, x2]
        ny1, nx1, ny2, nx2 = obj['bbox']
        x1, y1, x2, y2 = nx1 * width, ny1 * height, nx2 * width, ny2 * height

        label = obj.get("label", "object")
        color = COLORS.get(label.lower(), DEFAULT_COLOR)

        # Hrubší box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Pozadie pre label
        text_y = max(0, y1 - 25)
        try:
            bbox_text = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(bbox_text, fill=color)
        except:
            draw.rectangle([x1, text_y, x1 + len(label) * 10, text_y + 20], fill=color)

        draw.text((x1 + 2, text_y), label, fill="white", font=font)

    img.save(output_path)
    return img

def draw_boxes_qwen(image_path, detections, output_path):
    draw_boxes_paligemma(image_path, detections, output_path)