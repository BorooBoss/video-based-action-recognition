from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import re

model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

image_path = "path/to/your/image.jpg"  # Tu zadaj cestu k svojmu obrázku
image = Image.open(image_path)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "detect car"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)


def draw_bounding_boxes(image, detection_string, output_path="output_with_boxes.jpg"):

    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    width, height = image.size

    pattern = r'<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s*(\w+)'
    matches = re.findall(pattern, detection_string)

    for match in matches:
        y_min, x_min, y_max, x_max, label = match

        y_min = int(y_min) / 1024 * height
        x_min = int(x_min) / 1024 * width
        y_max = int(y_max) / 1024 * height
        x_max = int(x_max) / 1024 * width


        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), label, fill="red")


    img_with_boxes.save(output_path)
    print(f"Obrázok s bounding boxami uložený do: {output_path}")

    return img_with_boxes


result_image = draw_bounding_boxes(image, decoded)