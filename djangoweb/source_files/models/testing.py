from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

model_id = "google/paligemma2-3b-mix-224"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)

# Load model without device_map
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).eval().to(device)

processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = "describe en"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)