# run_paligemma2.py
import argparse, os, re, torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login
from dotenv import load_dotenv
import json

MODEL_ID = "google/paligemma-3b-mix-224"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def normalize_paligemma(decoded):
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)"
    matches = re.findall(pattern, decoded)

    results = []
    for y1, x1, y2, x2, label in matches:
        results.append({
            "label": label,
            "bbox": [
                int(y1) / 1024,
                int(x1) / 1024,
                int(y2) / 1024,
                int(x2) / 1024
            ]
        })
    return results


def main(image_path, prompt):
    # 🔐 HF LOGIN
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    image = Image.open(image_path).convert("RGB")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        revision="bfloat16"
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    decoded = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    result = normalize_paligemma(decoded)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    main(args.image, args.prompt)
