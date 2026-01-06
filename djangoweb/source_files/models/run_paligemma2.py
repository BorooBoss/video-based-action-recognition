# run_paligemma2.py
import argparse, os, re, torch
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from huggingface_hub import login
from dotenv import load_dotenv
import json

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def predict(image_path, prompt, model_id):

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    image = Image.open(image_path).convert("RGB")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).eval().to(DEVICE)

    processor = PaliGemmaProcessor.from_pretrained(model_id)

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

    raw_result = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    print(json.dumps(raw_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    predict(args.image, args.prompt, args.model_id)
