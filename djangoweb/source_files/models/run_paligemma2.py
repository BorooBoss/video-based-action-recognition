# run_paligemma2.py
import argparse, os, torch
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from huggingface_hub import login
from dotenv import load_dotenv




def predict(image_path, prompt, model_id):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print("BEZIM NA cuda")
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    image = Image.open(image_path).convert("RGB")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map=DEVICE  # Explicitne nastav device_map
    ).eval()

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # Explicitne presun všetky tensory na GPU
    inputs = {k: v.to(DEVICE, dtype=DTYPE if v.dtype == torch.float32 else v.dtype)
              for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    # Generovanie s explicitným device
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=DTYPE):  # Použitie mixed precision
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )


    raw_result = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    print(raw_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    predict(args.image, args.prompt, args.model_id)