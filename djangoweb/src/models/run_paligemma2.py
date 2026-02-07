import argparse, os, torch

from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from huggingface_hub import login
from dotenv import load_dotenv
from src.cache_manager import cache

#DISABLE CUDA GRAPHS + torch.compile
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Vypne torch.compile úplne
torch._dynamo.config.suppress_errors = True

#laod model into cache
def initialize_model(model_id):
    if cache.model_id == model_id and cache.model is not None:
        print(f"Working with already loaded {model_id}")
        return

    if cache.model is not None and cache.model_id != model_id:
        cache.unload_model()

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto"   # 🔧 OPRAVA
    ).eval()

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    cache.switch_model(model_id, model, processor, DEVICE, DTYPE)
    print(f"MODEL {model_id} LOADED SUCCESSFULLY")

def predict(image_path, prompt, model_id=None):
    if model_id:
        print("uz sem tady zas")
        initialize_model(model_id)
    print("jo")
    image = Image.open(image_path).convert("RGB")

    inputs = cache.processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(cache.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]
    print("cojeeeeeeeeeee")
    with torch.inference_mode():
        if cache.device == "cuda":
            with torch.cuda.amp.autocast(dtype=cache.dtype):
                outputs = cache.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
        else:
            outputs = cache.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

    raw_result = cache.processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    print(raw_result)
    return raw_result


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    predict(args.image, args.prompt, args.model_id)
    
"""
