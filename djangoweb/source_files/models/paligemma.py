
import os, re
from dotenv import load_dotenv
from PIL import Image
from source_files.model_manager import manager
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login
import gc
from source_files.models import profiles
from source_files.vision_adapter import normalize_output
from source_files import user_input


def initialize_model(model_id):
    # LOAD MODEL ONLY ONCE
    if manager.model_id == model_id and manager.model is not None:
        print(f"Working with already loaded {model_id}")
        return
    if manager.model is not None and manager.model_id != model_id:
        manager.unload_model()

    if not os.getenv("HF_TOKEN_LOADED"):  # HUGGING FACE LOGIN
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(token=HF_TOKEN)
        os.environ["HF_TOKEN_LOADED"] = "1"

    # SET UP MODEL, DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on device: {device} with dtype: {dtype}")

    # Načítaj model a PRESUŇ ho na GPU/CPU
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    # Načítaj processor
    processor = AutoProcessor.from_pretrained(model_id)

    manager.switch_model(model_id, mix_model, processor, device, dtype)

    print(f"MODEL {model_id} LOADED SUCCESSFULLY on {device}")


def prompt_manager(prompt):
    prompt = prompt.lower()
    if "detect" in prompt:
        return "detect"
    else:
        return "text_generation"


def predict(image_path, prompt="describe\n", model_id=None, task_type=None, base_prompt=None):
    if model_id:
        initialize_model(model_id)

    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_size = image.size  # (width, height)

    # Spracuj vstup
    inputs = manager.processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # KRITICKÉ: presuň VŠETKY tensory na správne zariadenie A správny dtype
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue

        if k in ["input_ids", "attention_mask"]:
            # These must stay integer
            inputs[k] = v.to(manager.device)
        else:
            # Everything else can use dtype
            inputs[k] = v.to(manager.device, dtype=manager.dtype)

    print(f"Inputs moved to device: {manager.device}, dtype: {manager.dtype}")

    # Generate response
    if task_type is None:
        task_type = prompt_manager(prompt)
    gen_config = profiles.GENERATION_CONFIGS.get(task_type, profiles.GENERATION_CONFIGS["text_generation"])

    print(f"Generating response with {task_type} config...")
    with torch.no_grad():
        outputs = manager.model.generate(
            **inputs,
            **gen_config
        )

    raw_result = manager.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if raw_result.lower().startswith(prompt.lower()):
        raw_result = raw_result[len(prompt):].strip()

    if base_prompt == "detect":
        print(f"Normalizing output with image_size: {image_size}")
        result = normalize_output(raw_result, "paligemma", image_size=image_size)
    else:
        result = raw_result

    print("\n" + "=" * 60)
    print("PALIGEMMA 2 RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    return result
"""
ls ~/.cache/huggingface/hub/models--*

PALIGEMMA 2 OFFICIAL PROMPTS
"cap {lang}\n": Very raw short caption (only supported by PT)
"caption {lang}\n": Short captions
"describe {lang}\n": Somewhat longer, more descriptive captions (only supported by PT)
"ocr": Optical character recognition (only supported by PT)
"answer {lang} {question}\n": Question answering about the image contents
"question {lang} {answer}\n": Question generation for a given answer (only supported by PT)
"detect {object} ; {object}\n": Locate listed objects in an image and return the bounding boxes for those objects
"segment {object} ; {object}\n": Locate the area occupied by the listed objects in an image to create an image segmentation for that object

"""