import os, re
from dotenv import load_dotenv
from PIL import Image
from source_files.model_manager import manager
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import gc
from source_files.models import profiles
from source_files.vision_adapter import normalize_output


def initialize_model(model_id):
    # LOAD MODEL ONLY ONCE
    if manager.model_id == model_id and manager.model is not None:
        print("Working with already loaded {model_id}")
        return

    if manager.model is not None and manager.model_id != model_id:
        manager.unload_model()

    # HUGGING FACE LOGIN
    if not os.getenv("HF_TOKEN_LOADED"):
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(token=HF_TOKEN)
        os.environ["HF_TOKEN_LOADED"] = "1"

    # SET UP MODEL, DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # LOAD MODEL & PROCESSOR
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    processor = AutoProcessor.from_pretrained(model_id)
    manager.switch_model(model_id, model, processor, device, dtype)

    print(f"MODEL {model_id} LOADED SUCCESSFULLY")


def prompt_manager(prompt):
    prompt = prompt.lower()
    if "detect" in prompt:
        return "detect"
    else:
        return "text_generation"

def predict(image_path, prompt="describe\n", model_id=None, task_type=None):
    #Load image from path
    # image_path = "/mnt/c/Users/boris/Desktop/5.semester/bp/source_files/samples/test2.jpg"
    if model_id:
        initialize_model(model_id)

    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB")

    inputs = manager.processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(manager.device, dtype=manager.dtype)

    #Generate response
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
    if prompt == "detect":
        result = normalize_output(raw_result, "paligemma")
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
