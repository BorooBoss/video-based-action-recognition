import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import gc
from source_files.model_manager import manager

def initialize_model(model_id):
    if manager.model_id == model_id and manager.model is not None:
        print(f"Working with already loaded {model_id}")
        return
    
    if manager.model is not None and manager.model_id != model_id:
        manager.unload_model()
    #Model ID + device
    # MODEL_ID = "microsoft/Florence-2-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    #Load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype, #torch_dtype
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    manager.switch_model(model_id, model, processor, device, dtype)
    print(f"MODEL {model_id} LOADED SUCCESSFULLY")

def predict(image_path, prompt="describe", model_id=None):
    if model_id:
        initialize_model(model_id)
    
    #Load image from path 
    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB") # convert for PNG working

    #Preprocess inputs
    inputs = manager.processor(text=prompt, images=image, return_tensors="pt").to(manager.device, manager.dtype)
    generated_ids = manager.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
    )

    #Generate response
    generated_text = manager.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    result = manager.processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    print("\n" + "=" * 60)
    print("FLORENCE 2 RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)
    return result

"""

<MORE_DETAILED_CAPTION> → výstup = text s <locXXXX>
<OD> → výstup = reálne [x1, y1, x2, y2] čísla


EXAMPLES 

Caption
prompt = "<CAPTION>"

Detailed Caption
prompt = "<DETAILED_CAPTION>"

More Detailed Caption
prompt = "<MORE_DETAILED_CAPTION>"

Caption to Phrase Grounding
caption to phrase grounding task requires additional text input, i.e. caption.

Caption to phrase grounding results format: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")

Object Detection
OD results format: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }

prompt = "<OD>"
run_example(prompt)

Dense Region Caption
Dense region caption results format: {'<DENSE_REGION_CAPTION>' : {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }

prompt = "<DENSE_REGION_CAPTION>"
run_example(prompt)

Region proposal
Dense region caption results format: {'<REGION_PROPOSAL>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}

prompt = "<REGION_PROPOSAL>"
run_example(prompt)

OCR
prompt = "<OCR>"
run_example(prompt)

OCR with Region
OCR with region output format: {'<OCR_WITH_REGION>': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}

prompt = "<OCR_WITH_REGION>"
run_example(prompt)
"""
