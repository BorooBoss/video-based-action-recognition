import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from source_files.model_manager import manager
from source_files.vision_adapter import normalize_output


def initialize_model(model_id):
    if manager.model_id == model_id and manager.model is not None:
        print(f"Working with already loaded {model_id}")
        return
    
    if manager.model is not None and manager.model_id != model_id:
        manager.unload_model()

    # MODEL_ID = "microsoft/Florence-2-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype, #torch_dtype
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    manager.switch_model(model_id, model, processor, device, dtype)
    print(f"MODEL {model_id} LOADED SUCCESSFULLY")

def predict(image_path, prompt="describe", model_id=None, base_prompt=None):
    if model_id:
        initialize_model(model_id)
    
    #LOAD from path
    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB") # convert for PNG working

    inputs = manager.processor(text=prompt, images=image, return_tensors="pt").to(manager.device, manager.dtype)
    generated_ids = manager.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
    )

    generated_text = manager.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    raw_result = manager.processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )
    if base_prompt == "<OD>":
        result = normalize_output(raw_result, "florence")
    else:
        result = raw_result


    print("\n" + "=" * 60)
    print("FLORENCE 2 RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)
    return result

""" 
EXAMPLES 

prompt = "<CAPTION>"
prompt = "<DETAILED_CAPTION>"
prompt = "<MORE_DETAILED_CAPTION>"
prompt = "<OD>"

"""
