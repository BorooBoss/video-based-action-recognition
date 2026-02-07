import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from src.cache_manager import cache
from src.vision_adapter import normalize_output

#laod model into cache
def initialize_model(model_id):
    if cache.model_id == model_id and cache.model is not None:
        print(f"Working with already loaded {model_id}")
        return
    
    if cache.model is not None and cache.model_id != model_id:
        cache.unload_model()


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype, #torch_dtype
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    cache.switch_model(model_id, model, processor, device, dtype)
    print(f"MODEL {model_id} LOADED SUCCESSFULLY")

#predict function with results
def predict(image_path, prompt="describe", model_id=None, base_prompt=None):
    if model_id:
        initialize_model(model_id) #first check cache


    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB") # convert for PNG working

    inputs = cache.processor(text=prompt, images=image, return_tensors="pt").to(cache.device, cache.dtype)
    generated_ids = cache.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
    )

    generated_text = cache.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    raw_result = cache.processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )
    #change structure of output because of detect prompt
    if base_prompt == "<OD>":
        result = normalize_output(raw_result, "florence")
    #standard output
    else:
        result = raw_result


    print("\n" + "=" * 60)
    print("FLORENCE 2 RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)
    return result
