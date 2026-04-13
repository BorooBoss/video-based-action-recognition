import os
import re
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from src.cache_manager import cache
from src.vision_adapter import normalize_output

# Záplaty knižnice
PretrainedConfig.forced_bos_token_id = None
nn.Module._supports_sdpa = False
nn.Module._supports_flash_attn_2 = False

ORIGINAL_MODEL_ID = "microsoft/Florence-2-large-ft"
WEAPON_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../train/florence2_weapon_finetune_v2")
WEAPON_CHECKPOINT  = os.path.join(WEAPON_MODEL_PATH, "checkpoint-6695")


def _is_weapon_model(model_id: str) -> bool:
    return "weapon" in model_id.lower()


def initialize_model(model_id: str):
    if cache.model_id == model_id and cache.model is not None:
        print(f"Working with already loaded {model_id}")
        return

    if cache.model is not None and cache.model_id != model_id:
        cache.unload_model()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if _is_weapon_model(model_id):
        print(f"--- INIT WEAPON MODEL ---")

        # 1. Čistý config z originálu (vision_config má správny model_type=davit)
        clean_config = AutoConfig.from_pretrained(ORIGINAL_MODEL_ID, trust_remote_code=True)

        # 2. Processor z originálu + pridaj trénované tokeny (weapon/person)
        #    Tokenizer musí mať 51291 tokenov aby sedeli váhy z checkpointu
        processor = AutoProcessor.from_pretrained(ORIGINAL_MODEL_ID, trust_remote_code=True)
        processor.tokenizer.add_tokens(["detect weapon;", "detect person;"])

        # 3. Uprav vocab_size v configu aby sedel s checkpointom (51291)
        vocab_size = len(processor.tokenizer)
        clean_config.text_config.vocab_size = vocab_size
        clean_config.vocab_size = vocab_size
        print(f"Tokenizer veľkosť: {vocab_size}")

        # 4. Načítaj váhy z checkpointu (nie z OUTPUT_DIR — ten má poškodený config)
        checkpoint_path = WEAPON_CHECKPOINT
        if not os.path.exists(checkpoint_path):
            # Fallback: nájdi posledný checkpoint
            import glob
            checkpoints = sorted(glob.glob(os.path.join(WEAPON_MODEL_PATH, "checkpoint-*")))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Používam checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"Žiadny checkpoint nenájdený v {WEAPON_MODEL_PATH}")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=clean_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
            ignore_mismatched_sizes=False,
        ).to(device)

    else:
        print(f"--- INIT STANDARD MODEL ({model_id}) ---")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model.eval()
    cache.switch_model(model_id, model, processor, device, dtype)
    print(f"MODEL {model_id} JE PRIPRAVENÝ")


def predict(image_path, prompt="describe", model_id=None, base_prompt=None):
    if model_id:
        initialize_model(model_id)

    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path).convert("RGB")

    inputs = cache.processor(
        text=prompt, images=image, return_tensors="pt"
    ).to(cache.device, cache.dtype)

    with torch.no_grad():
        generated_ids = cache.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
        )

    generated_text = cache.processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    print(f"RAW OUTPUT: {generated_text}")

    # Weapon model — vráť raw string, konverzia prebehne cez florence_weapon_adapter
    if _is_weapon_model(model_id or ""):
        print("=" * 60)
        print(f"FLORENCE WEAPON RESULT ({prompt}): {generated_text[:200]}")
        print("=" * 60)
        return generated_text

    # Štandardný Florence — post_process_generation
    raw_result = cache.processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    if base_prompt == "<OD>":
        result = normalize_output(raw_result, "florence")
    else:
        result = raw_result

    print("=" * 60)
    print(f"FLORENCE 2 RESULT ({prompt}):")
    print(result)
    print("=" * 60)
    return result