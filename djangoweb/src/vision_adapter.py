from src.adapters import florence_adapter
from src.adapters import paligemma_adapter
from src.adapters import qwen_adapter
from src.adapters import florence_weapon_adapter

#open convert/normalize for used model - works with detect prompts
ADAPTERS = {
    "florence": florence_adapter,
    "florence_weapon": florence_weapon_adapter,
    "paligemma": paligemma_adapter,
    "qwen": qwen_adapter,
}

def normalize_output(raw_result, model_name, image_size=None):
    if model_name in ADAPTERS:
        return ADAPTERS[model_name].convert(raw_result, image_size=image_size)
    return raw_result