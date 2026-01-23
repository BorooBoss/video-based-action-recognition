from source_files.adapters import florence_adapter
from source_files.adapters import paligemma_adapter

#open convert/normalize for used model - works with detect prompts
ADAPTERS = {
    "florence": florence_adapter,
    "paligemma": paligemma_adapter,
}

def normalize_output(raw_result, model_name, image_size=None):
    if model_name in ADAPTERS:
        return ADAPTERS[model_name].convert(raw_result, image_size=image_size)
    return raw_result