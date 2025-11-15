from source_files.adapters import florence_adapter
from source_files.adapters import paligemma_adapter

ADAPTERS = {
    "florence": florence_adapter,
    "paligemma": paligemma_adapter,
    # next
}

def normalize_output(raw_result, model_name):
    if model_name in ADAPTERS:
        return ADAPTERS[model_name].convert(raw_result)
    return raw_result