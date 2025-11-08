import importlib

def load_model(model_name):
    try:
        module = importlib.import_module(f"source_files.models.{model_name}")
        return module
    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' not found.")