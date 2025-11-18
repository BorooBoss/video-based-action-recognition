import os
from dotenv import load_dotenv
from PIL import Image
from source_files.model_manager import manager
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import gc
from qwen_vl_utils import process_vision_info


def initialize_model(model_id):
    """Initialize Qwen2-VL model - loads only once, reuses if already loaded"""

    # LOAD MODEL ONLY ONCE
    if manager.model_id == model_id and manager.model is not None:
        print(f"Working with already loaded {model_id}")
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
    print(f"Loading Qwen2-VL model: {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # Store in manager
    manager.switch_model(model_id, model, processor, device, dtype)

    print(f"MODEL {model_id} LOADED SUCCESSFULLY")


def predict(image_path, prompt="Describe this image.", model_id=None, max_new_tokens=512):
    """
    Generate description/analysis of image using Qwen2-VL model

    Args:
        image_path: Path to input image
        prompt: Text prompt/question about the image
        model_id: Model identifier (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text response about the image
    """

    # Initialize model if needed
    if model_id:
        initialize_model(model_id)

    print(f"LOADING IMAGE FROM: {image_path}")

    # Prepare messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs for the model
    text = manager.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = manager.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(manager.device)

    # Generate response
    print(f"Generating response (max {max_new_tokens} tokens)...")
    with torch.no_grad():
        generated_ids = manager.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    result = manager.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "=" * 60)
    print("QWEN2-VL RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    return result


# Example usage:

# Initialize with smaller vision model
initialize_model("Qwen/Qwen2-VL-2B-Instruct")

# Analyze image
response = predict(
    image_path="/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/source_files/samples/test2.jpg",
    prompt="What objects do you see in this image?",
    max_new_tokens=256
)
