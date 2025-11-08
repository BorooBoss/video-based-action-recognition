import os
from dotenv import load_dotenv
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import gc

MODEL_ID = None
model = None
processor = None
device = None
dtype = None

def unload_model():
    """Uvoľní model a RAM/GPU pamäť"""
    global MODEL_ID, model, processor, device, dtype

    if model is not None:
        print("🧹 Releasing old model from memory...")
        try:
            del model
        except:
            pass

    if processor is not None:
        del processor

    model = None
    processor = None
    MODEL_ID = None
    device = None
    dtype = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory cleaned\n")

def initialize_model(model_id):
    # LOAD MODEL ONLY ONCE
    global MODEL_ID, model, processor, device, dtype
    if MODEL_ID == model_id and model is not None:
        print("Working with already loaded {model_id}")
        return

    if model is not None and MODEL_ID != model_id:
        unload_model()

    # HUGGING FACE LOGIN
    if not os.getenv("HF_TOKEN_LOADED"):
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(token=HF_TOKEN)
        os.environ["HF_TOKEN_LOADED"] = "1"

    # SET UP MODEL, DEVICE
    MODEL_ID = model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    #print("Loading Paligemma 2 🚀 using device: {device}")

    # LOAD MODEL & PROCESSOR
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"MODEL {model_id} LOADED SUCCESSFULLY")


def predict(image_path, prompt="describe\n", model_id=None):
    #Load image from path
    # image_path = "/mnt/c/Users/boris/Desktop/5.semester/bp/source_files/samples/test2.jpg"
    if model_id:
        initialize_model(model_id)

    print(f"LOADING IMAGE FROM: {image_path}")
    image = Image.open(image_path)

    #DEFINE PROMPT
    # prompt = "detect elephant\n"

    #Preprocess inputs
    #print("Preparing inputs...")
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device, dtype=dtype)

    #Generate response
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.6,         # nižšia hodnota = menej halucinácií
            top_p=0.9, 
            repetition_penalty=1.2,
            no_repeat_ngram_size=8
        )

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

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
