import os
from dotenv import load_dotenv

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login


#Login to Hugging Face
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

#Model ID + device
MODEL_ID = "google/paligemma2-3b-mix-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print("🚀 Loading Paligemma 2 model using device: {device}")

#Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def predict(image_path, prompt="describe\n"):
    #Load image from path
    image_path = "/mnt/c/Users/boris/Desktop/5.semester/bp/source_files/samples/test2.jpg"
    print(f"🖼️ Loading image from: {image_path}")
    image = Image.open(image_path)

    #DEFINE PROMPT
    prompt = "detect elephant\n"




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
