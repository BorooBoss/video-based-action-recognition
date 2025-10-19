import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

#Model ID + device
MODEL_ID = "microsoft/Florence-2-base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



#Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype, #torch_dtype
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

#Load image from path
image_path = "/mnt/c/Users/boris/Desktop/5.semester/bp/source_files/frames/frame_0012.jpg"  
print(f"🖼️ Loading image from: {image_path}")
image = Image.open(image_path)

#DEFINE PROMPT
prompt = "<MORE_DETAILED_CAPTION>"


#Preprocess inputs
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=256,
    num_beams=3,
)

#Generate response
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text,
    task="<MORE_DETAILED_CAPTION>",
    image_size=(image.width, image.height)
)

print("\n" + "=" * 60)
print("FLORENCE 2 RESULT:")
print("=" * 60)
print(parsed_answer)
print("=" * 60)


"""
EXAMPLES 

Caption
prompt = "<CAPTION>"

Detailed Caption
prompt = "<DETAILED_CAPTION>"

More Detailed Caption
prompt = "<MORE_DETAILED_CAPTION>"

Caption to Phrase Grounding
caption to phrase grounding task requires additional text input, i.e. caption.

Caption to phrase grounding results format: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")

Object Detection
OD results format: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }

prompt = "<OD>"
run_example(prompt)

Dense Region Caption
Dense region caption results format: {'<DENSE_REGION_CAPTION>' : {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }

prompt = "<DENSE_REGION_CAPTION>"
run_example(prompt)

Region proposal
Dense region caption results format: {'<REGION_PROPOSAL>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}

prompt = "<REGION_PROPOSAL>"
run_example(prompt)

OCR
prompt = "<OCR>"
run_example(prompt)

OCR with Region
OCR with region output format: {'<OCR_WITH_REGION>': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}

prompt = "<OCR_WITH_REGION>"
run_example(prompt)
"""
