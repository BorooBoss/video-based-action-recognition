from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://static.independent.co.uk/s3fs-public/thumbnails/image/2015/11/02/10/SA-police-shooting.jpg"
image = Image.open(requests.get(url, stream=True).raw)

labels = [
    "a person physically attacking another person",
    "police officers arresting a suspect",
    "a building on fire",
    "a burglar breaking into a house",
    "industrial accident with explosion",
    "people punching and kicking each other",
    "a robbery in a store",
    "a traffic accident",
    "a person shooting with a gun",
    "a person stealing items from a shop",
    "a suspect with weapon during a robbery",
    "a person vandalizing property",
]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)[0]

#print(probs)

best_idx = probs.argmax().item()
print(f"Label: {labels[best_idx]}")
print(f"Score: {probs[best_idx]:.1%}")