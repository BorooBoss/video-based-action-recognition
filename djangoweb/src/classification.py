from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://static.independent.co.uk/s3fs-public/thumbnails/image/2015/11/02/10/SA-police-shooting.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=[
    "a person physically attacking another person", #ABUSE / ASSAULT
    "police officers arresting a suspect", #AREST
    "a building on fire", #ARSON
    "a burglar breaking into a house", #BURGLARY
    "industrial accident with explosion", #EXPLOSION
    "people punching and kicking each other", #FIGHTING
    "a robbery in a store", #ROBBERY
    "a traffic accident" #ROAD ACCIDENT
    "a person shooting with a gun", #SHOOTING
    "a person stealing items from a shop", #SHOPLIFTING / STEALING
    "a suspect with weapon during a robbery", #ROBBERY
    "a person vandalizing property", #VANDALISM
], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities


print(probs)