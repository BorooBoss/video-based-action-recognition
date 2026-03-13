from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


_model = None
_processor = None

LABELS = [
    "a person physically attacking another person",                 #ABUSE / ASSAULT
    "police officers arresting a suspect",                          #AREST
    "a building on fire",                                           #ARSON
    "a burglar breaking into a house",                              #BURGLARY
    "industrial accident with explosion",                           #EXPLOSION
    "people punching and kicking each other",                       #FIGHTING
    "a robbery in a store",                                         #ROBBERY
    "a traffic accident",                                           #ROAD ACCIDENT
    "a person shooting with a gun",                                 #SHOOTING
    "a person stealing items from a shop",                          #SHOPLIFTING / STEALING
    "a suspect with weapon during a robbery",                       #ROBBERY
    "a person vandalizing property",                                #VANDALISM
    "normal everyday scene with no suspicious activity",            #NORMAL
]


def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor


def classify_image(image_path: str):
    model, processor = _load_model()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    scores = []
    for i in range(len(LABELS)):
        scores.append((LABELS[i], float(probs[i])))
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "label": scores[0][0],
        "score": scores[0][1],
        "top3": [[label, score] for label, score in scores[:3]],
    }


def classify_frames(image_paths: list):
    if not image_paths:
        return {"label": "No frames", "score": 0.0, "top3": []}

    model, processor = _load_model()

    accumulated = torch.zeros(len(LABELS))

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            accumulated += probs
        except Exception:
            continue

    avg = accumulated / len(image_paths)
    scores = []
    for i in range(len(LABELS)):
        scores.append((LABELS[i], float(avg[i])))

    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "label": scores[0][0],
        "score": scores[0][1],
        "top3": [[label, score] for label, score in scores[:3]],
    }

def classify_text(descriptions: list):

    if not descriptions:
        return {"label": "No descriptions", "score": 0.0, "top3": []}

    # filter out empty / None strings
    descriptions = [d for d in descriptions if d and str(d).strip()]
    if not descriptions:
        return {"label": "No descriptions", "score": 0.0, "top3": []}

    model, processor = _load_model()

    # ── encode LABELS once ────────────────────────────────────────────────────
    label_inputs = processor(
        text=LABELS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,          # CLIP token limit
    )
    with torch.no_grad():
        label_embeds = model.get_text_features(**label_inputs)   # [num_labels, 512]
        label_embeds = F.normalize(label_embeds, dim=-1)

    # ── encode each description and accumulate cosine-similarity scores ───────
    accumulated = torch.zeros(len(LABELS))
    valid_count  = 0

    for desc in descriptions:
        text = str(desc).strip()
        if not text:
            continue
        # CLIP max 77 tokens — truncate long captions automatically
        desc_inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            desc_embed = model.get_text_features(**desc_inputs)  # [1, 512]
            desc_embed = F.normalize(desc_embed, dim=-1)

        # cosine similarity → softmax → accumulate
        sims  = (desc_embed @ label_embeds.T).squeeze(0)         # [num_labels]
        probs = sims.softmax(dim=0)
        accumulated += probs
        valid_count  += 1

    if valid_count == 0:
        return {"label": "No valid descriptions", "score": 0.0, "top3": []}

    avg    = accumulated / valid_count
    scores = [(LABELS[i], float(avg[i])) for i in range(len(LABELS))]
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "label": scores[0][0],
        "score": scores[0][1],
        "top3":  [[label, score] for label, score in scores[:3]],
    }