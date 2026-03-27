from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

_model = None
_processor = None

LABELS = [
    "a person physically attacking another person",
    "police officers arresting a suspect",
    "a building on fire",
    "a burglar breaking into a house",
    "industrial accident with explosion",
    "a street fight with multiple people",
    "a robbery in a store",
    "a traffic accident on road",
    "a person aiming or shooting with a gun",
    "a person stealing items from a shop",
    "person with a weapon threatening others",
    "a person vandalizing property",

    "people walking on a street",
    "empty room or corridor"
    "peaceful place",
]

TEMPERATURE = 50.0  # vyššie = ostrejšie rozdelenie, skús 30-100

def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor


def _encode_labels(processor, model) -> torch.Tensor:
    inputs = processor(
        text=LABELS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        embeds = model.get_text_features(**inputs)
    return F.normalize(embeds, dim=-1)


def _build_result(sims: torch.Tensor) -> dict:
    # Softmax s temperature — dáva zmysluplné percentá ktoré sa sčítajú na 100%
    probs = F.softmax(sims * TEMPERATURE, dim=0) * 100.0
    scores = [(LABELS[i], float(probs[i])) for i in range(len(LABELS))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {
        "label": scores[0][0],
        "confidence": scores[0][1],
        "top3": [[label, score] for label, score in scores[:3]],
    }


def classify_image(image_path: str) -> dict:
    model, processor = _load_model()
    label_embeds = _encode_labels(processor, model)

    image = Image.open(image_path).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_embed = model.get_image_features(**img_inputs)
    img_embed = F.normalize(img_embed, dim=-1)

    sims = (img_embed @ label_embeds.T).squeeze(0)
    return _build_result(sims)


def classify_frames(image_paths: list) -> dict:
    if not image_paths:
        return {"label": "No frames", "confidence": 0.0, "top3": []}

    model, processor = _load_model()
    label_embeds = _encode_labels(processor, model)

    accumulated = torch.zeros(len(LABELS))
    valid = 0

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            img_inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_embed = model.get_image_features(**img_inputs)
            img_embed = F.normalize(img_embed, dim=-1)

            sims = (img_embed @ label_embeds.T).squeeze(0)
            accumulated += sims
            valid += 1
        except Exception:
            continue

    if valid == 0:
        return {"label": "No valid frames", "confidence": 0.0, "top3": []}

    # Priemerujeme similarities, potom softmax
    return _build_result(accumulated / valid)


def classify_text(descriptions: list) -> dict:
    if not descriptions:
        return {"label": "No descriptions", "confidence": 0.0, "top3": []}

    descriptions = [str(d).strip() for d in descriptions if str(d).strip()]
    if not descriptions:
        return {"label": "No valid descriptions", "confidence": 0.0, "top3": []}

    model, processor = _load_model()
    label_embeds = _encode_labels(processor, model)

    accumulated = torch.zeros(len(LABELS))

    for desc in descriptions:
        desc_inputs = processor(
            text=[desc],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            desc_embed = model.get_text_features(**desc_inputs)
        desc_embed = F.normalize(desc_embed, dim=-1)

        sims = (desc_embed @ label_embeds.T).squeeze(0)
        accumulated += sims

    return _build_result(accumulated / len(descriptions))