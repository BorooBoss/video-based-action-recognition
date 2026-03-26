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


def cosine_to_pct(score: float):
    """Convert cosine similarity [-1, 1] → percentage [0, 100]."""
    return float((score + 1) / 2)


def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor


def _encode_labels(processor, model) -> torch.Tensor:
    """Encode all labels once → normalized embeddings [num_labels, 512]."""
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


def _build_result(avg_sims: torch.Tensor) -> dict:
    """
    avg_sims: averaged cosine similarities in [-1, 1]
    Returns sorted dict with confidence in [0, 100].
    """
    scores = [(LABELS[i], cosine_to_pct(float(avg_sims[i]))) for i in range(len(LABELS))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {
        "label": scores[0][0],
        "confidence": scores[0][1],
        "top3": [[label, score] for label, score in scores[:3]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def classify_image(image_path: str) -> dict:
    """Classify a single image via cosine similarity (no softmax)."""
    model, processor = _load_model()
    label_embeds = _encode_labels(processor, model)

    image = Image.open(image_path).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_embed = model.get_image_features(**img_inputs)
    img_embed = F.normalize(img_embed, dim=-1)

    sims = (img_embed @ label_embeds.T).squeeze(0)  # [-1, 1]
    return _build_result(sims)


def classify_frames(image_paths: list) -> dict:
    """
    Classify a video clip by averaging cosine similarities across frames.
    No softmax – each label scored independently.
    """
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

            sims = (img_embed @ label_embeds.T).squeeze(0)  # [-1, 1]
            accumulated += sims
            valid += 1
        except Exception:
            continue

    if valid == 0:
        return {"label": "No valid frames", "confidence": 0.0, "top3": []}

    return _build_result(accumulated / valid)  # average BEFORE converting to %


def classify_text(descriptions: list) -> dict:
    """
    Classify from text descriptions via cosine similarity.
    No softmax – each label scored independently.
    """
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

        sims = (desc_embed @ label_embeds.T).squeeze(0)  # [-1, 1], NO softmax
        accumulated += sims

    return _build_result(accumulated / len(descriptions))  # average BEFORE converting to %