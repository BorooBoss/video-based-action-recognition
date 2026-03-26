#normalize paligemma image co-ordinates to one format
import re
import re
from torchvision.ops import nms
import torch

def apply_nms(predictions, iou_threshold=0.5):
    if not predictions:
        return []

    by_label = {}
    for pred in predictions:
        label = pred["label"]
        by_label.setdefault(label, []).append(pred)

    result = []
    for label, preds in by_label.items():
        if len(preds) == 1:
            result.extend(preds)
            continue

        boxes = torch.tensor(
            [[p["bbox"][1], p["bbox"][0], p["bbox"][3], p["bbox"][2]] for p in preds],
            dtype=torch.float32
        )
        # Skóre = plocha boxu (väčší box = vyššia priorita pre keep)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        scores = areas  # <-- toto dáva zmysel bez confidence score

        keep = nms(boxes, scores, iou_threshold)
        result.extend([preds[i] for i in keep])

    return result

def convert(decoded, image_size=None):
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)"
    matches = re.findall(pattern, decoded)

    results = []
    for y1, x1, y2, x2, label in matches:
        y1n, x1n, y2n, x2n = int(y1)/1024, int(x1)/1024, int(y2)/1024, int(x2)/1024

        y_min, y_max = min(y1n, y2n), max(y1n, y2n)
        x_min, x_max = min(x1n, x2n), max(x1n, x2n)

        if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
            continue

        if (x_max - x_min) > 0.95 and (y_max - y_min) > 0.95:
            continue

        results.append({
            "label": label,
            "bbox": [y_min, x_min, y_max, x_max]  # vždy min, max
        })

    #results = apply_nms(results, iou_threshold=0.5)
    return results


