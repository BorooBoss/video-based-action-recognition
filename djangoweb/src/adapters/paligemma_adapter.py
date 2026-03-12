#normalize paligemma image co-ordinates to one format
import re
import re
from torchvision.ops import nms
import torch

def apply_nms(predictions, iou_threshold=0.3):
    if not predictions:
        return []

    by_label = {}
    for pred in predictions:
        label = pred["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(pred)

    result = []
    for label, preds in by_label.items():
        print(f"[NMS] {label}: {len(preds)} boxov pred NMS")
        boxes = torch.tensor(
            [[p["bbox"][1], p["bbox"][0], p["bbox"][3], p["bbox"][2]] for p in preds],
            dtype=torch.float32
        )
        scores = torch.ones(len(preds))
        keep = nms(boxes, scores, iou_threshold)
        result.extend([preds[i] for i in keep])

    return result

def convert(decoded, image_size=None): #parsing
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)"
    matches = re.findall(pattern, decoded)

    results = []
    for y1, x1, y2, x2, label in matches:
        results.append({
            "label": label,
            "bbox": [
                int(y1) / 1024,
                int(x1) / 1024,
                int(y2) / 1024,
                int(x2) / 1024
            ]
        })

    #results =  apply_nms(results, iou_threshold=0.3)
    return results


