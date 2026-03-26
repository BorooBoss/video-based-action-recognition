def normalize_qwen(result, image_size):
    import json, re

    # Qwen vracia markdown code block — vytiahni JSON
    if isinstance(result, str):
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', result, re.DOTALL)
        if match:
            result = match.group(1)
        try:
            result = json.loads(result)
        except Exception:
            return []

    if not isinstance(result, list):
        return []

    h, w = image_size
    normalized = []
    for obj in result:
        bbox = obj.get("bbox_2d") or obj.get("bbox") or []
        label = obj.get("label", "object")
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # → paligemma formát: [y1_rel, x1_rel, y2_rel, x2_rel]
        normalized.append({
            "label": label,
            "bbox": [
                round(y1 / h, 3),
                round(x1 / w, 3),
                round(y2 / h, 3),
                round(x2 / w, 3),
            ]
        })
    return normalized