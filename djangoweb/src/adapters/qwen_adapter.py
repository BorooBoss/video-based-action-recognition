def convert(result, image_size):
    import json, re

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

    normalized = []
    for obj in result:
        bbox = obj.get("bbox_2d") or obj.get("bbox") or []
        label = obj.get("label", "object")
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        # Qwen3-VL pouziva 0-1000 normalizovany priestor
        normalized.append({
            "label": label,
            "bbox": [
                round(y1 / 1000, 3),  # y1_rel
                round(x1 / 1000, 3),  # x1_rel
                round(y2 / 1000, 3),  # y2_rel
                round(x2 / 1000, 3),  # x2_rel
            ]
        })

    return normalized