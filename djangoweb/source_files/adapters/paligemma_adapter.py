import re


def convert(raw: str, image_size=None):
    """
    Convert PaliGemma raw OUTPUT into pixel bounding boxes.
    PaliGemma format: <locXMIN><locYMIN><locXMAX><locYMAX> label
    Confirmed format: [x_min, y_min, x_max, y_max] (Interpretation 2)
    All values are in 0-1024 range.
    """
    pattern = r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s*([a-zA-Z0-9_-]+)"
    matches = re.findall(pattern, raw)

    if not matches:
        return {"<OD>": {"bboxes": [], "labels": []}}

    bboxes = []
    labels = []

    if image_size:
        orig_w, orig_h = image_size

        for x_min, y_min, x_max, y_max, label in matches:
            # Interpretation 2: [x,y,x,y] - CONFIRMED CORRECT
            x1 = (float(x_min) / 1024.0) * orig_w
            y1 = (float(y_min) / 1024.0) * orig_h
            x2 = (float(x_max) / 1024.0) * orig_w
            y2 = (float(y_max) / 1024.0) * orig_h

            bboxes.append([x1, y1, x2, y2])
            labels.append(label)
    else:
        for x_min, y_min, x_max, y_max, label in matches:
            x1 = float(x_min) / 1024.0
            y1 = float(y_min) / 1024.0
            x2 = float(x_max) / 1024.0
            y2 = float(y_max) / 1024.0

            bboxes.append([x1, y1, x2, y2])
            labels.append(label)

    return {
        "<OD>": {
            "bboxes": bboxes,
            "labels": labels
        }
    }