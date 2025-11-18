import re

def convert(raw: str, image_size=None):
    """
    Convert Paligemma raw output into real bounding boxes using image_size.
    image_size must be: (width, height)

    Paligemma uses normalized coordinates: 0–1023 range.
    """

    # 1) Extract <locxxxx>... pattern + label
    pattern = r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s*([a-zA-Z0-9_-]+)"
    matches = re.findall(pattern, raw)

    bboxes = []
    labels = []

    for x1, y1, x2, y2, label in matches:
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        # 2) Normalize if image size is provided
        if image_size:
            print("tu soooom")
            w, h = image_size
            x1 = (x1 / 1023.0) * w
            x2 = (x2 / 1023.0) * w
            y1 = (y1 / 1023.0) * h
            y2 = (y2 / 1023.0) * h

        bboxes.append([x1, y1, x2, y2])
        labels.append(label)
        print(labels)
        print(bboxes)

    return {
        "<OD>": {
            "bboxes": bboxes,
            "labels": labels
        }
    }
