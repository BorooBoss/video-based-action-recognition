import re

def convert(result):
    entries = [e.strip() for e in result.split(";") if e.strip()]
    bboxes, labels = [], []

    for entry in entries:
        locs = re.findall(r"<loc(\d{4})>", entry)
        if len(locs) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in locs]
        label = entry.split(">")[-1].strip()

        bboxes.append([x1, y1, x2, y2])
        labels.append(label)

    return {
        "<OD>": {
            "bboxes": bboxes,
            "labels": labels
        }
    }
