import re


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
    return results
