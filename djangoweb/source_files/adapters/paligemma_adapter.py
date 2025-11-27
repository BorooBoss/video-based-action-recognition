import re


def convert(decoded_output, image_size=None): #parsing
   # Regex pattern to match four <locxxxx> tags and the label at the end (e.g., 'cat')
   loc_pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s+(\w+)"
   matches = re.findall(loc_pattern, decoded_output)
   coords_and_labels = []

   for match in matches:
       # Extract the coordinates and label
       y1 = int(match[0]) / 1024
       x1 = int(match[1]) / 1024
       y2 = int(match[2]) / 1024
       x2 = int(match[3]) / 1024
       label = match[4]
       coords_and_labels.append({
           'label': label,
           'bbox': [y1, x1, y2, x2]
       })
   print("som v convert paligemma")
   return coords_and_labels

"""
def convert(raw: str, image_size=None):
    
    Convert Paligemma raw output into real bounding boxes using image_size.
    image_size must be: (width, height)

    Paligemma uses normalized coordinates: 0–1023 range.
    

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
    
"""