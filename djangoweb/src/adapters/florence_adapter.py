#normalize florence image co-ordinates to one format
def convert(result, image_size=None):
    results = []

    if "<OD>" not in result:
        return results

    bboxes = result["<OD>"].get("bboxes", [])
    labels = result["<OD>"].get("labels", [])

    # bezpečnosť – keby náhodou nesedel počet
    count = min(len(bboxes), len(labels))

    for i in range(count):
        bbox = bboxes[i]
        label = labels[i]

        # ak chceš normalizovať podľa veľkosti obrázka
        if image_size:
            h, w = image_size
            y1, x1, y2, x2 = bbox
            bbox = [
                y1 / h,
                x1 / w,
                y2 / h,
                x2 / w
            ]

        results.append({
            "label": label,
            "bbox": bbox
        })

    return results
