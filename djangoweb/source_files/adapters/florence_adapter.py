def convert(result, image_size=None):
    if "<OD>" in result:
        return {
            "<OD>": {
                "bboxes": result["<OD>"]["bboxes"],
                "labels": result["<OD>"]["labels"]
            }
        }
    return {}