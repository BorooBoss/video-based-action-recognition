def convert(result, image_size=None):
    if "<OD>" in result:
        print("hububububu")
        return {
            "<OD>": {
                "bboxes": result["<OD>"]["bboxes"],
                "labels": result["<OD>"]["labels"]
            }
        }
    return {}