from PIL import Image, ImageDraw, ImageFont

#drawing with florence normalized co-ordinates
def draw_boxes_florence(image_path, detections, output_path):


    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, max(0, y1 - 12)), obj["label"], fill="red")
    img.save(output_path)


#drawing with paligemma normalized co-ordinates
def draw_boxes_paligemma(image_path, coords_and_labels, output_path):

   image = Image.open(image_path)
   draw = ImageDraw.Draw(image)

   #draw = ImageDraw.Draw(image)
   width, height = image.size
   for obj in coords_and_labels:
       # Extract the bounding box coordinates
       y1, x1, y2, x2 = obj['bbox'][0] * height, obj['bbox'][1] * width, obj['bbox'][2] * height, obj['bbox'][3] * width

       draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
       draw.text((x1, y1), obj['label'], fill="red")
   print("som v draw funkcii v paligemme")
   image.save(output_path)
   return image