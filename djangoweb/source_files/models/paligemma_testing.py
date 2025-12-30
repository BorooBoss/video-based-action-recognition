# paligemma_od_cuda.py
"""
Spustenie PaliGemma 2 mix (224) na CUDA pre jednoduché prompty typu: <OD>chair;table
- Vyžaduje: transformers, accelerate, torch, pillow, matplotlib
- Model: "google/paligemma2-3b-mix-224" (Hugging Face)
"""

import re
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

# --- Pomocné funkcie -------------------------------------------------------
def translate_od_prompt(user_prompt: str) -> str:
    """
    Prevedie vstup vo forme "<OD>obj1;obj2" -> prompt pre model: "detect obj1 ; obj2\n"
    Ak user_prompt == "<OD>" -> vráti "detect\n" (môžeš doplniť podľa potreby).
    """
    p = user_prompt.strip()
    if not p.startswith("<OD>"):
        raise ValueError("Prompt musí začínať '<OD>'")
    body = p[4:].strip()
    if body == "":
        return "detect\n"
    # očakávame objekty oddelené stredníkom alebo čiarkou
    objs = re.split(r"[;,]\s*", body)
    objs = [o.strip() for o in objs if o.strip()]
    # PaliGemma mix očakáva "detect obj ; obj"
    return "detect " + " ; ".join(objs) + "\n"

def parse_bboxes_from_text(text: str):
    """
    Pokus o parsovanie bounding boxov z textového výstupu modelu.
    Model môže vracať rôzne formáty; toto je robustné záchytávanie číselných sekvencií.
    Nájde všetky skupiny 4 čísel a vráti list (label, (x1,y1,x2,y2)) ak label prítomný.
    """
    results = []
    # Pokusíme sa najprv nájsť pattern: label [x1,y1,x2,y2] alebo label: x1,y1,x2,y2
    # Najjednoduchší heuristický prístup: nájdeme všetky čísla a rozdelíme podľa labelov.
    # Regex na label + čísla:
    for m in re.finditer(r"([A-Za-z0-9_ -]{1,40})\s*[:\[]\s*([0-9\.\-]+)[\s,]+([0-9\.\-]+)[\s,]+([0-9\.\-]+)[\s,]+([0-9\.\-]+)\s*[\]\n]?", text):
        label = m.group(1).strip()
        coords = tuple(float(x) for x in m.groups()[1:5])
        results.append((label, coords))
    # Ak nič nenájdené, skúsiť nájsť len 4-tice čísel a priradiť bez labelu
    if not results:
        for m in re.finditer(r"([0-9\.\-]+)[\s,]+([0-9\.\-]+)[\s,]+([0-9\.\-]+)[\s,]+([0-9\.\-]+)", text):
            coords = tuple(float(x) for x in m.groups())
            results.append(("object", coords))
    return results

def draw_bboxes(image: Image.Image, bboxes, save_path=None):
    """
    Nakreslí bounding boxy na PIL Image.
    bboxes = list of (label, (x1,y1,x2,y2)) - súradnice očakávame v relatívnych alebo absolútnych hodnotách.
    Ak sú súradnice v rozsahu 0..1, prepočítame ich na pixely.
    """
    img = image.convert("RGB")
    w,h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i,(label,(x1,y1,x2,y2)) in enumerate(bboxes):
        # ak sú v 0..1 rozsahu -> map na pixely
        if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
            x1_ = int(x1 * w); y1_ = int(y1 * h)
            x2_ = int(x2 * w); y2_ = int(y2 * h)
        else:
            x1_, y1_, x2_, y2_ = int(x1), int(y1), int(x2), int(y2)

        # jednoduché farby cyklicky
        color = (255, 0, 0, 120) if i % 3 == 0 else ((0,255,0,120) if i % 3 == 1 else (0,0,255,120))
        draw.rectangle([x1_, y1_, x2_, y2_], outline=color[:3], width=3)
        # label box
        text = label
        if font:
            text_size = draw.textsize(text, font=font)
        else:
            text_size = (len(text)*6, 11)
        draw.rectangle([x1_, y1_ - text_size[1] - 4, x1_ + text_size[0] + 4, y1_], fill=color)
        draw.text((x1_ + 2, y1_ - text_size[1] - 2), text, fill=(255,255,255), font=font)

    if save_path:
        img.save(save_path)
    return img

# --- Hlavná inference funkcia ---------------------------------------------
def run_detection(image_path_or_url: str, od_prompt: str, model_id: str = "google/paligemma2-3b-mix-224"):
    """
    Spustí model pre daný obraz a <OD> prompt. Vráti decoded text a list bboxes.
    """
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] používa sa device: {device}")

    # načítanie modelu a processoru
    # použiť torch_dtype=float16 pre GPU (FP16) - môžeš zmeniť na bfloat16 ak tvoje GPU podporuje
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    # načítaj obrázok (môže byť URL alebo lokálna cesta)
    image = load_image(image_path_or_url) if image_path_or_url.startswith("http") else Image.open(image_path_or_url).convert("RGB")

    # preložiť prompt <OD>... -> model prompt
    prompt = translate_od_prompt(od_prompt)
    print(f"[info] prerobený prompt pre model: {prompt!r}")

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(dtype).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print("[raw decode]\n", decoded)

    # pokus o parsovanie bboxes
    bboxes = parse_bboxes_from_text(decoded)
    return decoded, bboxes, image

# --- CLI ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PaliGemma2-mix-224 CUDA OD wrapper (<OD>object1;object2)")
    parser.add_argument("image", help="URL alebo lokalna cesta k obrazku")
    parser.add_argument("prompt", help="Prompt vo forme: <OD>chair;table  (alebo '<OD>')")
    parser.add_argument("--model", default="google/paligemma2-3b-mix-224", help="HF model id")
    parser.add_argument("--out", default="od_result.jpg", help="Kam ulozit vysledny obraz s bboxes")
    args = parser.parse_args()

    decoded, bboxes, image = run_detection(args.image, args.prompt, model_id=args.model)
    print("\n[decoded text]\n", decoded)
    print("\n[parsed bboxes]\n", bboxes)

    vis = draw_bboxes(image, bboxes, save_path=args.out)
    print(f"[done] Uložené: {args.out}")

if __name__ == "__main__":
    # Tu nastav svoje hodnoty
    image_path = "/mnt/c/Users/boris/Desktop/5.semester/bp/djangoweb/source_files/samples/test2.jpg"
    prompt = "<OD>giraffe"
    output_path = "od_result.jpg"

    decoded, bboxes, image = run_detection(image_path, prompt)
    print("\n[decoded text]\n", decoded)
    print("\n[parsed bboxes]\n", bboxes)

    vis = draw_bboxes(image, bboxes, save_path=output_path)
    print(f"[done] Uložené: {output_path}")