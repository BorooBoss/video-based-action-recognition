from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import tempfile
import os
import uvicorn

app = FastAPI()


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)
#if not cpu
if device == "cpu":
    model.to("cpu")
    print("BEZIM NA CPU")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print(f"Qwen3 model loaded on {device}")


@app.post("/predict")
async def predict(
        image: UploadFile = File(...),
        prompt: str = Form(...)
):
    try:
        #save temp frame
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image_bytes = await image.read()
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)

            generated = model.generate(**inputs, max_new_tokens=128)
            trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            return JSONResponse({"result": answer})

        finally:
            #delete temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Qwen3-VL-2B", "device": device}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)