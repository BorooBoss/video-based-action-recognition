import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to("cpu")
    print("BEZIM NA CPU")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": args.image},
            {"type": "text", "text": args.prompt},
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

print(answer)