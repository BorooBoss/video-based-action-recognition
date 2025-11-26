from transformers import AutoModelForCausalLM, AutoProcessor
import requests
import torch
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("nirusanan/Florence-2_FT_VQA", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("nirusanan/Florence-2_FT_VQA", trust_remote_code=True)

prompt = "DocVQA abcd?"


url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYoDmTW0JLiW2VNGbT0OozJqi3biTszTOQKCnuuvdPuFUzHS6gQwFdCLQi7mmY7hGD150&usqp=CAU"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="DocVQA", image_size=(image.width, image.height))

print(parsed_answer)