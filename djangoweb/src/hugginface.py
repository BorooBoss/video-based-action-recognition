import os
from dotenv import load_dotenv
#hugginface login

from huggingface_hub import login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

from huggingface_hub import whoami

user_info = whoami()
print(user_info)

