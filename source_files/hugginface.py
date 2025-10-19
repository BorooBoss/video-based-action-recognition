import os
from dotenv import load_dotenv

from huggingface_hub import login

# Replace 'your_token_here' with your actual token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

from huggingface_hub import whoami

user_info = whoami()
print(user_info)
