import json
import random

INPUT_FILE = "annotations.json"
OUTPUT_FILE = "annotations_fixed.json"

prompts = [
    "detect person",
    "detect weapon",
    "detect person; weapon"
]

with open(INPUT_FILE) as f:
    data = json.load(f)

for item in data:
    item["prefix"] = random.choice(prompts)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)
