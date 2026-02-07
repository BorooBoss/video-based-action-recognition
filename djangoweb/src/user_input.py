import re


class UserInput:
    def __init__(self):
        self.image = None
        self.base_prompt = None #detect or <OD>
        self.prompt_type = None #DETECT or CAPTION or DESCRIBE
        self.model_name = None
        self.prompt_input = "" #dog
        self.full_prompt = "" #detect dog

    #dictionary for user-friendly use
    PROMPT_MAP = {
        "google/paligemma2-3b-pt-224": {
            "DETECT": "detect",
            "SIMPLE CAPTION": "cap en",
            "STANDARD CAPTION": "caption en",
            "DETAILED CAPTION": "describe en",
            "VQA": "answer en"
        },
        "google/paligemma2-3b-mix-224": {
            "DETECT": "detect",
            "SIMPLE CAPTION": "cap en",
            "STANDARD CAPTION": "caption en",
            "DETAILED CAPTION": "describe en",
            "VQA": "answer en"
        },
        "microsoft/Florence-2-large": {
            "DETECT": "<OD>",
            "SIMPLE CAPTION": "<CAPTION>",
            "STANDARD CAPTION": "<DETAILED_CAPTION>",
            "DETAILED CAPTION": "<MORE_DETAILED_CAPTION>",
        },
        "microsoft/Florence-2-large-ft": {
            "DETECT": "<OD>",
            "SIMPLE CAPTION": "<CAPTION>",
            "STANDARD CAPTION": "<DETAILED_CAPTION>",
            "DETAILED CAPTION": "<MORE_DETAILED_CAPTION>",
            "VQA": "<VQA>"
        },
        "Qwen/Qwen3-VL-2B-Instruct": {
            "VQA": "",
        },
        "OpenGVLab/InternVL3_5-2B": {
            "VQA": "",
        }
    }

    def set_prompt_type(self):
        if "<OD>" in self.base_prompt or "detect" in self.base_prompt:
            self.prompt_type = "DETECT"
        elif "<CAPTION>" in self.base_prompt or "cap en" in self.base_prompt:
            self.prompt_type = "SIMPLE CAPTION"
        elif "<DETAILED_CAPTION>" in self.base_prompt or "caption en" in self.base_prompt:
            self.prompt_type = "STANDARD CAPTION"
        elif "<MORE_DETAILED_CAPTION>" in self.base_prompt or "describe en" in self.base_prompt:
            self.prompt_type = "DETAILED CAPTION"
        elif "<VQA>" in self.base_prompt or "answer en" in self.base_prompt or "VQA" in self.base_prompt:
            self.prompt_type = "VQA"

    def set_base_prompt(self):
        model_type = None
        for key in self.PROMPT_MAP:
            #print(key, self.model_name)
            if key.lower() == self.model_name.lower():
                model_type = key
                break

        if model_type is None:
            return None
        prompt_config = self.PROMPT_MAP[model_type]
        if self.prompt_type in prompt_config:
            self.base_prompt = prompt_config[self.prompt_type]
        else:
            return None  #prompt_type neexistuje v mape

        self.full_prompt = self.base_prompt
        if self.prompt_input:
            self.full_prompt = f"{self.base_prompt} {self.prompt_input}"

        return model_type


    def split_if_needed(self, prompt_type, prompt_input):
        prompt_input = prompt_input.strip()
        if not prompt_input:
            return [""]

        if prompt_type == "DETECT": # ";" in DETECT -> split
            if ";" in prompt_input:
                return [p.strip() for p in prompt_input.split(";") if p.strip()]
            return [prompt_input]

        if prompt_type == "VQA": # ";" in VQA -> split
            if ";" in prompt_input:
                return [p.strip() for p in prompt_input.split(";") if p.strip()]

            questions = re.findall(r"[^?]+\?", prompt_input)

            if len(questions) <= 1:
                return [prompt_input]

            #case if more than one "?" in VQA
            return [q.strip() for q in questions if q.strip()]

        return [prompt_input]


