class UserInput:
    def __init__(self):
        self.image = None
        self.base_prompt = None #detect
        self.prompt_type = None #DETECT
        self.model_name = None
        self.addition = "" #dog
        self.full_prompt = "" #detect dog


    PROMPT_MAP = {
        "google/paligemma2-3b-pt-224": {
            "CAP": "cap en",
            "CAPTION": "caption en",
            "DESCRIBE": "describe en",
            "VQA": "answer en"
        },
        "google/paligemma2-3b-mix-224": {
            "DETECT": "detect",
            "CAP": "cap en",
            "CAPTION": "caption en",
            "DESCRIBE": "describe en",
            "VQA": "answer en"

        },
        "microsoft/Florence-2-base": {
            "DETECT": "<OD>",
            "CAP": "<CAPTION>",
            "CAPTION": "<DETAILED_CAPTION>",
            "DESCRIBE": "<MORE_DETAILED_CAPTION>",
        },
        "microsoft/Florence-2-base-ft": {
            "DETECT": "<OD>",
            "CAP": "<CAPTION>",
            "CAPTION": "<DETAILED_CAPTION>",
            "DESCRIBE": "<MORE_DETAILED_CAPTION>",
            "VQA": "<VQA>"
        },
        "qwen-testing": {
            "VQA": "",
        }
        # pridať ďalšie modely
    }

    def set_prompt_type(self):
        if "<OD>" in self.base_prompt or "detect" in self.base_prompt:
            self.prompt_type = "DETECT"
        elif "<CAPTION>" in self.base_prompt or "cap en" in self.base_prompt:
            self.prompt_type = "CAP"
        elif "<DETAILED_CAPTION>" in self.base_prompt or "caption en" in self.base_prompt:
            self.prompt_type = "CAPTION"
        elif "<MORE_DETAILED_CAPTION>" in self.base_prompt or "describe en" in self.base_prompt:
            self.prompt_type = "DESCRIBE"
        elif "<VQA>" in self.base_prompt or "answer en" in self.base_prompt or "VQA" in self.base_prompt:
            self.prompt_type = "VQA"

    def set_base_prompt(self):
        model_type = None
        for key in self.PROMPT_MAP:
            print(key, self.model_name)
            if key.lower() == self.model_name.lower():
                model_type = key
                break

        if model_type is None:
            print("hihihi")
            return None
        prompt_config = self.PROMPT_MAP[model_type]

        if self.prompt_type in prompt_config:
            self.base_prompt = prompt_config[self.prompt_type]
        else:
            return None  # prompt_type neexistuje v mape

        self.full_prompt = self.base_prompt
        if self.addition:
            self.full_prompt = f"{self.base_prompt} {self.addition}"

        return model_type

