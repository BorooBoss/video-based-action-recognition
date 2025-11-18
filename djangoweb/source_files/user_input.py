class UserInput:
    def __init__(self):
        self.image = None
        self.base_prompt = None
        self.prompt_type = None
        self.model_name = None
        self.addition = ""
        self.full_prompt = ""

    PROMPT_MAP = {
        "paligemma": {
            "DETECT": "detect",
            "CAPTION": "describe",
            "OCR": "ocr"
        },
        "florence": {
            "DETECT": "<OD>",
            "CAPTION": "<CAPTION>",
            "OCR": "<OCR>"
        },
        "qwen": {
            "DETECT": "detect",
            "CAPTION": "describe",
            "OCR": "ocr"
        }
        # pridať ďalšie modely
    }
    def set_prompt_type(self):
        if  ("<OD>" or "detect") in self.prompt:
            self.prompt_type = "DETECT"
        elif ("<CAPTION>" or "cap") in self.prompt:
            self.prompt_type = "CAPTION"
        elif ("<<DETAILED_CAPTION>>" or "CAPTION") in self.prompt:
            self.prompt_type = "CAPTION"

    def set_base_prompt(self):
        # Nájdi typ modelu v PROMPT_MAP
        model_type = None
        for key in self.PROMPT_MAP:
            if key in self.model_name.lower():
                model_type = key
                break

        if model_type is None:
            return None

        # Získaj mapku pre daný model
        prompt_config = self.PROMPT_MAP[model_type]

        # Nastav base_prompt podľa prompt_type
        if self.prompt_type in prompt_config:
            self.base_prompt = prompt_config[self.prompt_type]
        else:
            return None  # prompt_type neexistuje v mape

        # Vytvor finálny prompt
        self.full_prompt = self.base_prompt
        if self.addition:
            self.full_prompt = f"{self.base_prompt} {self.addition}"

        return model_type

