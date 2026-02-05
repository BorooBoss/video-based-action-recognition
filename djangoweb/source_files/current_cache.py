import gc, torch

class CurrentCache:
    def __init__(self):
        self.model_id = None 
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        #self.name = None
    
    def unload_model(self):
        if self.model is not None:
            try:
                del self.model
            except:
                pass

        if self.processor is not None:
            del self.processor

        self.model = None
        self.processor = None
        self.model_id = None
        self.device = None
        self.dtype = None
        #self.name = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("OLD MODEL CLEARED FROM MEMORY\n")


    def switch_model(self, model_id, model, processor, device, dtype):
        if self.model is not None and self.model_id != model_id :
            self.unload() #TOTO pozriet

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        #self.name = name

cache = CurrentCache() # ONE AND ONLY instance of ModelManager