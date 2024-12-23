from transformers import AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path
from peft import PeftModel

class Classificator():

    HG_MODEL_ID = "FacebookAI/xlm-roberta-base" # Default model ID from the Mistral Hub
    MODEL_PATH = "model/xlm-roberta" # Default path to store the model
    LORA_PATH = "model/lora/doc-roberta" # Default path to store the LoRa adapter

    def __init__(self, hg_model_id:str=HG_MODEL_ID, model_path:str=MODEL_PATH, lora_path=LORA_PATH, half_precision:bool=False):
        self.hg_model_id = hg_model_id
        self.model_path = Path.home() / model_path
        self.loras_path = Path.home() / lora_path
        self.half_precision = half_precision

        # Load base model from local directory
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            if self.half_precision:
                self.model = self.model.half()

        except Exception as e:
            raise RuntimeError(f"Failed to load the base model: {e}")
        
        # Wrap model as Peft model
        try:
            self.model = PeftModel(self.model, task_type='MASKED_LM')

        except Exception as e:
                raise RuntimeError(f"Failed to initialize PEFT model: {e}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load the tokenizer : {e}")

        # Load LoRa adapter
        try:
            self.model.load_adapter(str(lora_path), adapter_name="doc-roberta")
            print(f"Loaded LoRA adapter: doc-roberta")

        except Exception as e:
            print(f"Error loading LoRA adapter 'doc-roberta': {e}")

    # Inference on the model
    def classify(self, text:str) -> str:
        try:
            input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**input)
            return output
        
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""
