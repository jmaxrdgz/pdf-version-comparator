from huggingface_hub import snapshot_download, login
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HG_MODEL_ID = "mistralai/Mistral-7B-v0.3" # Model ID from the Mistral Hub
MODEL_PATH = "models/mistral-7B-v0.3" 
TOKENIZER_PATH = "/models/tokenizer.model.v3"
HALF_PRECISION = False # If less than 16GB GPU memory available

class Interpretor():
    lora_adapters = {
        "example" : "models/lora/example.pt"
    }

    def __init__(self, hg_model_id:str, model_path:str, half_precision:bool=False):
        self.hg_model_id = hg_model_id
        self.model_path = Path.home().joinpath(model_path)
        self.half_precision = half_precision

        # Download the model if it doesn't exist on the local machine
        if not self.model_path.exists() or not any(self.model_path.iterdir()):
            self.model_path.mkdir(parents=True, exist_ok=True)
            login()
            snapshot_download(
                repo_id=self.hg_model_id, 
                allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
                local_dir=self.model_path
            )
            print(f"Model loaded at path: {self.model_path}")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        if self.half_precision:
            model = model.half()
        self.model = PeftModel(model)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # Load LoRa adapters
        for lora, path in self.lora_adapters.items():
            lora_path = Path(path)
            if lora_path.exists():
                self.model.load_adapter(lora_path, adapter_name=lora)
            else:
                print(f"Error: LoRa adapter file {lora_path} does not exist.")

    def set_lora(self, loras:list, add:bool=False):
        '''
        Allows to change fine-tuned configuration of the model
        - add=False : sets a new configuration and applies it 
        - add=True : add configuration to active configuration
        '''
        if not add:
            self.model.disable_adapter()
        for lora in loras:
            if lora in self.model.config.adapters:
                self.model.set_adapter(lora)
            else:
                print(f"Error: LoRa adapter {lora} is not loaded into the model.")

    # Inference on the model
    def interpret_report(self, context:str, text:str):
        prompt = (
            "Given the following context :\n"
            + context + "\n\n"
            + "Interpret the following text :\n"
            + text
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        interpretation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return interpretation
