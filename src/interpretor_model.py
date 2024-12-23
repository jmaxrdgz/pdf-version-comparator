from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
from pathlib import Path
from peft import PeftModel

class Interpretor():

    LORAS_PATH = "model/interpretor_lora" # Default path to LoRa folder
    HG_MODEL_ID = "mistralai/Mistral-7B-v0.3" # Default model ID from the Mistral Hub
    MODEL_PATH = "model/mistral-7B-v0.3" # Default path to store the model

    def __init__(self, hg_model_id:str=HG_MODEL_ID, model_path:str=MODEL_PATH, loras_path=LORAS_PATH, half_precision:bool=False):
        self.hg_model_id = hg_model_id
        self.model_path = Path.home() / model_path
        self.loras_path = Path.home() / loras_path
        self.half_precision = half_precision

        # Download the model if not present locally
        if not self.model_path.exists() or not any(self.model_path.iterdir()):
            self.model_path.mkdir(parents=True, exist_ok=True)
            login()
            snapshot_download(
                repo_id=self.hg_model_id, 
                allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
                local_dir=self.model_path
            )
            print(f"Model loaded at path: {self.model_path}")

        # Load base model from local directory
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
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
            self.model = PeftModel(self.model, task_type='CAUSAL_LM')

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

        # Load LoRa adapters
        self.loras = {}
        if self.loras_path.exists() and self.loras_path.is_dir():
            for lora_dir in self.loras_path.iterdir():
                if lora_dir.is_dir() and (lora_dir / "adapter_config.json").exists():
                    adapter_name = lora_dir.name
                    try:
                        self.model.load_adapter(str(lora_dir), adapter_name=adapter_name)
                        self.loras[adapter_name] = lora_dir
                        print(f"Loaded LoRA adapter: {adapter_name}")

                    except Exception as e:
                        print(f"Error loading LoRA adapter '{adapter_name}': {e}")
        else:
            print(f"LoRA adapters directory '{self.loras_path}' does not exist or is not a directory.")

    # Inference on the model
    def interpret_report(self, context:str, text:str) -> str:
        prompt = (
            "Given the following context:\n"
            f"{context}\n\n"
            "Interpret the following text:\n"
            f"{text}"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            interpretation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return interpretation
        
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""

    def set_lora(self, loras:list, add:bool=False):
        '''
        Allows to change fine-tuned configuration of the model
        - add=False : sets a new configuration and applies it 
        - add=True : add configuration to active configuration
        '''
        if not add:
            self.model.disable_adapter()
        for lora in loras:
            try:
                self.model.set_adapter(lora)

            except KeyError:
                print(f"Error: LoRa adapter {lora} is not loaded into the model.")