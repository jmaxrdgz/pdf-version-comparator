# Training

This section contains all training scripts to adapt the app to a custom purpose.  

The *mistral_finetune_7b* notebook is the default Mistral procedure to finetune it's 7B model. It aims to create LoRa adapters for the interpretor model. Once the LoRa trained and generated it must be added to "model/lora/lora_name.pt" path.
**Do not clone mistral finetune repo in this repo, training is supposed to be performed in a Google Colab notebook.**