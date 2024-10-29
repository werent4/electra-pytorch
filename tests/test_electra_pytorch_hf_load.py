import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForTokenClassification
from electra_pytorch.electra_pytorch_hf import ElectraHuggingFace #type: ignore

output_dir = "../test_electra/deberta-electra"

electra_model = ElectraHuggingFace.from_pretrained(output_dir= output_dir)
print("Loaded!")