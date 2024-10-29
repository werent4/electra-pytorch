import sys
sys.path.append('..')

from electra_pytorch.electra_pytorch_hf import ElectraHuggingFace, Trainer # type: ignore

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

generator_model_name = "microsoft/deberta-v3-small"
discriminator_model_name = generator_model_name

generator = AutoModelForMaskedLM.from_pretrained(generator_model_name).to(device)
discriminator = AutoModelForTokenClassification.from_pretrained(discriminator_model_name, num_labels=1).to(device)

generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)


electra_model = ElectraHuggingFace(
    generator,
    generator_tokenizer,
    discriminator,
    device = device
)

dataset = load_dataset("text", data_files={"train": "./datsets_txt/random_100_samples.txt"})
dataset = dataset['train'].shuffle(seed=42) 

def tokenize_function(examples):
    tokens = generator_tokenizer(examples["text"], truncation=True, padding=True, max_length= 512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=generator_tokenizer, mlm_probability=0.15, pad_to_multiple_of=8)

num_examples = len(tokenized_dataset)
batch_size = 7
num_train_epochs = 1

# Calculate total steps
total_steps = (num_examples // batch_size) * num_train_epochs

print("tokenized_dataset") 
print(tokenized_dataset) 
output_dir="./test_electra"

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    learning_rate= 1e-5,
    weight_decay = 0.02,
    num_train_epochs= num_train_epochs,
    per_device_train_batch_size=batch_size,
    save_steps=5,
    save_total_limit=1,
    warmup_steps=int(0.02 * total_steps),
    lr_scheduler_type='cosine',
    logging_steps=1,
    logging_dir="./logs",
    save_safetensors= True,
)

trainer = Trainer(
    model=electra_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator= data_collator
)

trainer.train()

generator_tokenizer.save_pretrained(f"{output_dir}/deberta-electra")
trainer.save_model(f"{output_dir}/deberta-electra") 