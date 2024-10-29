import sys
sys.path.append('../BioLMs')
sys.path.append('/home/werent4/BioLMs/electra-pytorch')

from electra_pytorch.electra_pytorch_hf import ElectraHuggingFace, Trainer # type: ignore

from datasets import load_from_disk
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments


generator_model_name = "microsoft/deberta-v3-small"
discriminator_model_name = generator_model_name

generator = AutoModelForMaskedLM.from_pretrained(generator_model_name)
discriminator = AutoModelForTokenClassification.from_pretrained(discriminator_model_name, num_labels=1)

generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)


electra_model = ElectraHuggingFace(
    generator,
    generator_tokenizer,
    discriminator,
)

tokenized_dataset = load_from_disk("../datsets_txt/test_ds/BIO-Plain-text-tokenized-100-samples").select(range(10))
tokenized_dataset.shuffle(seed=42) 

data_collator = DataCollatorForLanguageModeling(tokenizer=generator_tokenizer, mlm_probability=0.15)

num_examples = len(tokenized_dataset)
batch_size = 1
num_train_epochs = 1

# Calculate total steps
total_steps = (num_examples // batch_size) * num_train_epochs

print("tokenized_dataset") 
print(tokenized_dataset) 
output_dir="../test_electra"

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