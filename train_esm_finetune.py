import torch
from transformers import (
    EsmForMaskedLM,
    EsmTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# === Load Dataset ===
# Prepare a CSV with a column named "sequence"
dataset = load_dataset("csv", data_files={"train": "kinase_sequences.csv"})
dataset = dataset["train"].train_test_split(test_size=0.1)
train_ds, val_ds = dataset["train"], dataset["test"]

# === Load Tokenizer and Model ===
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)

# === Tokenize Sequences ===
def tokenize_function(batch):
    return tokenizer(
        batch["sequence"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=["sequence"])
tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=["sequence"])

# === Data Collator for Masked LM ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

# === Training Configuration ===
training_args = TrainingArguments(
    output_dir="./esm_finetuned_kinase",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    fp16=True,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    logging_steps=100,
    report_to="none",
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Run Training ===
trainer.train()
trainer.save_model("./esm_finetuned_kinase_final")

print("✅ Fine-tuning complete. Model saved to ./esm_finetuned_kinase_final")
