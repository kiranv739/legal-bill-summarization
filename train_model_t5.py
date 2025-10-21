from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch

# === Load dataset and split ===
dataset = load_dataset("json", data_files="dataset.jsonl")["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === Load tokenizer & model ===
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# === Tokenization function with updated limits ===
def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["text"],
        max_length=1024,           # ✅ input length for bills
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        batch["summary"],
        max_length=512,           # ✅ summary length
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === Tokenize datasets ===
train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

# === Data collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# === Training Arguments with best model saving ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./flant5_model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=5,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True,
    generation_max_length=512,
    fp16=False,
    bf16=True,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# === Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Train ===
trainer.train()

# === Save best model manually at end ===
model.save_pretrained("./flant5_model")
tokenizer.save_pretrained("./flant5_model")