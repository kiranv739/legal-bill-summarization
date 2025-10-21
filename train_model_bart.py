from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
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
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# === Tokenization function with updated limits ===
def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["text"],
        max_length=1024,           # ✅ increased from 512
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["summary"],
            max_length=512,         # ✅ increased from 128
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
    output_dir="./bart_model",
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
    generation_max_length=512,                # ✅ match summary length
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    load_best_model_at_end=True,              # ✅ enabled
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
model.save_pretrained("./bart_model")
tokenizer.save_pretrained("./bart_model")
