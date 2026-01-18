#!/usr/bin/env python3
"""
Fine-tune rubert-tiny2 for Armenian museum record classification
"""

import os
# Force CPU usage - disable MPS to avoid OOM
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_DISABLE'] = '1'  # Completely disable MPS

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Force CPU
#torch.set_default_device('cuda')

print("=" * 80)
print("FINE-TUNING RUBERT-TINY2 FOR ARMENIAN CLASSIFICATION")
print("=" * 80)

# Use CPU to avoid MPS OOM issues
# MPS has limited memory and causes OOM even with small batch sizes
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device} (CPU training to avoid MPS OOM)")

# Model selection
MODEL_NAME = "sergeyzh/rubert-mini-sts"
print(f"Model: {MODEL_NAME}")
print()

# Load datasets
def load_jsonl(filepath):
    """Load JSONL dataset"""
    data = {'text': [], 'label': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data['text'].append(item['text'])
            data['label'].append(item['label'])
    return Dataset.from_dict(data)

print("Loading datasets...")
train_dataset = load_jsonl('training_data.jsonl')
#train_dataset = load_jsonl('train_claude_labeled.jsonl')
val_dataset = load_jsonl('val.jsonl')
test_dataset = load_jsonl('test.jsonl')

print(f"  Train: {len(train_dataset):,} examples")
print(f"  Val:   {len(val_dataset):,} examples")
print(f"  Test:  {len(test_dataset):,} examples")
print()

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "not_armenian", 1: "armenian"},
    label2id={"not_armenian": 0, "armenian": 1}
)

# Explicitly move model to CPU
model = model.to(device)

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# Tokenize datasets
def tokenize_function(examples):
    """Tokenize text with truncation"""
    return tokenizer(
        examples['text'],
        padding=False,  # Will be done by data collator
        truncation=True,
        max_length=1024
    )

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print("Tokenization complete.")
print()

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Training arguments
output_dir = "./rubert-mini-armenian"
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps = 1000,
    save_strategy="steps",
    save_steps = 1000,
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # CPU has more memory than MPS
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    max_grad_norm = 1.0,
    label_smoothing_factor = 0.05,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,      # Tells Trainer that higher F1 is better
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    fp16=True,  # CPU doesn't support fp16
    optim="adamw_torch_fused",      # Faster GPU optimizer
    dataloader_num_workers=4,  # Disable multiprocessing to avoid spawn issues
    use_cpu=False,  # Force CPU usage
)

print("Training configuration:")
print(f"  Batch size (train): {training_args.per_device_train_batch_size}")
print(f"  Batch size (eval):  {training_args.per_device_eval_batch_size}")
print(f"  Learning rate:      {training_args.learning_rate}")
print(f"  Epochs:             {training_args.num_train_epochs}")
print(f"  Output directory:   {output_dir}")
print()

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stops if no improvement after 3 checks
)

# Train
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)
trainer.train()

# Evaluate on test set
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)
test_results = trainer.evaluate(tokenized_test)

print("\nTest Results:")
for metric, value in test_results.items():
    if not metric.startswith('eval_'):
        continue
    metric_name = metric.replace('eval_', '')
    print(f"  {metric_name}: {value:.4f}")

# Save final model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")

print(f"\nModel saved to: {output_dir}/final")
print("\nFine-tuning complete!")
print()

# Print sample predictions
print("=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

# Load test data for samples
with open('test.jsonl', 'r', encoding='utf-8') as f:
    test_samples = [json.loads(line) for line in f]

# Get predictions on a few samples
sample_texts = [s['text'][:200] for s in test_samples[:10]]
sample_labels = [s['label'] for s in test_samples[:10]]

inputs = tokenizer(sample_texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    predictions = torch.argmax(logits, dim=-1).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()

print("\nSample predictions:")
for i, (text, true_label, pred_label, prob) in enumerate(zip(sample_texts, sample_labels, predictions, probs), 1):
    print(f"\n{i}. Text: {text}...")
    print(f"   True: {'Armenian' if true_label == 1 else 'Not Armenian'}")
    print(f"   Pred: {'Armenian' if pred_label == 1 else 'Not Armenian'} (confidence: {prob[pred_label]:.2%})")
    print(f"   {'✓' if true_label == pred_label else '✗'}")
