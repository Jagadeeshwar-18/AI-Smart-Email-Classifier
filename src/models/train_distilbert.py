import os
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
from label_mapping import LABEL2ID, ID2LABEL

# -----------------------
# 1. Load & clean data
# -----------------------
train_df = pd.read_csv("data/splits/train.csv")
val_df   = pd.read_csv("data/splits/val.csv")

train_df = train_df.dropna(subset=["cleaned_email", "category"])
val_df   = val_df.dropna(subset=["cleaned_email", "category"])

train_df = train_df[train_df["cleaned_email"].str.strip() != ""]
val_df   = val_df[val_df["cleaned_email"].str.strip() != ""]

train_df["label"] = train_df["category"].map(LABEL2ID)
val_df["label"]   = val_df["category"].map(LABEL2ID)

# -----------------------
# 2. Tokenizer
# -----------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["cleaned_email"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df).map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)

# -----------------------
# 3. Model
# -----------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# -----------------------
# 4. Metrics
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# -----------------------
# 5. Training args
# -----------------------
os.makedirs("models/category/distilbert", exist_ok=True)

args = TrainingArguments(
    output_dir="models/category/distilbert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=100,
    save_total_limit=1,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------
# 6. Train
# -----------------------
trainer.train()
trainer.save_model("models/category/distilbert")
print("\n✅ DistilBERT model saved at: models/category/distilbert")