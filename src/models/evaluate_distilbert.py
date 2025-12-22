import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from label_mapping import LABEL2ID, ID2LABEL

# -----------------------
# 1. Load test data
# -----------------------
test_df = pd.read_csv("data/splits/test.csv")

# Clean NaN / empty
test_df = test_df.dropna(subset=["cleaned_email", "category"])
test_df = test_df[test_df["cleaned_email"].str.strip() != ""]
test_df["label"] = test_df["category"].map(LABEL2ID)

# -----------------------
# 2. Tokenizer
# -----------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["cleaned_email"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------
# 3. Load trained model
# -----------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "models/category/distilbert",
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
# 5. Evaluate
# -----------------------
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(test_ds)

# -----------------------
# 6. Detailed report
# -----------------------
predictions = trainer.predict(test_ds)
y_true = test_df["label"].values
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n📊 DistilBERT Test Set Results\n")
print(results)
print("\n📄 Classification Report\n")
print(classification_report(y_true, y_pred, target_names=list(LABEL2ID.keys())))
