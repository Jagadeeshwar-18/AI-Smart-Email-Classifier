import os
import torch
import torch.nn.functional as F
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../models/category/distilbert")
)

URGENCY_MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../models/urgency/urgency_lr.pkl")
)

# ---------------- CATEGORY MODEL (DistilBERT) ----------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

category_model = DistilBertForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_PATH,
    local_files_only=True
)
category_model.eval()

CATEGORY_LABELS = ["complaint", "request", "feedback", "spam"]

# ---------------- URGENCY MODEL (Logistic Regression) ----------------
urgency_model = joblib.load(URGENCY_MODEL_PATH)

# ---------------- PREDICTION ----------------
def predict_email(text: str):
    if not text or not text.strip():
        return {
            "category": "unknown",
            "urgency": "low",
            "confidence": 0.0
        }

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = category_model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    cat_idx = torch.argmax(probs, dim=1).item()

    category = CATEGORY_LABELS[cat_idx]
    confidence = round(probs.max().item(), 2)

    urgency = urgency_model.predict([text])[0]

    return {
        "category": category,
        "urgency": urgency,
        "confidence": confidence
    }
