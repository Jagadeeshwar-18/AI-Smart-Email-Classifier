import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
train = pd.read_csv("data/splits/train.csv")
test  = pd.read_csv("data/splits/test.csv")

# Clean
for df in (train, test):
    df.dropna(subset=["cleaned_email", "urgency"], inplace=True)
    df = df[df["cleaned_email"].str.strip() != ""]

X_train, y_train = train["cleaned_email"], train["urgency"]
X_test,  y_test  = test["cleaned_email"],  test["urgency"]

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("\n📊 Urgency ML Classification Report\n")
print(classification_report(y_test, preds))

# Save
os.makedirs("models/urgency", exist_ok=True)
joblib.dump(pipeline, "models/urgency/urgency_lr.pkl")
print("✅ Urgency model saved at models/urgency/urgency_lr.pkl")
