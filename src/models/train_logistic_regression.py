import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from vectorizer import get_vectorizer
import joblib

# Load data
train = pd.read_csv("data/splits/train.csv")
test = pd.read_csv("data/splits/test.csv")

# FIX: remove NaN + empty text
train = train.dropna(subset=["cleaned_email", "category"])
test = test.dropna(subset=["cleaned_email", "category"])

train = train[train["cleaned_email"].str.strip() != ""]
test = test[test["cleaned_email"].str.strip() != ""]

X_train = train["cleaned_email"].astype(str)
y_train = train["category"]

X_test = test["cleaned_email"].astype(str)
y_test = test["category"]

pipeline = Pipeline([
    ("tfidf", get_vectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(pipeline, "models/category/logistic_regression.pkl")
