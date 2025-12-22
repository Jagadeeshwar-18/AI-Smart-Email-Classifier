import pandas as pd
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from vectorizer import get_vectorizer


def main():
    # -----------------------------
    # 1. Load datasets
    # -----------------------------
    train = pd.read_csv("data/splits/train.csv")
    test = pd.read_csv("data/splits/test.csv")

    # -----------------------------
    # 2. FIX: Remove NaN + empty text
    # -----------------------------
    train = train.dropna(subset=["cleaned_email", "category"])
    test = test.dropna(subset=["cleaned_email", "category"])

    train = train[train["cleaned_email"].str.strip() != ""]
    test = test[test["cleaned_email"].str.strip() != ""]

    X_train = train["cleaned_email"].astype(str)
    y_train = train["category"]

    X_test = test["cleaned_email"].astype(str)
    y_test = test["category"]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # -----------------------------
    # 3. Build Pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("tfidf", get_vectorizer()),
        ("clf", MultinomialNB())
    ])

    # -----------------------------
    # 4. Train Model
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # 5. Evaluate Model
    # -----------------------------
    predictions = pipeline.predict(X_test)

    print("\n📊 Naive Bayes Classification Report\n")
    print(classification_report(y_test, predictions))

    # -----------------------------
    # 6. Save Model
    # -----------------------------
    joblib.dump(
        pipeline,
        "models/category/naive_bayes.pkl"
    )

    print("\n✅ Naive Bayes model saved at:")
    print("models/category/naive_bayes.pkl")


if __name__ == "__main__":
    main()
