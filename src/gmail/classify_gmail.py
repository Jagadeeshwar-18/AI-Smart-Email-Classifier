from src.gmail.fetch_emails import fetch_emails
from src.inference.predict import predict_email

def classify_gmail_emails():
    raw_emails = fetch_emails()
    results = []

    for mail in raw_emails:
        text = f"{mail['subject']} {mail['body']}".strip()

        if not text:
            continue

        pred = predict_email(text)

        results.append({
            "email": mail["subject"],
            "category": pred["category"],
            "urgency": pred["urgency"]
        })

    return results
