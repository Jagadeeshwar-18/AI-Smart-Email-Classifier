import joblib
from urgency_rules import rule_based_urgency

# Load ML model once
ml_model = joblib.load("models/urgency/urgency_lr.pkl")

def predict_urgency(text: str) -> str:
    """
    Hybrid urgency prediction:
    - Rule-based HIGH overrides everything
    - Otherwise, use ML prediction
    """
    rule_pred = rule_based_urgency(text)

    if rule_pred == "high":
        return "high"

    ml_pred = ml_model.predict([text])[0]
    return ml_pred


# Quick test
if __name__ == "__main__":
    samples = [
        "System is down, need fix ASAP",
        "Please provide an update on my request",
        "Just sharing feedback about your service"
    ]

    for s in samples:
        print(f"{s} --> {predict_urgency(s)}")
