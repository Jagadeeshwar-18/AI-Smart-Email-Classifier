import re

HIGH_URGENCY = [
    "urgent", "asap", "immediately", "not working", "down",
    "failed", "error", "unable", "critical", "blocked"
]

MEDIUM_URGENCY = [
    "soon", "delay", "issue", "problem", "help", "support",
    "follow up", "request", "pending"
]

def rule_based_urgency(text: str) -> str:
    if not isinstance(text, str):
        return "low"

    t = text.lower()

    for kw in HIGH_URGENCY:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "high"

    for kw in MEDIUM_URGENCY:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "medium"

    return "low"
