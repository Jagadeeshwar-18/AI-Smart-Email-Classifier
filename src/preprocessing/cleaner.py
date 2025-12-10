import re
from bs4 import BeautifulSoup

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.replace("\n", " ")
    return text.lower()

def remove_signature(text):
    signature_keywords = ["regards", "thanks", "sincerely", "yours truly", "best wishes"]
    for kw in signature_keywords:
        text = text.split(kw)[0]
    return text

def clean_specials(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_email(text):
    text = basic_clean(text)
    text = remove_signature(text)
    text = clean_specials(text)
    return text
