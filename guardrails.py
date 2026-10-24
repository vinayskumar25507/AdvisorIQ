# guardrails.py
import re

def redact_pii(text: str | list) -> str:
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])
    if not isinstance(text, str):
        text = str(text)
        
    text = re.sub(r"\b\d{12,19}\b", "[ACCOUNT]", text)             # card / account numbers
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text) # emails
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)                 # phone numbers
    return text

BANNED = ["guaranteed return", "guaranteed profit", "can't lose", "risk-free"]
DISCLAIMER = "\n\n_Disclaimer: This is educational information, not financial advice._"

def make_safe(answer: str | list, sources: str | list = "") -> str:
    if isinstance(answer, list):
        answer = " ".join([str(a) for a in answer])
    if not isinstance(answer, str):
        answer = str(answer)

    for phrase in BANNED:                                         # block over-promising
        answer = re.sub(phrase, "may vary", answer, flags=re.I)
        
    if not sources:                                               # grounding check
        answer += "\n\n_Note: limited source support for this answer._"
        
    return redact_pii(answer + DISCLAIMER)