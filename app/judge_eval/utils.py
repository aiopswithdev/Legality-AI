import hashlib
import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def hash_clause(clause_text: str, label: str) -> str:
    base = normalize_text(clause_text) + "::" + label.lower()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def clause_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
