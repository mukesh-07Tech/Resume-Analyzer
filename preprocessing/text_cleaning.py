import re


def clean_text(text: str) -> str:
    """Normalize text: lowercase, replace non-letters with spaces, collapse spaces."""
    if not isinstance(text, str):
        text = str(text or "")
    text = text.lower()
    # Replace any sequence of non-letter characters with a single space
    text = re.sub(r"[^a-z]+", " ", text)
    # Collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text
