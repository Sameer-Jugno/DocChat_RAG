"""Normalize whitespace in extracted text."""


def normalize_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    joined = "\n".join(ln for ln in lines if ln)
    return "\n".join(p.strip() for p in joined.split("\n\n") if p.strip())
