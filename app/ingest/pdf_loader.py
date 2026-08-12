"""Extract text from PDF pages (thin wrapper around loader)."""

from __future__ import annotations

from pathlib import Path

from app.ingest.loader import _extract_pdf
from app.ingest.types import PageText

# Re-export for existing imports/tests
__all__ = ["PageText", "extract_pages"]


def extract_pages(pdf_path: Path) -> list[PageText]:
    return _extract_pdf(Path(pdf_path))
