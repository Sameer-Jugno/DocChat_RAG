"""Shared document units for chunking (page / section / row-group)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """One extractable unit of a document (PDF page, text block, CSV row-group, …)."""

    page_number: int  # 1-based unit index (used for citations)
    text: str
    unit: str = "page"  # page | section | rows
