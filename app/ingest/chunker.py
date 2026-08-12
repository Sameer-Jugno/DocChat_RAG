"""Fixed-size overlapping chunks with page/section metadata."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.types import PageText


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    source_name: str
    unit: str = "page"


def chunk_pages(
    pages: list[PageText],
    *,
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Character-based chunking across page boundaries.

    Each chunk records the page range it covers for citations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    # Flatten pages into a stream of (char, page_number)
    stream: list[tuple[str, int]] = []
    for page in pages:
        if stream and not stream[-1][0].isspace():
            stream.append(("\n\n", page.page_number))
        for ch in page.text:
            stream.append((ch, page.page_number))

    if not stream:
        return []

    default_unit = pages[0].unit if pages else "page"
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(stream)

    while start < n:
        end = min(start + chunk_size, n)
        # Prefer breaking on whitespace near the end
        if end < n:
            window = stream[start:end]
            break_at = None
            for i in range(len(window) - 1, max(len(window) // 2, 0), -1):
                if window[i][0].isspace():
                    break_at = start + i + 1
                    break
            if break_at is not None:
                end = break_at

        piece = stream[start:end]
        text = "".join(ch for ch, _ in piece).strip()
        if text:
            page_start = piece[0][1]
            page_end = piece[-1][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{source_name}::{idx}",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    source_name=source_name,
                    unit=default_unit,
                )
            )
            idx += 1

        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks
