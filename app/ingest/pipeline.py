"""End-to-end ingest: file → chunks (+ optional CLIP images) → Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.ingest.chunker import chunk_pages
from app.ingest.image_index import index_pdf_images
from app.ingest.loader import extract_document, is_supported, supported_list
from app.rag import store
from app.rag import image_store


@dataclass(frozen=True)
class IngestResult:
    source_name: str
    pages: int
    chunks: int
    file_type: str
    images: int = 0


def ingest_file(file_path: Path, source_name: str | None = None) -> IngestResult:
    """
    Replace the current index with this document (one file at a time).

    Supports: .pdf, .txt, .md, .csv, .docx
    PDFs also get CLIP image indexing for figure/page retrieval.
    """
    settings = get_settings()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not is_supported(path):
        raise ValueError(
            f"Unsupported file type `{path.suffix}`. Supported: {supported_list()}"
        )

    max_mb = settings.upload_max_mb
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"File is {size_mb:.1f} MB; max allowed is {max_mb} MB")

    name = source_name or path.name
    pages = extract_document(path)
    if not pages:
        raise ValueError(
            "No extractable text found (native or OCR). "
            "The PDF may be empty or images too low-quality to read."
        )

    chunks = chunk_pages(
        pages,
        source_name=name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("Document produced no chunks after extraction.")

    store.reset_collection()
    written = store.upsert_chunks(chunks)

    images_written = 0
    if path.suffix.lower() == ".pdf":
        page_count = max((p.page_number for p in pages), default=0)
        try:
            import pymupdf

            doc = pymupdf.open(str(path))
            try:
                page_count = max(page_count, doc.page_count)
            finally:
                doc.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            images_written = index_pdf_images(path, name, page_count)
        except Exception as exc:  # noqa: BLE001
            # Text RAG should still work even if CLIP figure indexing fails
            import logging

            logging.getLogger("pdf_chat.pipeline").exception(
                "Image index failed (continuing with text only): %s", exc
            )
            images_written = 0
    else:
        image_store.reset_image_collection()

    return IngestResult(
        source_name=name,
        pages=len(pages),
        chunks=written,
        file_type=path.suffix.lower(),
        images=images_written,
    )


def ingest_pdf(pdf_path: Path, source_name: str | None = None) -> IngestResult:
    """Backward-compatible alias."""
    return ingest_file(pdf_path, source_name=source_name)
