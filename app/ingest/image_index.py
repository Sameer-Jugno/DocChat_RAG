"""Index only PDF pages that contain figures (generic for any PDF)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import ROOT_DIR, get_settings
from app.ingest.pdf_images import (
    extract_large_embedded_images,
    find_figure_pages,
    render_pdf_pages,
)
from app.rag.image_store import ImageRecord, reset_image_collection, upsert_images

logger = logging.getLogger("pdf_chat.image_index")

PREVIEWS = ROOT_DIR / "data" / "previews"


def index_pdf_images(pdf_path: Path, source_name: str, page_count: int) -> int:
    """
    CLIP-index only pages that contain figures (bitmaps or vector drawings).

    Does NOT index every text page — avoids retrieving unrelated page screenshots.
    Works for any PDF, not a specific paper.
    """
    settings = get_settings()
    reset_image_collection()

    if not settings.image_index_enabled:
        return 0
    if pdf_path.suffix.lower() != ".pdf":
        return 0

    out_dir = PREVIEWS / Path(source_name).stem / "clip"
    max_pages = settings.image_max_pages

    figure_pages = find_figure_pages(pdf_path, max_pages=max_pages)
    if not figure_pages:
        logger.info("No figure pages detected in %s — image index empty", source_name)
        return 0

    # Render those figure-containing pages once for CLIP + UI
    rendered = render_pdf_pages(
        pdf_path,
        figure_pages,
        out_dir=out_dir / "figure_pages",
        dpi=settings.image_index_dpi,
        max_pages=max_pages,
    )
    embedded = extract_large_embedded_images(
        pdf_path,
        figure_pages,
        out_dir=out_dir / "figures",
        max_images=settings.image_max_figures,
    )

    records: list[ImageRecord] = [
        ImageRecord(path=p, page=n, source_name=source_name, kind="figure_page")
        for n, p in rendered
    ]
    records.extend(
        ImageRecord(path=p, page=n, source_name=source_name, kind="figure")
        for n, p in embedded
    )

    if not records:
        return 0

    written = upsert_images(records)
    logger.info(
        "Indexed %s image vectors from figure pages %s (%s)",
        written,
        figure_pages,
        source_name,
    )
    return written
