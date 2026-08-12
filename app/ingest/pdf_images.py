"""Detect and export PDF pages that contain figures (bitmaps or vector drawings)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("pdf_chat.pdf_images")


def render_pdf_pages(
    pdf_path: Path,
    page_numbers: list[int],
    *,
    out_dir: Path,
    dpi: int = 140,
    max_pages: int = 4,
) -> list[tuple[int, Path]]:
    """Render 1-based page numbers to PNG files. Returns (page_number, png_path)."""
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted: list[int] = []
    seen: set[int] = set()
    for n in page_numbers:
        if n < 1 or n in seen:
            continue
        seen.add(n)
        wanted.append(n)
        if len(wanted) >= max_pages:
            break

    if not wanted:
        return []

    zoom = max(dpi, 72) / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    results: list[tuple[int, Path]] = []

    doc = pymupdf.open(str(pdf_path))
    try:
        for page_no in wanted:
            if page_no > doc.page_count:
                continue
            page = doc.load_page(page_no - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            dest = out_dir / f"page_{page_no}.png"
            pix.save(str(dest))
            results.append((page_no, dest))
    finally:
        doc.close()

    return results


def find_figure_pages(
    pdf_path: Path,
    *,
    max_pages: int | None = None,
    min_image_pixels: int = 40_000,
    min_drawing_paths: int = 12,
) -> list[int]:
    """
    Return 1-based page numbers that likely contain figures.

    Detects:
    - embedded bitmaps above a size threshold
    - vector drawings (common in academic PDFs like architecture diagrams)
    """
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []

    doc = pymupdf.open(str(pdf_path))
    figure_pages: list[int] = []
    try:
        limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(limit):
            page = doc.load_page(i)
            page_no = i + 1

            # Bitmap figures
            has_bitmap = False
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    info = doc.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                w = int(info.get("width") or 0)
                h = int(info.get("height") or 0)
                if w * h >= min_image_pixels:
                    has_bitmap = True
                    break

            # Vector figures (Attention paper style)
            has_vector = False
            try:
                drawings = page.get_drawings()
                if len(drawings) >= min_drawing_paths:
                    has_vector = True
                else:
                    # Fewer paths but covering a meaningful area
                    area = 0.0
                    page_area = float(page.rect.width * page.rect.height) or 1.0
                    for d in drawings:
                        r = d.get("rect")
                        if r is not None:
                            area += abs(float(r.width) * float(r.height))
                    if area / page_area >= 0.08 and len(drawings) >= 3:
                        has_vector = True
            except Exception:  # noqa: BLE001
                has_vector = False

            if has_bitmap or has_vector:
                figure_pages.append(page_no)
    finally:
        doc.close()

    logger.info(
        "Figure pages in %s: %s",
        pdf_path.name,
        figure_pages[:30] + (["…"] if len(figure_pages) > 30 else []),
    )
    return figure_pages


def extract_large_embedded_images(
    pdf_path: Path,
    page_numbers: list[int],
    *,
    out_dir: Path,
    min_pixels: int = 80_000,
    max_images: int = 4,
) -> list[tuple[int, Path]]:
    """Save large embedded bitmaps from selected pages."""
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    saved: list[tuple[int, Path]] = []
    try:
        for page_no in page_numbers:
            if page_no < 1 or page_no > doc.page_count:
                continue
            page = doc.load_page(page_no - 1)
            for idx, img in enumerate(page.get_images(full=True)):
                if len(saved) >= max_images:
                    return saved
                xref = img[0]
                try:
                    extracted = doc.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                w = int(extracted.get("width") or 0)
                h = int(extracted.get("height") or 0)
                if w * h < min_pixels:
                    continue
                image_bytes = extracted.get("image")
                ext = extracted.get("ext") or "png"
                if not image_bytes:
                    continue
                dest = out_dir / f"p{page_no}_img{idx}.{ext}"
                dest.write_bytes(image_bytes)
                saved.append((page_no, dest))
    finally:
        doc.close()

    return saved
