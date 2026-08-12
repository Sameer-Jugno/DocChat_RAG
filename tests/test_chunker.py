from app.ingest.chunker import chunk_pages
from app.ingest.pdf_loader import PageText


def test_chunk_pages_basic():
    pages = [
        PageText(1, "A" * 400),
        PageText(2, "B" * 400),
    ]
    chunks = chunk_pages(pages, source_name="demo.pdf", chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    assert chunks[0].source_name == "demo.pdf"
    assert chunks[0].page_start == 1
    assert all(c.text for c in chunks)


def test_chunk_empty():
    assert chunk_pages([], source_name="x.pdf") == []
