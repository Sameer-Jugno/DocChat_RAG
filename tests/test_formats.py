from pathlib import Path

from app.ingest.loader import extract_document, is_supported
from app.ingest.pipeline import ingest_file


def test_is_supported():
    assert is_supported("a.pdf")
    assert is_supported("a.TXT")
    assert is_supported("a.docx")
    assert is_supported("a.csv")
    assert not is_supported("a.doc")
    assert not is_supported("a.xlsx")


def test_extract_txt_md_csv(tmp_path: Path):
    txt = tmp_path / "note.txt"
    txt.write_text("Hello world.\n\nSecond paragraph about cats.")
    pages = extract_document(txt)
    assert pages and "Hello" in pages[0].text

    md = tmp_path / "note.md"
    md.write_text("# Title\n\nSome markdown body.")
    assert extract_document(md)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age\nAda,36\nBob,41\n")
    csv_pages = extract_document(csv_path)
    assert csv_pages
    assert "Ada" in csv_pages[0].text
    assert csv_pages[0].unit == "rows"


def test_ingest_txt(tmp_path: Path):
    path = tmp_path / "facts.txt"
    path.write_text(
        "Project Codename: Bluebird.\n"
        "The release date is March 2026.\n"
        "Owner: Platform team."
    )
    result = ingest_file(path)
    assert result.chunks >= 1
    assert result.file_type == ".txt"
