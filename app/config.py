"""Application settings (pydantic-settings)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "pdf_chat"

    # MiniLM is typically faster to embed than bge-small for ingest
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_threads: int = 0  # 0 = use all CPU cores
    embedding_parallel: int = 0  # 0 = fastembed default; >0 = multi-proc

    chat_history_turns: int = 6  # prior user/assistant pairs sent to the LLM

    data_dir: Path = ROOT_DIR / "data"
    uploads_dir: Path = ROOT_DIR / "data" / "uploads"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # File limits (alias max_pdf_mb kept for older .env)
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    max_file_mb: int = 25
    max_pdf_mb: int = 25  # deprecated alias — prefer max_file_mb

    # OCR for scanned / image-heavy PDFs (RapidOCR, no system tesseract)
    ocr_enabled: bool = True
    ocr_min_chars: int = 40  # native text below this → page OCR
    ocr_dpi: int = 200
    ocr_embedded_images: bool = True

    @property
    def upload_max_mb(self) -> int:
        return self.max_file_mb or self.max_pdf_mb


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
