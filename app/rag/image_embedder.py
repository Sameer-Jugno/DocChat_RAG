"""CLIP image/text embeddings via fastembed (shared vector space)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastembed import ImageEmbedding, TextEmbedding

from app.config import get_settings


@lru_cache
def get_image_embedder() -> ImageEmbedding:
    settings = get_settings()
    return ImageEmbedding(model_name=settings.image_embedding_model)


@lru_cache
def get_clip_text_embedder() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.clip_text_embedding_model)


@lru_cache
def image_embedding_dim() -> int:
    settings = get_settings()
    for meta in ImageEmbedding.list_supported_models():
        if meta.get("model") == settings.image_embedding_model:
            return int(meta["dim"])
    # Fallback: embed a tiny blank via path not available — use known CLIP dim
    return 512


def embed_images(paths: list[Path]) -> list[list[float]]:
    if not paths:
        return []
    model = get_image_embedder()
    # fastembed accepts paths or PIL; paths as str work
    return [vec.tolist() for vec in model.embed([str(p) for p in paths])]


def embed_clip_query(text: str) -> list[float]:
    model = get_clip_text_embedder()
    return next(model.embed([text])).tolist()
