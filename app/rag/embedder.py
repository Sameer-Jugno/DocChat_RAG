"""Local embeddings via fastembed (optimized for ingest latency)."""

from __future__ import annotations

import os
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings


@lru_cache
def get_embedder() -> TextEmbedding:
    settings = get_settings()
    threads = settings.embedding_threads or max(1, (os.cpu_count() or 2))
    return TextEmbedding(
        model_name=settings.embedding_model,
        threads=threads,
    )


@lru_cache
def embedding_dim() -> int:
    """Vector size from model metadata — no probe embed on every ingest."""
    settings = get_settings()
    for meta in TextEmbedding.list_supported_models():
        if meta.get("model") == settings.embedding_model:
            return int(meta["dim"])
    # Fallback if model string is custom/unknown
    return len(embed_query("dimension probe"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns dense vectors as float lists."""
    if not texts:
        return []
    settings = get_settings()
    model = get_embedder()
    # Larger batches + multi-process ONNX when parallel > 1
    parallel = settings.embedding_parallel
    vectors = model.embed(
        texts,
        batch_size=settings.embedding_batch_size,
        parallel=parallel if parallel and parallel > 0 else None,
    )
    return [vec.tolist() for vec in vectors]


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    return next(model.embed([text], batch_size=1)).tolist()
