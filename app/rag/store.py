"""Qdrant vector store helpers — one collection, one PDF at a time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.ingest.chunker import Chunk
from app.rag.embedder import embed_texts, embedding_dim


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_start: int
    page_end: int
    source_name: str
    score: float
    unit: str = "page"


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def reset_collection() -> None:
    """Drop and recreate the collection (one-PDF replace semantics)."""
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=embedding_dim(),
            distance=qm.Distance.COSINE,
        ),
    )


def upsert_chunks(chunks: list[Chunk]) -> int:
    """Embed and upsert chunks. Returns number of points written."""
    if not chunks:
        return 0

    settings = get_settings()
    client = get_client()
    vectors = embed_texts([c.text for c in chunks])

    points = [
        qm.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "source_name": chunk.source_name,
                "unit": chunk.unit,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(query_vector: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    client = get_client()

    if not client.collection_exists(settings.qdrant_collection):
        return []

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    out: list[RetrievedChunk] = []
    for point in results.points:
        payload = point.payload or {}
        out.append(
            RetrievedChunk(
                text=str(payload.get("text", "")),
                page_start=int(payload.get("page_start", 0)),
                page_end=int(payload.get("page_end", 0)),
                source_name=str(payload.get("source_name", "")),
                score=float(point.score or 0.0),
                unit=str(payload.get("unit", "page")),
            )
        )
    return out
