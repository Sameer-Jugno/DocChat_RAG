"""Qdrant store for CLIP-embedded figure pages (any PDF)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from qdrant_client.http import models as qm

from app.config import get_settings
from app.rag.image_embedder import embed_clip_query, embed_images, image_embedding_dim
from app.rag.store import get_client


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    page: int
    source_name: str
    kind: str  # figure_page | figure


@dataclass(frozen=True)
class RetrievedImage:
    path: str
    page: int
    source_name: str
    kind: str
    score: float


def reset_image_collection() -> None:
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_image_collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=image_embedding_dim(),
            distance=qm.Distance.COSINE,
        ),
    )


def upsert_images(records: list[ImageRecord]) -> int:
    if not records:
        return 0
    settings = get_settings()
    client = get_client()
    vectors = embed_images([r.path for r in records])
    points = [
        qm.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "path": str(record.path),
                "page": record.page,
                "source_name": record.source_name,
                "kind": record.kind,
            },
        )
        for record, vector in zip(records, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_image_collection, points=points)
    return len(points)


def search_images(query: str, top_k: int | None = None) -> list[RetrievedImage]:
    """Return top matching figure pages — at most one visual per page."""
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_image_collection
    if not client.collection_exists(name):
        return []

    k = top_k if top_k is not None else settings.image_top_k
    # Fetch extra then dedupe by page
    vector = embed_clip_query(query)
    results = client.query_points(
        collection_name=name,
        query=vector,
        limit=max(k * 4, 8),
        with_payload=True,
    )

    kind_rank = {"figure": 2, "figure_page": 1, "page": 0}
    best_by_page: dict[int, RetrievedImage] = {}

    for point in results.points:
        payload = point.payload or {}
        path = str(payload.get("path", ""))
        if not path or not Path(path).is_file():
            continue
        score = float(point.score or 0.0)
        if score < settings.image_min_score:
            continue
        page = int(payload.get("page", 0))
        kind = str(payload.get("kind", "figure_page"))
        cand = RetrievedImage(
            path=path,
            page=page,
            source_name=str(payload.get("source_name", "")),
            kind=kind,
            score=score,
        )
        prev = best_by_page.get(page)
        if prev is None:
            best_by_page[page] = cand
            continue
        # Prefer higher score; tie-break toward cropped figure over full page
        if cand.score > prev.score + 0.01:
            best_by_page[page] = cand
        elif abs(cand.score - prev.score) <= 0.01 and kind_rank.get(
            cand.kind, 0
        ) > kind_rank.get(prev.kind, 0):
            best_by_page[page] = cand

    ranked = sorted(best_by_page.values(), key=lambda x: x.score, reverse=True)
    return ranked[:k]
