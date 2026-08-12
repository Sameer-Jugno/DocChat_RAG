"""Minimal FastAPI health + ingest/chat API (optional; Chainlit is the main UX)."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest.loader import is_supported, supported_list
from app.ingest.pipeline import ingest_file
from app.rag.chat import answer

app = FastAPI(title="Doc Chat RAG", version="0.2.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceOut(BaseModel):
    page_start: int
    page_end: int
    source_name: str
    score: float
    unit: str = "page"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "supported": sorted(supported_list().split(", "))}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    if not file.filename or not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file. Supported: {supported_list()}",
        )

    dest = settings.uploads_dir / Path(file.filename).name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        result = ingest_file(dest, source_name=dest.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    return {
        "source_name": result.source_name,
        "pages": result.pages,
        "chunks": result.chunks,
        "file_type": result.file_type,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    try:
        result = answer(body.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                page_start=s.page_start,
                page_end=s.page_end,
                source_name=s.source_name,
                score=s.score,
                unit=s.unit,
            )
            for s in result.sources
        ],
    )
