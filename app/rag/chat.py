"""Retrieve + generate answers with Groq (text RAG + chat history)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from groq import Groq

from app.config import get_settings
from app.rag.embedder import embed_query
from app.rag.store import RetrievedChunk, search

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a single uploaded document
(PDF, text, Markdown, CSV, or Word .docx).

Rules:
- Use ONLY the provided context excerpts, plus prior chat turns when the user refers to them.
- If the context is insufficient, say you don't know based on the document.
- Cite sources using the provided labels like [p.3], [sec.2], or [rows.1] when claiming facts.
- Be concise and accurate. Do not invent content not present in the context.
- Resolve follow-ups using conversation history (e.g. "what about that?", "explain more").
"""


def _cite_label(chunk: RetrievedChunk) -> str:
    unit = chunk.unit or "page"
    prefix = {"page": "p", "section": "sec", "rows": "rows"}.get(unit, "p")
    if chunk.page_start == chunk.page_end:
        return f"{prefix}.{chunk.page_start}"
    return f"{prefix}.{chunk.page_start}–{chunk.page_end}"


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[RetrievedChunk]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c.source_name}, {_cite_label(c)})\n{c.text}")
    return "\n\n".join(parts)


def _trim_history(
    history: Sequence[dict[str, Any]] | None,
    *,
    current_question: str,
    max_turns: int,
) -> list[dict[str, str]]:
    """
    Keep recent user/assistant turns from Chainlit chat_context.to_openai().

    Drops a trailing user message that matches the current question (already
    present in chat_context when on_message runs).
    """
    if not history:
        return []

    cleaned: list[dict[str, str]] = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        # Skip upload / system-ish UI prompts
        if role == "assistant" and content.startswith("# Doc Chat"):
            continue
        if role == "assistant" and "Chat is locked" in content:
            continue
        if role == "assistant" and content.startswith("Indexing "):
            continue
        if role == "assistant" and "`" in content and " ready (" in content:
            continue
        cleaned.append({"role": role, "content": content})

    if (
        cleaned
        and cleaned[-1]["role"] == "user"
        and cleaned[-1]["content"].strip() == current_question.strip()
    ):
        cleaned = cleaned[:-1]

    # Keep last N user/assistant pairs
    max_msgs = max(0, max_turns) * 2
    if max_msgs and len(cleaned) > max_msgs:
        cleaned = cleaned[-max_msgs:]
    return cleaned


def build_llm_messages(
    question: str,
    context: str,
    history: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    settings = get_settings()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        _trim_history(
            history,
            current_question=question,
            max_turns=settings.chat_history_turns,
        )
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Context from the document:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using the document context above. "
                "Use prior conversation turns only to resolve follow-ups."
            ),
        }
    )
    return messages


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    vector = embed_query(question)
    return search(vector, top_k=top_k)


def _client() -> Groq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your key."
        )
    return Groq(api_key=settings.groq_api_key)


def answer(
    question: str,
    top_k: int | None = None,
    history: Sequence[dict[str, Any]] | None = None,
) -> ChatResult:
    settings = get_settings()
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return ChatResult(
            answer="No document is indexed yet. Please upload a file first.",
            sources=[],
        )

    context = _format_context(chunks)
    messages = build_llm_messages(question, context, history=history)

    completion = _client().chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
    )
    text = completion.choices[0].message.content or ""
    return ChatResult(answer=text, sources=chunks)


def stream_answer(
    question: str,
    top_k: int | None = None,
    history: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[RetrievedChunk], Iterator[str]]:
    """Return text sources and a streaming token iterator."""
    settings = get_settings()
    chunks = retrieve(question, top_k=top_k)
    if not chunks:

        def empty() -> Iterator[str]:
            yield "No document is indexed yet. Please upload a file first."

        return [], empty()

    context = _format_context(chunks)
    messages = build_llm_messages(question, context, history=history)

    stream = _client().chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    def tokens() -> Iterator[str]:
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    return chunks, tokens()
