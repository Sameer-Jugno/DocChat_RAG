"""Chainlit UI: upload document → chat (sources as text only, no side panel)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import chainlit as cl

from app.config import ROOT_DIR, get_settings
from app.ingest.loader import ACCEPT_MAP, is_supported, supported_list
from app.ingest.pipeline import ingest_file
from app.rag.chat import stream_answer
from app.rag import store

logger = logging.getLogger("pdf_chat.ui")

(ROOT_DIR / ".files").mkdir(parents=True, exist_ok=True)


def _cite_label(page_start: int, page_end: int, unit: str = "page") -> str:
    prefix = {"page": "p", "section": "sec", "rows": "rows"}.get(unit, "p")
    if page_start == page_end:
        return f"{prefix}.{page_start}"
    return f"{prefix}.{page_start}–{page_end}"


async def _save_upload(upload) -> Path:
    settings = get_settings()
    name = Path(getattr(upload, "name", None) or "document.bin").name
    if not is_supported(name):
        raise ValueError(f"Unsupported file. Supported: {supported_list()}")

    dest = settings.uploads_dir / name
    if getattr(upload, "path", None):
        await asyncio.to_thread(shutil.copy, upload.path, dest)
    else:
        await asyncio.to_thread(dest.write_bytes, upload.content)
    return dest


async def _index_file(dest: Path) -> bool:
    msg = cl.Message(content=f"Indexing `{dest.name}`…")
    await msg.send()

    try:
        result = await asyncio.to_thread(ingest_file, dest, dest.name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest failed")
        cl.user_session.set("ready", False)
        msg.content = f"Indexing failed: {exc}"
        await msg.update()
        return False

    # Fresh document → fresh conversation history (Chainlit built-in)
    cl.chat_context.clear()

    cl.user_session.set("ready", True)
    cl.user_session.set("source_name", result.source_name)
    cl.user_session.set("source_path", str(dest))
    cl.user_session.set("file_type", result.file_type)

    unit_word = {
        ".pdf": "pages",
        ".csv": "row-groups",
    }.get(result.file_type, "sections")
    msg.content = (
        f"`{result.source_name}` ready "
        f"({result.pages} {unit_word}, {result.chunks} text chunks). "
        "You can chat now."
    )
    await msg.update()
    return True


async def _wait_for_upload() -> None:
    settings = get_settings()
    max_mb = settings.upload_max_mb

    while True:
        files = await cl.AskFileMessage(
            content=(
                "Chat is locked until you upload a document.\n\n"
                f"Supported: `{supported_list()}`\n"
                "Select a file, then click **Open**."
            ),
            accept=ACCEPT_MAP,
            max_size_mb=max_mb,
            max_files=1,
            timeout=600,
        ).send()

        if not files:
            await cl.Message(
                content="No file received. Upload a document to unlock chat."
            ).send()
            continue

        try:
            dest = await _save_upload(files[0])
        except Exception as exc:  # noqa: BLE001
            await cl.Message(content=f"Upload error: {exc}").send()
            continue

        if await _index_file(dest):
            return


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    cl.user_session.set("ready", False)

    await cl.Message(
        content=(
            "# Doc Chat\n"
            f"Upload a document ({supported_list()}) to unlock the chat bar."
        )
    ).send()

    problems: list[str] = []
    if not settings.groq_api_key:
        problems.append("`GROQ_API_KEY` is missing in `.env`")
    try:
        store.get_client().get_collections()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Qdrant unreachable ({exc})")

    if problems:
        await cl.Message(
            content="Setup needed:\n" + "\n".join(f"- {p}" for p in problems)
        ).send()
        return

    await _wait_for_upload()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()

    if text.lower() in {"/upload", "/replace"}:
        cl.user_session.set("ready", False)
        await cl.Message(content="Upload a new document to continue chatting.").send()
        await _wait_for_upload()
        return

    if not cl.user_session.get("ready"):
        await cl.Message(
            content="Chat is locked. Upload a document first (type `/upload`)."
        ).send()
        await _wait_for_upload()
        return

    if not text:
        await cl.Message(content="Type a question about the document.").send()
        return

    reply = cl.Message(content="")
    await reply.send()

    try:
        # Chainlit built-in conversation history (OpenAI-compatible message list)
        history = cl.chat_context.to_openai()
        sources, token_iter = await asyncio.to_thread(
            stream_answer, text, None, history
        )
        loop = asyncio.get_running_loop()
        it = iter(token_iter)

        def _next():
            try:
                return next(it), False
            except StopIteration:
                return None, True

        while True:
            item, done = await loop.run_in_executor(None, _next)
            if done:
                break
            await reply.stream_token(item)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        reply.content = f"Chat failed: {exc}"
        await reply.update()
        return

    # Sources only as trailing text — no side-panel elements
    if sources:
        lines = ["\n\n---\n**Sources**"]
        seen: set[tuple[str, int, int]] = set()
        for s in sources:
            key = (s.source_name, s.page_start, s.page_end)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- `{s.source_name}` ({_cite_label(s.page_start, s.page_end, s.unit)}) "
                f"· score {s.score:.3f}"
            )
        await reply.stream_token("\n".join(lines))

    await reply.update()
