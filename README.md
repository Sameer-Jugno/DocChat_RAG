# DocChat RAG

Upload **one document** and chat with grounded answers and citations.

## Features

- Formats: PDF (text + OCR), TXT, MD, CSV, DOCX
- Local embeddings (`all-MiniLM-L6-v2`) + Qdrant text retrieval
- Groq chat
- Chainlit UI with conversation history (`cl.chat_context`)
- Optional FastAPI `/ingest` + `/chat`

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY
docker compose up -d
PYTHONPATH=. chainlit run app_ui.py --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

See `APPLICATION_UNDERSTANDING.txt` for full architecture.
See `data/samples/` for test files.
