# PDF Chat — Decisions

| Status | Decision |
|--------|----------|
| **Decided** | Product: upload **one PDF** → ingest → chat with page citations |
| **Decided** | Stack: Chainlit + FastAPI + pypdf + fastembed (`all-MiniLM-L6-v2`) + local Qdrant + Groq |
| **Decided** | One PDF at a time: new upload **replaces** the vector collection |
| **Decided** | No OCR in MVP (scanned PDFs unsupported) |
| **Decided** | Lives in `pdf-chat/` — separate from CodeSage repo-RAG |
| **Decided** | Formats now: `.pdf` (text + OCR), `.txt`, `.md`, `.csv`, `.docx`. Not yet: `.doc`, Excel, image-only uploads. |
| **Decided** | PDF visuals: CLIP (`clip-ViT-B-32`) embeds page/figure images; query text retrieves images at runtime alongside text RAG. |

## Deliberately not built

- Multi-document libraries
- Auth / multi-user
- Reranker / Pinecone
- OCR
