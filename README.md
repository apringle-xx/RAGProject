# Agentic RAG

A local document library and retrieval agent using FAISS, OpenAI, and Streamlit.

## Install

Use Python 3.10–3.12:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in your environment or `.env`. The default models are
`text-embedding-3-small` and `gpt-4o-mini`; override them with `RAG_EMBEDDING_MODEL`
and `RAG_CHAT_MODEL`. Never commit your key. Real OpenAI use sends extracted document
text, queries and bounded chat history to OpenAI and incurs API charges. Original files
are not saved. Chunk text and vectors remain in the ignored `.rag-data/` directory.

## Core pipeline

`rag.documents` accepts UTF-8 TXT, PDF (with text, no OCR), and DOCX (including tables).
It creates overlapping 1,800-character chunks with source, page/section and offset metadata.
Uploads are limited to 20 MB, 500 PDF pages, 2 million extracted characters and 2,000
chunks per document. Libraries are capped at 10,000 chunks.

`Library.ingest()` embeds in batches and transactionally stores float32 vectors plus
metadata in SQLite. FAISS builds a cosine index from persisted vectors on each search,
so there are no mismatched index/metadata files or pickles. This favors correctness for
small local libraries over large-scale throughput. Identical uploads are skipped;
changed content with the same filename replaces the previous version. Embedding model
changes require a different library directory. Clearing a library resets its dimension.

`RetrievalAgent` lets the model choose whether to retrieve and rewrite queries, then
performs up to three bounded tool searches. It resolves follow-ups using the last ten
chat messages, answers with `[S1]` source-chunk citations, and rejects unknown citation
labels. Documents are treated as untrusted data. Citation validation checks provenance,
not semantic entailment; the model can still make mistakes.

`NaiveRAG` uses the same embedding/chat models and retrieval count per search, but always
retrieves once with the raw question and generates once without tools or query rewriting.

## Evaluation

```sh
# Real model calls, fixed corpus and eight questions:
.venv/bin/python -m rag.evaluate --output evaluation.json
# No API calls; explicit extractive demo to verify pipeline wiring:
.venv/bin/python -m rag.evaluate --offline
```

The harness prints both systems' keyword/alias answer accuracy, expected-source recall,
citation validity, mean end-to-end question latency, searches, model calls and errors.
The JSON includes every answer, citations, trace and corpus/question hashes. Cases cover
direct lookup, multiple documents, multi-hop questions, abstention and greetings. The
retrieval `--k` defaults to 2 for both systems. A temporary evaluation library isolates
the fixed corpus from uploaded documents. Ingestion time is excluded from answer latency.

These small, deterministic keyword metrics are a reproducible starting point, not a
semantic faithfulness judge or proof that the agent outperforms the baseline. Offline
demo scores are only a wiring check. No OpenAI quality results are claimed without a
live evaluation. `--output` may contain source text; keep it private.

## Development

```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

Core tests run without credentials or network access. Read `AGENTS.md` and follow
`.agents/skills/test-app/SKILL.md` before every PR. UI changes require a recorded
browser test before opening a PR.
