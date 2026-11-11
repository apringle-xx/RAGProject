---
name: test-app
description: Install, run, and verify Agentic RAG before every PR, including recorded browser tests for UI changes.
---

# Test Agentic RAG

Run from the repository root. Use this procedure before every PR.

## Install

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Python 3.10–3.12 is supported. No database service is needed: FAISS indexes are stored locally.
No git hooks exist; CI enforces lint, formatting and tests.

## Required checks (every PR)

```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

For a setup-only slice with no tests yet, verify dependency imports and Ruff instead; add core tests with the backend slice.

## Evaluation (when backend exists)

```sh
.venv/bin/python -m rag.evaluate --offline
# Paid, real OpenAI calls; OPENAI_API_KEY must be set:
.venv/bin/python -m rag.evaluate --output evaluation.json
```

The offline mode is a wiring smoke test, not evidence of OpenAI answer quality.
The live command runs the same fixed corpus/questions against both systems, prints a comparison table and records individual results.

## Run and record UI (mandatory before UI PRs)

The testing agent owns dependency setup, starting Streamlit, browser access and recording.
Read this skill and `AGENTS.md` first, then run:

```sh
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open port 8501 in the browser. Default mode is OpenAI; if no key is available, explicitly select **Offline demo** and report that the OpenAI provider is untested. Do not set fake credentials or silently mock the live provider.

Record the following through the rendered UI, adding test/assertion annotations:
1. Starting state: fresh browser session, empty library, chat greeting. Verify the API-key notice if live mode lacks a key.
2. Select Offline demo if needed. Create a UTF-8 text fixture containing `Project Atlas launches on 15 March 2027. The project owner is Maya Chen.` Upload it and click **Index documents**. Confirm the filename and chunk count.
3. Ask `When does Project Atlas launch and who owns it?`. Verify the answer includes 15 March 2027 and Maya Chen, a clickable source/citation, source chunk text, latency, and retrieval trace.
4. Ask a follow-up `Who owns it?`. Verify conversation context resolves Atlas. Ask `Hello` and confirm no retrieval.
5. Upload a second TXT and a generated PDF and DOCX with distinct facts. Index, ask for each fact, inspect source names/page or section metadata. Reindex the same file and verify no duplicate chunks.
6. Upload an empty TXT and verify a friendly validation error with the existing library still usable. Ask an unsupported factual question and confirm uncertainty rather than fabricated evidence.
7. Clear the chat, then clear the library; verify messages and document count reset as intended. Refresh and ensure the interface remains usable. Verify narrow viewport layout.
8. If a valid OPENAI_API_KEY is available, repeat upload, cited question and follow-up in **OpenAI** mode; distinguish this result from demo validation.

Stop and report if the required starting state cannot be reached. Record any gap honestly. Deliver one primary video and the final verdict. Include an animated WebP version of the primary flow for embedding in the UI PR (MP4 cannot be embedded by the PR tool).
