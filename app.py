"""Streamlit frontend. Parsing, storage, retrieval and generation live in rag/."""

import hashlib
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError

from rag.agent import RetrievalAgent
from rag.providers import OfflineEmbeddings, OfflineModel, OpenAIEmbeddings, OpenAIModel
from rag.service import Library
from rag.store import VectorStore

load_dotenv()
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Agentic RAG · Document workspace", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/styles.css').read_text()}</style>", unsafe_allow_html=True)

# A random, validated workspace token preserves the local library across refreshes.
# This is a local single-user app, not an authentication mechanism.
workspace = st.query_params.get("workspace", "")
if not re.fullmatch(r"[a-f0-9]{32}", workspace):
    workspace = uuid.uuid4().hex
    st.query_params["workspace"] = workspace
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_version", 0)
st.session_state.setdefault("ingestion_results", [])
if st.session_state.get("active_workspace") != workspace:
    st.session_state.messages = []
    st.session_state.ingestion_results = []
    st.session_state.pop("bundle", None)
    st.session_state.active_workspace = workspace


def make_bundle(mode: str):
    embedder = OfflineEmbeddings() if mode == "Offline demo" else OpenAIEmbeddings()
    model = OfflineModel() if mode == "Offline demo" else OpenAIModel(client=embedder.client)
    model_directory = hashlib.sha256(embedder.name.encode()).hexdigest()[:16]
    directory = Path(os.getenv("RAG_DATA_DIR", ".rag-data")) / workspace / model_directory
    library = Library(VectorStore(directory, embedder.name), embedder)
    return library, RetrievalAgent(library, model)


def render_message(message: dict):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
            return
        citations = message.get("citations", [])
        labels = {c["label"] for c in citations}
        anchor = message["id"]
        linked_text = re.sub(
            r"\[(S\d+)\]",
            lambda match: (
                f"[{match[1]}](#source-{anchor}-{match[1]})" if match[1] in labels else match[0]
            ),
            message["content"],
        )
        st.markdown(linked_text)
        searches = len(message.get("trace", []))
        st.caption(
            f"{message['latency_ms']:.1f} ms · {searches} "
            f"{'search' if searches == 1 else 'searches'} · "
            f"{message['model_calls']} model calls"
        )
        for citation in citations:
            chunk = citation["chunk"]
            st.markdown(
                f'<div id="source-{anchor}-{citation["label"]}"></div>', unsafe_allow_html=True
            )
            with st.expander(f"{citation['label']} · {chunk['source']} · {chunk['location']}"):
                st.text(chunk["text"])
                st.caption(f"Chunk {chunk['id'][:12]} · character offset {chunk['start']}")
        if message.get("trace"):
            with st.expander("Retrieval trace"):
                for i, event in enumerate(message["trace"], 1):
                    if "error" in event:
                        st.caption(f"Search {i}: invalid tool call rejected")
                    else:
                        st.text(f"{i}. {event['query']}")
                        st.caption(f"{event['hits']} chunks · {event.get('latency_ms', 0):.1f} ms")


with st.sidebar:
    st.markdown("### Agentic RAG")
    st.caption("DOCUMENT WORKSPACE")
    mode = st.radio("Provider", ["OpenAI", "Offline demo"], horizontal=True)
    if st.session_state.get("active_mode") != mode:
        st.session_state.messages = []
        st.session_state.ingestion_results = []
        st.session_state.pop("bundle", None)
        st.session_state.active_mode = mode
    ready = mode == "Offline demo" or bool(os.getenv("OPENAI_API_KEY"))
    if not ready:
        st.info(
            "OpenAI key not configured. Set OPENAI_API_KEY, or choose Offline demo to try the app."
        )
    if mode == "Offline demo":
        st.warning("Offline demo uses word matching and extracts. It does not use an LLM.")
    else:
        st.caption(
            "OpenAI receives extracted text and questions. Original files stay on your machine."
        )
    bundle = None
    if ready:
        try:
            if "bundle" not in st.session_state:
                st.session_state.bundle = make_bundle(mode)
            bundle = st.session_state.bundle
        except (ValueError, OSError):
            st.error(
                "Could not open the local library. Check its directory permissions and settings."
            )
    library, agent = bundle if bundle else (None, None)
    st.divider()
    st.markdown("#### Add sources")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploads-{st.session_state.uploader_version}",
        help="PDF, DOCX or UTF-8 TXT · up to 20 MB each",
    )
    if st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
        disabled=not uploads or library is None,
    ):
        results = []
        with st.spinner("Reading and indexing documents…"):
            for upload in uploads:
                try:
                    result = library.ingest(upload.name, upload.getvalue())
                    results.append({"ok": True, **result})
                except ValueError as exc:
                    results.append({"ok": False, "source": upload.name, "error": str(exc)})
                except OpenAIError:
                    results.append(
                        {
                            "ok": False,
                            "source": upload.name,
                            "error": "OpenAI request failed. Check your key, network and quota.",
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "ok": False,
                            "source": upload.name,
                            "error": "Indexing failed. Previously indexed documents are unchanged.",
                        }
                    )
        st.session_state.ingestion_results = results
    for result in st.session_state.ingestion_results:
        if not result["ok"]:
            st.error(f"{result['source']}: {result['error']}")
        elif result["duplicate"]:
            st.info(f"{result['source']} · already indexed")
        else:
            st.success(
                f"{result['source']} · {result['chunks']} chunks · {result['latency_ms']:.1f} ms"
            )
    documents = library.store.documents() if library else []
    st.divider()
    st.markdown(f"#### Your library · {len(documents)}")
    if not documents:
        st.caption("No documents yet. Add your first source above.")
    for document in documents:
        st.text(document["source"])
        st.caption(f"{document['chunks']} indexed chunks")
    if st.button("Clear library", disabled=not documents, use_container_width=True):
        library.store.clear()
        st.session_state.messages = []
        st.session_state.ingestion_results = []
        st.session_state.uploader_version += 1
        st.rerun()
    st.caption(
        "Libraries are separate for each provider. Your workspace URL preserves them on refresh."
    )

st.markdown('<p class="eyebrow">RETRIEVE · REASON · REFERENCE</p>', unsafe_allow_html=True)
st.title("Ask your documents.")
st.markdown("Turn your files into a conversation. Every grounded answer leads back to its source.")
if mode == "Offline demo":
    st.info(
        "Offline demo · local word matching, no API calls. "
        "Switch to OpenAI for the retrieval agent."
    )

last_answer = next(
    (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None
)
col1, col2, col3, col4 = st.columns([1, 1, 1, 0.75])
col1.metric("Documents", len(documents))
col2.metric("Indexed chunks", sum(doc["chunks"] for doc in documents))
col3.metric("Last response", f"{last_answer['latency_ms']:.1f} ms" if last_answer else "—")
with col4:
    if st.button("Clear chat", disabled=not st.session_state.messages, use_container_width=True):
        st.session_state.messages = []
        st.rerun()
st.divider()
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("**Start with a source.**")
        st.write(
            "Upload a document in the sidebar, then ask a question. "
            "I’ll show the source chunks, the retrieval steps, and how long the answer took."
        )
        st.caption(
            "Try a specific question, then follow up naturally. "
            "Say hello to chat without retrieval."
        )
for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input(
    "Ask a question about your documents…", disabled=agent is None, max_chars=4000
):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    try:
        with st.spinner("Thinking and checking sources…"):
            answer = agent.answer(question, history)
        message = {
            "role": "assistant",
            "content": answer.text,
            "id": uuid.uuid4().hex,
            **asdict(answer),
        }
        message.pop("text")
        st.session_state.messages.append(message)
        st.rerun()
    except OpenAIError:
        st.session_state.messages.pop()
        st.error(
            "OpenAI request failed. Check your API key, connectivity and quota, then try again."
        )
    except Exception:
        st.session_state.messages.pop()
        st.error("The question could not be completed. Your documents are safe; please try again.")
