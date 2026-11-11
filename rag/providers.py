"""OpenAI production providers and explicitly opt-in offline demonstration providers."""

import hashlib
import json
import os
import re
from typing import Protocol

import numpy as np
from openai import OpenAI

from rag.grounding import is_conversation
from rag.types import ToolCall, Turn

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the uploaded library. Rewrite or decompose queries as needed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Model(Protocol):
    def complete(self, messages: list[dict], *, tools: bool) -> Turn: ...


def openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in your environment, or select Offline demo.")
    return OpenAI(timeout=30, max_retries=2)


class OpenAIEmbeddings:
    def __init__(self, client=None, model: str | None = None):
        self.client = client if client is not None else openai_client()
        self.model = model or os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for offset in range(0, len(texts), 64):
            response = self.client.embeddings.create(
                model=self.model, input=texts[offset : offset + 64], encoding_format="float"
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return np.asarray(vectors, dtype="float32")


class OpenAIModel:
    def __init__(self, client=None, model: str | None = None):
        self.client = client if client is not None else openai_client()
        self.model = model or os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")

    def complete(self, messages: list[dict], *, tools: bool) -> Turn:
        options = {"tools": [SEARCH_TOOL], "parallel_tool_calls": False} if tools else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=1200, **options
        )
        message = response.choices[0].message
        return Turn(
            message.content or "",
            [
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in message.tool_calls or []
            ],
        )


STOP_WORDS = set(
    "a an the on in of to is are was were be and or what when who where does do how "
    "it its this that i you me my your please tell about project owns owner s".split()
)


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


class OfflineEmbeddings:
    """Stable hashed bag of words. Only for demo/testing; no semantic understanding."""

    name = "offline:hashed-words-v1"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 512), dtype="float32")
        for row, text in enumerate(texts):
            for term in terms(text) or {"empty"}:
                column = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big") % 512
                vectors[row, column] += 1
        return vectors


class OfflineModel:
    """A transparent extractive demo, not an LLM substitute or quality benchmark."""

    def complete(self, messages: list[dict], *, tools: bool) -> Turn:
        question_positions = [
            i
            for i, m in enumerate(messages)
            if m["role"] == "user" and not m["content"].startswith("Evidence JSON:\n")
        ]
        questions = [messages[i]["content"] for i in question_positions]
        question = questions[-1]
        if is_conversation(question):
            return Turn("Hello! Upload a document and ask me about it.")
        evidence_messages = [
            m["content"]
            for m in messages[question_positions[-1] + 1 :]
            if m["role"] == "tool" or (m.get("content") or "").startswith("Evidence JSON:\n")
        ]
        query = question
        # An explicit subject wins over a pronoun elsewhere in the same question.
        # This is a small demo heuristic; the OpenAI provider rewrites using the LLM.
        explicit_subject = re.search(r"\bproject\s+\w+", question, re.I) or re.search(
            r"\b[A-Z][\w-]+\b", " ".join(question.split()[1:])
        )
        if (
            re.search(r"\b(it|its|they|their|that)\b", question, re.I)
            and not explicit_subject
            and len(questions) > 1
        ):
            query = questions[-2] + " " + question
        if tools and not evidence_messages:
            return Turn(
                calls=[ToolCall("offline-search", "search_documents", json.dumps({"query": query}))]
            )
        evidence = []
        for message in evidence_messages:
            evidence.extend(json.loads(message.removeprefix("Evidence JSON:\n")).get("sources", []))
        matching = [source for source in evidence if terms(query) & terms(source["text"])]
        if not matching:
            return Turn("I don't know based on the available documents.")
        return Turn("\n\n".join(f"{s['text']} [{s['label']}]" for s in matching[:3]))
