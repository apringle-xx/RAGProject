from pathlib import PurePath
from time import perf_counter

from rag.documents import chunk_document, document_id
from rag.providers import Embedder
from rag.store import VectorStore


class Library:
    def __init__(self, store: VectorStore, embedder: Embedder):
        if store.embedding_name != embedder.name:
            raise ValueError("The store and embedder must use the same model.")
        self.store = store
        self.embedder = embedder

    def ingest(self, filename: str, data: bytes) -> dict:
        started = perf_counter()
        source = PurePath(filename.replace("\\", "/")).name
        if self.store.has_document(document_id(source, data)):
            return {"source": source, "chunks": 0, "duplicate": True, "latency_ms": 0}
        chunks = chunk_document(source, data)
        vectors = self.embedder.embed([c.text for c in chunks])
        count = self.store.add(chunks, vectors)
        return {
            "source": source,
            "chunks": count,
            "duplicate": False,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    def search(self, query: str, k: int = 4):
        return self.store.search(self.embedder.embed([query]), k)
