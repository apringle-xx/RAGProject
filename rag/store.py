"""Transactional local vectors/metadata in SQLite, cosine search with FAISS.

FAISS is rebuilt from persisted float32 vectors on each read. This deliberately
avoids unsafe pickle and mismatched index/metadata files after interrupted writes.
The small local library is capped at 10,000 chunks.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from filelock import FileLock

from rag.types import Chunk, Hit


class VectorStore:
    def __init__(self, directory: str | Path, embedding_name: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "library.sqlite3"
        self.lock = FileLock(str(self.directory / "library.lock"))
        self.embedding_name = embedding_name
        with self.lock, self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(id TEXT PRIMARY KEY, source TEXT, document_id TEXT, metadata TEXT, vector BLOB)"
            )
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('embedding_name', ?)", (embedding_name,)
            )
            if (
                db.execute("SELECT value FROM settings WHERE key='embedding_name'").fetchone()[0]
                != embedding_name
            ):
                raise ValueError(
                    "Embedding model changed. Use a new library directory or clear it."
                )

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def has_document(self, doc_id: str) -> bool:
        with self._connect() as db:
            return (
                db.execute("SELECT 1 FROM chunks WHERE document_id=? LIMIT 1", (doc_id,)).fetchone()
                is not None
            )

    def documents(self) -> list[dict]:
        with self._connect() as db:
            return [
                {"source": source, "chunks": count}
                for source, count in db.execute(
                    "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
                )
            ]

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        vectors = self._matrix(vectors)
        if len(vectors) != len(chunks) or len({c.source for c in chunks}) != 1:
            raise ValueError("Expected matching vectors for a single document.")
        with self.lock, self._connect() as db:
            dimension = db.execute("SELECT value FROM settings WHERE key='dimension'").fetchone()
            if dimension and int(dimension[0]) != vectors.shape[1]:
                raise ValueError("Embedding dimension changed; use a new library.")
            db.execute("DELETE FROM chunks WHERE source=?", (chunks[0].source,))
            count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count + len(chunks) > 10_000:
                raise ValueError("Library limit reached (10,000 chunks). Clear the library first.")
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('dimension', ?)", (vectors.shape[1],)
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (c.id, c.source, c.document_id, json.dumps(asdict(c)), v.tobytes())
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        return len(chunks)

    @staticmethod
    def _matrix(vectors) -> np.ndarray:
        matrix = np.array(vectors, dtype="float32", copy=True)
        if (
            matrix.ndim != 2
            or not matrix.shape[0]
            or not matrix.shape[1]
            or not np.isfinite(matrix).all()
            or np.any(np.linalg.norm(matrix, axis=1) == 0)
        ):
            raise ValueError("Expected finite, nonzero embedding vectors.")
        faiss.normalize_L2(matrix)
        return matrix

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[Hit]:
        if not 1 <= k <= 20:
            raise ValueError("Retrieval k must be between 1 and 20.")
        query = self._matrix(query_vector)
        if len(query) != 1:
            raise ValueError("Expected exactly one query vector.")
        with self._connect() as db:
            rows = db.execute("SELECT metadata, vector FROM chunks ORDER BY rowid").fetchall()
        if not rows:
            return []
        vectors = np.stack([np.frombuffer(row[1], dtype="float32") for row in rows])
        if query.shape[1] != vectors.shape[1]:
            raise ValueError("Query embedding dimension does not match the library.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, positions = index.search(query, min(k, len(rows)))
        return [
            Hit(Chunk(**json.loads(rows[int(position)][0])), float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]

    def clear(self) -> None:
        with self.lock, self._connect() as db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM settings WHERE key='dimension'")
