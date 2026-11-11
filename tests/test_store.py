import numpy as np
import pytest

from rag.documents import chunk_document
from rag.providers import OfflineEmbeddings
from rag.service import Library
from rag.store import VectorStore


def test_ingest_persists_searches_and_skips_duplicates(library, tmp_path):
    result = library.ingest("atlas.txt", b"Atlas launches in March. Maya Chen owns Atlas.")
    assert result["chunks"] == 1
    assert result["latency_ms"] > 0
    library.ingest("support.txt", b"Support responds within 48 hours.")
    assert library.ingest("atlas.txt", b"Atlas launches in March. Maya Chen owns Atlas.")[
        "duplicate"
    ]
    reopened = Library(VectorStore(tmp_path, library.embedder.name), OfflineEmbeddings())
    assert len(reopened.store.documents()) == 2
    assert reopened.search("Atlas", k=1)[0].chunk.source == "atlas.txt"
    assert reopened.search("48 hours", k=1)[0].chunk.source == "support.txt"


def test_replacing_filename_does_not_keep_stale_chunks(library):
    library.ingest("notes.txt", b"Old content")
    library.ingest("notes.txt", b"New content")
    hits = library.search("content")
    assert len(hits) == 1
    assert hits[0].chunk.text == "New content"


def test_failed_ingestion_preserves_existing_library(library):
    library.ingest("notes.txt", b"Original content")
    with pytest.raises(ValueError):
        library.ingest("notes.txt", b"")
    chunks = chunk_document("notes.txt", b"Replacement content")
    with pytest.raises(ValueError, match="dimension"):
        library.store.add(chunks, np.ones((1, 3)))
    assert library.search("Original")[0].chunk.text == "Original content"


def test_failed_transaction_rolls_back_source_deletion(library):
    library.ingest("notes.txt", b"Original content")
    # Duplicate primary keys fail after the old source has been deleted in the transaction.
    chunk = chunk_document("notes.txt", b"Replacement")[0]
    with pytest.raises(Exception, match="UNIQUE"):
        library.store.add([chunk, chunk], np.ones((2, 512)))
    assert library.search("Original")[0].chunk.text == "Original content"


def test_empty_store_clear_and_model_mismatch(library, tmp_path):
    assert library.search("anything") == []
    library.ingest("notes.txt", b"Some content")
    library.store.clear()
    assert library.store.documents() == []
    assert library.search("anything") == []
    with pytest.raises(ValueError, match="model changed"):
        VectorStore(tmp_path, "other-model")


@pytest.mark.parametrize("vectors", [np.zeros((1, 512)), [[np.nan]], [1, 2], []])
def test_invalid_embeddings_rejected(library, vectors):
    with pytest.raises(ValueError, match="embedding vectors"):
        library.store.add(chunk_document("x.txt", b"hello"), vectors)


def test_search_dimension_and_k_validation(library):
    library.ingest("x.txt", b"hello")
    with pytest.raises(ValueError, match="dimension"):
        library.store.search(np.ones((1, 2)))
    with pytest.raises(ValueError, match="k must"):
        library.search("hello", k=0)
