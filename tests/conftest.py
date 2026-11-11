import pytest

from rag.providers import OfflineEmbeddings
from rag.service import Library
from rag.store import VectorStore


@pytest.fixture
def library(tmp_path):
    embedder = OfflineEmbeddings()
    return Library(VectorStore(tmp_path, embedder.name), embedder)
