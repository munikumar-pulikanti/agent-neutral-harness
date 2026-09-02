import importlib
from difflib import SequenceMatcher

import pytest


@pytest.fixture()
def metrics_mod(tmp_path, monkeypatch):
    """A fresh metrics module pointed at an isolated temp database."""
    monkeypatch.setenv("AGENT_NEUTRAL_HARNESS_METRICS_DB", str(tmp_path / "metrics.db"))
    from agent_neutral_harness import metrics

    importlib.reload(metrics)
    metrics._initialized.clear()
    metrics.init_db()
    yield metrics


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    """A MemoryVault backed by a temp dir, with the semantic tier disabled."""
    monkeypatch.setenv("AI_MEMORY_VAULT_DIR", str(tmp_path / "vault"))
    from agent_neutral_harness.memory.vault import MemoryVault

    v = MemoryVault(
        db_path=str(tmp_path / "vault" / "global_brain.db"),
        chroma_path=str(tmp_path / "vault" / "chroma"),
    )
    # Force keyword-only behaviour regardless of whether chromadb is installed.
    v._chroma_ready = True
    v._collection = None
    return v


class FakeCollection:
    """In-memory stand-in for a Chroma collection, ranked by string ratio."""

    def __init__(self):
        self.docs: dict[str, str] = {}

    def upsert(self, ids, documents, metadatas=None, embeddings=None):
        for i, d in zip(ids, documents, strict=False):
            self.docs[str(i)] = d

    def delete(self, ids):
        for i in ids:
            self.docs.pop(str(i), None)

    def query(self, query_texts, n_results=1, include=None):
        q = query_texts[0]
        ranked = sorted(
            self.docs.items(),
            key=lambda kv: -SequenceMatcher(None, q, kv[1]).ratio(),
        )[:n_results]
        return {
            "ids": [[k for k, _ in ranked]],
            "distances": [[1.0 - SequenceMatcher(None, q, v).ratio() for _, v in ranked]],
            "documents": [[v for _, v in ranked]],
            "metadatas": [[{} for _ in ranked]],
        }


@pytest.fixture()
def sem_vault(tmp_path):
    """A MemoryVault whose semantic tier is a FakeCollection (no ChromaDB)."""
    from agent_neutral_harness.memory.vault import MemoryVault

    v = MemoryVault(db_path=str(tmp_path / "v.db"), chroma_path=str(tmp_path / "chroma"))
    v._chroma_ready = True
    v._collection = FakeCollection()
    return v
