import importlib

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
