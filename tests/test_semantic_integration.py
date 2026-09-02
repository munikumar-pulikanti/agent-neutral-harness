"""Real ChromaDB + embedding-model tests. Skipped unless the 'memory'
extra is installed (they download ~90MB on first run).

Run: uv sync --extra memory --group dev && uv run pytest -m semantic
"""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.semantic


@pytest.fixture()
def real_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_MEMORY_VAULT_DIR", str(tmp_path))
    from agent_neutral_harness.memory.vault import MemoryVault

    return MemoryVault(db_path=str(tmp_path / "v.db"), chroma_path=str(tmp_path / "chroma"))


def test_semantic_search_finds_paraphrase(real_vault):
    real_vault.save_memory("g", "fact", "The nightly backup job runs at 2am UTC via cron")
    out = real_vault.search_semantic("when does the automated backup happen")
    assert "backup" in out.lower()


def test_distinct_claims_do_not_corroborate(real_vault):
    real_vault.save_memory("g", "fact", "Postgres connection pool max size is 20")
    msg = real_vault.save_memory("g", "fact", "The frontend is built with Vue 3 and Vite")
    assert "hypothesis" in msg
    assert "Corroborated" not in msg


def test_paraphrase_corroborates(real_vault):
    real_vault.save_memory("g", "fact", "API rate limit is 100 requests per minute per key")
    msg = real_vault.save_memory("g", "fact", "API rate limit is 100 requests per minute per key")
    assert "Corroborated memory #1" in msg
