import pytest

from agent_neutral_harness.memory.vault import (
    CORROBORATION_THRESHOLD,
    MemoryVault,
    _sanitize_fts_query,
    _url_is_safe,
)


def test_save_and_keyword_search(vault):
    vault.save_memory("global", "fact", "The deploy script lives in bin/release.sh")
    out = vault.search_keyword("deploy script")
    assert "release.sh" in out
    assert "Confidence: hypothesis" in out


def test_keyword_search_survives_fts_metacharacters(vault):
    vault.save_memory("global", "fact", "handling quotes and dashes")
    # none of these should raise sqlite3.OperationalError
    for q in ['"unterminated', "foo -bar", "a NEAR b", "wild*card", "x AND OR y"]:
        vault.search_keyword(q)


def test_new_memory_defaults_to_hypothesis(vault):
    msg = vault.save_memory("global", "fact", "a brand new isolated claim")
    assert "hypothesis" in msg


def test_empty_content_refused(vault):
    assert vault.save_memory("global", "fact", "   ").startswith("Refused")


def test_sanitize_fts_query():
    assert _sanitize_fts_query("deploy script") == '"deploy" "script"'
    assert _sanitize_fts_query("   ") == ""
    # embedded quotes are escaped, never left to break the MATCH expression
    assert '"' in _sanitize_fts_query('say "hi"')


def test_url_is_safe_rejects_private_and_bad_schemes():
    assert not _url_is_safe("http://localhost/x")
    assert not _url_is_safe("http://127.0.0.1/x")
    assert not _url_is_safe("http://169.254.169.254/latest/meta-data/")
    assert not _url_is_safe(" file:///etc/passwd")
    assert not _url_is_safe("not-a-url")


def test_search_semantic_falls_back_to_keyword_without_chroma(vault):
    vault.save_memory("global", "fact", "vector store is optional")
    out = vault.search_semantic("vector store")
    assert "optional" in out


# --------------------------------------------------------------------------- #
# corroboration / confidence lifecycle -- exercised with a fake vector store
# --------------------------------------------------------------------------- #
from difflib import SequenceMatcher  # noqa: E402


class FakeCollection:
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
        ids = [k for k, _ in ranked]
        dists = [1.0 - SequenceMatcher(None, q, v).ratio() for _, v in ranked]
        return {"ids": [ids], "distances": [dists], "documents": [[v for _, v in ranked]],
                "metadatas": [[{} for _ in ranked]]}


@pytest.fixture()
def sem_vault(tmp_path):
    v = MemoryVault(db_path=str(tmp_path / "v.db"), chroma_path=str(tmp_path / "chroma"))
    v._chroma_ready = True
    v._collection = FakeCollection()
    return v


def test_restated_claim_corroborates_not_duplicates(sem_vault):
    sem_vault.save_memory("g", "fact", "The build cache lives under ~/.cache/build")
    msg = sem_vault.save_memory("g", "fact", "The build cache lives under ~/.cache/build")
    assert "Corroborated memory #1" in msg
    with sem_vault._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT corroborations FROM memories WHERE id=1").fetchone()[0] == 2


def test_confidence_capped_at_suspected_without_evidence(sem_vault):
    msg = ""
    for _ in range(5):
        msg = sem_vault.save_memory("g", "fact", "Retries use exponential backoff with jitter")
    assert "confidence=suspected" in msg
    with sem_vault._conn() as conn:
        assert conn.execute("SELECT confidence FROM memories WHERE id=1").fetchone()[0] == "suspected"


def test_corroboration_threshold_is_high():
    assert CORROBORATION_THRESHOLD >= 0.8
