from agent_neutral_harness.memory.vault import (
    CORROBORATION_THRESHOLD,
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


# --------------------------------------------------------------------------- #
# _verify_evidence_url -- SSRF guard must survive a redirect, not just the
# initial URL. These stub ``_url_is_safe`` so the fast suite stays network-free.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def close(self):
        pass


def test_verify_evidence_url_accepts_clean_200(vault, monkeypatch):
    monkeypatch.setattr("agent_neutral_harness.memory.vault._url_is_safe", lambda url: True)
    monkeypatch.setattr(
        "agent_neutral_harness.memory.vault.requests.head",
        lambda url, timeout=5, allow_redirects=False: _FakeResp(200),
    )
    assert vault._verify_evidence_url("https://example.com/evidence") is True


def test_verify_evidence_url_rejects_redirect_to_unsafe_host(vault, monkeypatch):
    # Only the original URL passes the safety check -- the redirect target
    # (standing in for something like a cloud metadata endpoint) does not.
    monkeypatch.setattr(
        "agent_neutral_harness.memory.vault._url_is_safe",
        lambda url: url == "https://example.com/evidence",
    )

    def fake_head(url, timeout=5, allow_redirects=False):
        assert url == "https://example.com/evidence", "must never request the unsafe redirect target"
        return _FakeResp(302, {"Location": "http://169.254.169.254/latest/meta-data/"})

    monkeypatch.setattr("agent_neutral_harness.memory.vault.requests.head", fake_head)
    assert vault._verify_evidence_url("https://example.com/evidence") is False


def test_verify_evidence_url_follows_redirect_to_validated_safe_host(vault, monkeypatch):
    monkeypatch.setattr("agent_neutral_harness.memory.vault._url_is_safe", lambda url: True)
    calls = []

    def fake_head(url, timeout=5, allow_redirects=False):
        calls.append(url)
        if url == "https://example.com/evidence":
            return _FakeResp(301, {"Location": "https://example.org/evidence"})
        return _FakeResp(200)

    monkeypatch.setattr("agent_neutral_harness.memory.vault.requests.head", fake_head)
    assert vault._verify_evidence_url("https://example.com/evidence") is True
    assert calls == ["https://example.com/evidence", "https://example.org/evidence"]


def test_verify_evidence_url_gives_up_after_max_redirects(vault, monkeypatch):
    monkeypatch.setattr("agent_neutral_harness.memory.vault._url_is_safe", lambda url: True)

    def fake_head(url, timeout=5, allow_redirects=False):
        # Always redirects to itself-ish -- an infinite chain.
        return _FakeResp(302, {"Location": "https://example.com/next"})

    monkeypatch.setattr("agent_neutral_harness.memory.vault.requests.head", fake_head)
    assert vault._verify_evidence_url("https://example.com/evidence") is False


def test_search_semantic_falls_back_to_keyword_without_chroma(vault):
    vault.save_memory("global", "fact", "vector store is optional")
    out = vault.search_semantic("vector store")
    assert "optional" in out


# --------------------------------------------------------------------------- #
# corroboration / confidence lifecycle -- exercised with a fake vector store
# --------------------------------------------------------------------------- #
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
