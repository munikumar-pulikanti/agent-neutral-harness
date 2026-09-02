from agent_neutral_harness.memory.vault import _sanitize_fts_query, _url_is_safe


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
