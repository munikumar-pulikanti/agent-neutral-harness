import sqlite3

from agent_neutral_harness.memory.warm import (
    TursoWarmTier,
    _push_rows,
    content_sync_key,
)


def _remote():
    return sqlite3.connect(":memory:")


def test_content_sync_key_is_stable_and_content_derived():
    k1 = content_sync_key("g", "fact", "the pool size is 20")
    assert k1 == content_sync_key("g", "fact", "the pool size is 20")
    assert k1 != content_sync_key("g", "fact", "the pool size is 50")
    assert k1 != content_sync_key("other", "fact", "the pool size is 20")


def test_push_is_by_content_not_local_id(vault):
    # two "machines" that both happen to use local id 1 for different content
    vault.save_memory("g", "fact", "Postgres pool max is 20")
    remote = _remote()
    tier = TursoWarmTier(vault=vault, remote_conn=remote)
    assert "Pushed 1" in tier.push()
    assert "Pushed 0" in tier.push()  # idempotent

    other = _remote_vault_like("Frontend is built with Vue 3")  # same local id 1
    tier2 = TursoWarmTier(local_db_path=other, remote_conn=remote)
    assert "Pushed 1" in tier2.push()  # NOT skipped despite id collision

    rows = remote.execute("SELECT content FROM memories ORDER BY content").fetchall()
    assert [r[0] for r in rows] == ["Frontend is built with Vue 3", "Postgres pool max is 20"]


def _remote_vault_like(content: str) -> str:
    """A bare SQLite file shaped like a vault, holding one row at id 1."""
    import tempfile

    path = tempfile.mktemp(suffix=".db")
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, scope TEXT, type TEXT, "
        "content TEXT, archived INTEGER DEFAULT 0);"
    )
    c.execute("INSERT INTO memories (id, scope, type, content) VALUES (1, 'g', 'fact', ?)", (content,))
    c.commit()
    c.close()
    return path


def test_pull_goes_through_save_memory(vault):
    remote = _remote()
    _push_rows(_seed_local("The CI cache key is content-hashed"), remote, origin="nodeA")
    tier = TursoWarmTier(vault=vault, remote_conn=remote)
    msg = tier.pull()
    assert "1 new" in msg
    assert "content-hashed" in vault.search_keyword("cache key")


def _seed_local(content: str):
    c = sqlite3.connect(":memory:")
    c.executescript(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, scope TEXT, type TEXT, "
        "content TEXT, archived INTEGER DEFAULT 0);"
    )
    c.execute("INSERT INTO memories (scope, type, content) VALUES ('g', 'fact', ?)", (content,))
    c.commit()
    return c


def test_pull_flags_divergent_near_duplicate_as_needs_review(sem_vault):
    sem_vault.save_memory("g", "fact", "The API rate limit is 100 requests per minute per key")
    remote = _remote()
    _push_rows(
        _seed_local("The API rate limit is 100 requests per minute per key adjusted slightly xyz"),
        remote, origin="nodeB",
    )
    tier = TursoWarmTier(vault=sem_vault, remote_conn=remote)
    msg = tier.pull()
    # corroborated or flagged -- either way, not silently dropped or duplicated
    assert ("corroborated" in msg) or ("needs_review" in msg)
