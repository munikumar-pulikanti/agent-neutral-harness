import sqlite3

from agent_neutral_harness.memory.warm import _shared_columns, sync_one_way

_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY, scope TEXT, type TEXT, content TEXT, tags TEXT,
    evidence_url TEXT, evidence_verified INTEGER DEFAULT 0,
    confidence TEXT DEFAULT 'hypothesis', corroborations INTEGER DEFAULT 0,
    created_at TIMESTAMP, updated_at TIMESTAMP
);
"""


def _db(rows=()):
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA)
    for r in rows:
        c.execute(
            "INSERT INTO memories (id, scope, type, content) VALUES (?, ?, ?, ?)", r
        )
    c.commit()
    return c


def test_sync_one_way_copies_only_missing_rows():
    src = _db([(1, "g", "fact", "alpha"), (2, "g", "fact", "beta"), (3, "g", "fact", "gamma")])
    dst = _db([(1, "g", "fact", "alpha")])
    copied = sync_one_way(src, dst)
    assert copied == 2
    ids = {r[0] for r in dst.execute("SELECT id FROM memories")}
    assert ids == {1, 2, 3}
    # idempotent
    assert sync_one_way(src, dst) == 0


def test_shared_columns_is_intersection():
    a = _db()
    b = sqlite3.connect(":memory:")
    b.executescript("CREATE TABLE memories (id INTEGER PRIMARY KEY, scope TEXT, content TEXT)")
    cols = _shared_columns(a, b)
    assert cols == ["id", "scope", "content"]
