import json

import pytest

from agent_neutral_harness.memory.cold import ObjectStoreColdTier


class _Body:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class FakeS3:
    """Minimal in-memory stand-in for the boto3 S3 client."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def head_bucket(self, Bucket):
        return {}

    def create_bucket(self, Bucket):
        return {}

    def put_object(self, Bucket, Key, Body, **kw):
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise KeyError(Key)
        return {"Body": _Body(self.store[Key])}

    def list_objects_v2(self, Bucket, Prefix=""):
        return {"Contents": [{"Key": k} for k in self.store if k.startswith(Prefix)]}


def _toy_embed(text: str):
    # deterministic 5-dim bag-of-letters vector; enough for ranking in tests
    v = [0.0] * 5
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[(ord(ch) - 97) % 5] += 1.0
    return v


@pytest.fixture()
def cold():
    return ObjectStoreColdTier(bucket="test", client=FakeS3(), prefix="mem/")


def test_archive_moves_idle_rows_to_object_store(vault, cold):
    vault.save_memory("g", "fact", "old finding about deploy scripts")
    # make it look stale
    with vault._conn() as conn:
        conn.execute("UPDATE memories SET updated_at = '2000-01-01 00:00:00', created_at = '2000-01-01 00:00:00'")

    msg = cold.archive(vault, days=30)
    assert "Archived 1" in msg
    assert cold.client.store  # object written

    with vault._conn() as conn:
        row = conn.execute("SELECT archived, cold_key FROM memories").fetchone()
    assert row["archived"] == 1 and row["cold_key"] == "mem/1.json"


def test_archive_dry_run_changes_nothing(vault, cold):
    vault.save_memory("g", "fact", "another stale note")
    with vault._conn() as conn:
        conn.execute("UPDATE memories SET updated_at = '2000-01-01', created_at = '2000-01-01'")
    msg = cold.archive(vault, days=30, dry_run=True)
    assert "Would archive 1" in msg
    assert not cold.client.store


def test_archive_skips_fresh_rows(vault, cold):
    vault.save_memory("g", "fact", "fresh note just now")
    assert "No memories idle" in cold.archive(vault, days=30)


def test_restore_reinserts_row(vault, cold):
    cold.client.store["mem/99.json"] = json.dumps(
        {"id": 99, "scope": "g", "type": "fact", "content": "resurrected memory"}
    ).encode()
    cold.restore(vault, 99)
    out = vault.search_keyword("resurrected")
    assert "resurrected memory" in out


def test_scan_ranks_by_similarity(cold):
    cold.client.store["mem/1.json"] = json.dumps(
        {"id": 1, "scope": "g", "type": "fact", "content": "aaaa bbbb"}
    ).encode()
    cold.client.store["mem/2.json"] = json.dumps(
        {"id": 2, "scope": "g", "type": "fact", "content": "zzzz yyyy"}
    ).encode()
    hits = cold.scan("aaaa bbbb", _toy_embed, threshold=0.9, limit=5)
    assert [h["id"] for h in hits] == [1]
