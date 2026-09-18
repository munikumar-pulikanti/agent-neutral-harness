"""Integration tests against real backend servers.

Skipped unless the relevant server is reachable (env vars set by CI):
  MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY  -> cold tier
  LIBSQL_URL  (+ optional LIBSQL_AUTH_TOKEN)            -> warm tier

Run locally, e.g.::

    docker run -p 9000:9000 -e MINIO_ROOT_USER=minioadmin \\
        -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data
    MINIO_ENDPOINT=http://localhost:9000 MINIO_ACCESS_KEY=minioadmin \\
        MINIO_SECRET_KEY=minioadmin uv run pytest -m integration
"""

import os
import time
import uuid

import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# cold tier -> real MinIO / S3
# --------------------------------------------------------------------------- #
@pytest.fixture()
def minio_cold():
    endpoint = os.environ.get("MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("MINIO_ENDPOINT not set")
    pytest.importorskip("boto3")
    from agent_neutral_harness.memory.cold import ObjectStoreColdTier

    return ObjectStoreColdTier(
        bucket=f"anh-test-{uuid.uuid4().hex[:8]}",
        endpoint=endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        region="us-east-1",
        prefix="memories/",
    )


def test_cold_archive_and_restore_roundtrip_real_s3(vault, minio_cold):
    vault.save_memory("g", "fact", "a genuinely old finding worth archiving")
    with vault._conn() as conn:
        conn.execute("UPDATE memories SET updated_at = '2000-01-01', created_at = '2000-01-01'")

    assert "Archived 1" in minio_cold.archive(vault, days=1)
    with vault._conn() as conn:
        row = conn.execute("SELECT archived, cold_key FROM memories").fetchone()
    assert row["archived"] == 1

    # keyword search no longer surfaces it (archived), restore brings it back
    assert "No matching" in vault.search_keyword("archiving")
    assert "Restored" in minio_cold.restore(vault, 1)
    assert "old finding" in vault.search_keyword("archiving")


def test_cold_scan_ranks_real_objects(vault, minio_cold):
    def toy_embed(text: str):
        v = [0.0] * 6
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[(ord(ch) - 97) % 6] += 1.0
        return v

    minio_cold._ensure_bucket()
    minio_cold.client.put_object(
        Bucket=minio_cold.bucket, Key="memories/1.json",
        Body=b'{"id": 1, "scope": "g", "type": "fact", "content": "aaaa bbbb cccc"}',
    )
    minio_cold.client.put_object(
        Bucket=minio_cold.bucket, Key="memories/2.json",
        Body=b'{"id": 2, "scope": "g", "type": "fact", "content": "wwww xxxx yyyy"}',
    )
    hits = minio_cold.scan("aaaa bbbb cccc", toy_embed, threshold=0.95, limit=5)
    assert [h["id"] for h in hits] == [1]


# --------------------------------------------------------------------------- #
# warm tier -> real libSQL server
# --------------------------------------------------------------------------- #
@pytest.fixture()
def libsql_warm(vault):
    url = os.environ.get("LIBSQL_URL")
    if not url:
        pytest.skip("LIBSQL_URL not set")
    pytest.importorskip("libsql_experimental")
    from agent_neutral_harness.memory.warm import TursoWarmTier

    tier = TursoWarmTier(
        vault=vault, sync_url=url,
        auth_token=os.environ.get("LIBSQL_AUTH_TOKEN", ""),
        replica_path=f"/tmp/anh-warm-{uuid.uuid4().hex[:8]}.db",
    )
    # LIBSQL_URL being set is an explicit assertion that a real server should
    # be reachable here -- unlike the "not set" skip above, a connection
    # failure at this point is a real failure (broken CI service container,
    # or an actual regression in the warm tier), not something to swallow
    # into a skip. That used to make this job go green even if the warm
    # tier silently broke and the server also failed to start -- the two
    # failures would cancel out into a false pass.
    #
    # Retries absorb the container's own startup lag (a timing issue, not a
    # reason to treat a genuine failure as a skip) before giving up.
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            tier._remote()
            return tier
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 4:
                time.sleep(2)
    raise RuntimeError(f"libSQL at {url} did not become reachable: {last_exc}") from last_exc


def test_warm_push_pull_roundtrip_real_libsql(vault, libsql_warm):
    vault.save_memory("g", "fact", f"warm roundtrip marker {uuid.uuid4().hex}")
    assert "Pushed" in libsql_warm.push()

    # a second vault pulls it down
    from agent_neutral_harness.memory.vault import MemoryVault

    other = MemoryVault(
        db_path=vault.db_path + ".other", chroma_path=vault.chroma_path + "-other"
    )
    other._chroma_ready = True
    other._collection = None
    libsql_warm.vault = other
    libsql_warm.pull()
    assert "warm roundtrip marker" in other.search_keyword("roundtrip marker")
