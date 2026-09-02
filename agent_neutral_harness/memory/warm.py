"""Warm tier: a shared libSQL / Turso replica of the memory vault.

Push/pull sync is by row id -- a row that already exists on the other
side (same id) is never re-copied. This is intentionally simple
(insert-only, no conflict resolution): the vault is append-mostly and ids
are assigned locally, so collisions are rare and a human resolves them.

Requires the ``warm`` extra::

    pip install "agent-neutral-harness[warm]"

Credentials are read from the constructor or, if omitted, from
``TURSO_DATABASE_URL`` / ``TURSO_AUTH_TOKEN`` in the environment.
"""

import logging
import os
import sqlite3

from agent_neutral_harness.memory._similarity import cosine_similarity

log = logging.getLogger(__name__)

# Columns synced between local and remote. Only columns present in *both*
# schemas are actually copied (computed at sync time), so this can list a
# superset safely.
SYNC_COLUMNS = (
    "id", "scope", "type", "content", "tags", "evidence_url", "evidence_verified",
    "confidence", "corroborations", "created_at", "updated_at",
)

_REMOTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    scope TEXT, type TEXT, content TEXT, tags TEXT,
    evidence_url TEXT, evidence_verified INTEGER DEFAULT 0,
    confidence TEXT DEFAULT 'hypothesis', corroborations INTEGER DEFAULT 0,
    created_at TIMESTAMP, updated_at TIMESTAMP
);
"""


def _shared_columns(conn_a, conn_b) -> list[str]:
    def cols(c):
        return {r[1] for r in c.execute("PRAGMA table_info(memories)").fetchall()}

    have = cols(conn_a) & cols(conn_b)
    return [c for c in SYNC_COLUMNS if c in have]


def sync_one_way(src, dst) -> int:
    """Copy rows present in ``src`` but not ``dst`` (matched by id). Returns count."""
    columns = _shared_columns(src, dst)
    existing = {r[0] for r in dst.execute("SELECT id FROM memories").fetchall()}
    src_rows = src.execute(f"SELECT {','.join(columns)} FROM memories").fetchall()
    placeholders = ",".join("?" for _ in columns)
    copied = 0
    for row in src_rows:
        if row[0] in existing:
            continue
        dst.execute(
            f"INSERT INTO memories ({','.join(columns)}) VALUES ({placeholders})", tuple(row)
        )
        copied += 1
    dst.commit()
    return copied


class TursoWarmTier:
    def __init__(
        self,
        local_db_path: str,
        sync_url: str | None = None,
        auth_token: str | None = None,
        replica_path: str = ".agent-neutral-warm-replica.db",
        accept_threshold: float = 0.65,
    ):
        self.local_db_path = local_db_path
        self.sync_url = sync_url or os.environ.get("TURSO_DATABASE_URL")
        self.auth_token = auth_token or os.environ.get("TURSO_AUTH_TOKEN")
        self.replica_path = replica_path
        self.accept_threshold = accept_threshold
        if not self.sync_url or not self.auth_token:
            raise ValueError(
                "Turso sync_url / auth_token not provided and not in "
                "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN"
            )

    def _remote(self):
        try:
            import libsql_experimental as libsql
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The warm tier needs: pip install 'agent-neutral-harness[warm]'") from exc
        conn = libsql.connect(
            self.replica_path, sync_url=self.sync_url, auth_token=self.auth_token
        )
        conn.execute(_REMOTE_SCHEMA)
        conn.sync()
        return conn

    def _local(self):
        conn = sqlite3.connect(self.local_db_path)
        return conn

    def push(self) -> str:
        remote = self._remote()
        local = self._local()
        try:
            n = sync_one_way(local, remote)
        finally:
            local.close()
        remote.sync()
        return f"Pushed {n} new memories to the warm tier."

    def pull(self) -> str:
        remote = self._remote()
        local = self._local()
        try:
            n = sync_one_way(remote, local)
        finally:
            local.close()
        return f"Pulled {n} new memories from the warm tier."

    def scan(self, query: str, embed_fn, threshold: float | None = None, limit: int = 3) -> list[dict]:
        """Semantic scan of the shared replica for the search cascade."""
        threshold = self.accept_threshold if threshold is None else threshold
        qv = embed_fn(query)
        if qv is None:
            return []
        remote = self._remote()
        rows = remote.execute(
            "SELECT id, scope, type, content, confidence FROM memories"
        ).fetchall()
        scored = []
        for row in rows:
            content = row[3]
            sim = cosine_similarity(qv, embed_fn(content))
            if sim >= threshold:
                scored.append((sim, {
                    "id": row[0], "scope": row[1], "type": row[2],
                    "content": content, "confidence": row[4],
                }))
        scored.sort(key=lambda t: -t[0])
        return [d for _, d in scored[:limit]]
