"""Warm tier: a shared libSQL / Turso replica of the memory vault.

Identity is by **content hash**, not local row id. Local ids are assigned
per-machine, so an id-based sync silently loses data the moment two
machines mint the same id for different content. ``sync_key`` =
``sha256(scope \\x1f type \\x1f content)`` is stable across machines, so
the same finding written on two laptops is recognised as one row.

Genuine divergence -- the "same" memory edited differently on two
machines -- produces two rows with different content hashes. ``pull()``
runs every incoming row back through :meth:`MemoryVault.save_memory`, so
that divergence surfaces through the normal corroboration / review-band
machinery: a near-duplicate either bumps the existing memory's
corroboration count or lands as ``needs_review = 1`` for a human. The
review queue *is* the conflict log.

Requires the ``warm`` extra::

    pip install "agent-neutral-harness[warm]"

Credentials come from the constructor or ``TURSO_DATABASE_URL`` /
``TURSO_AUTH_TOKEN``.
"""

import hashlib
import logging
import os
import socket
import sqlite3

from agent_neutral_harness.memory._similarity import cosine_similarity

log = logging.getLogger(__name__)

_REMOTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    sync_key TEXT PRIMARY KEY,
    scope TEXT, type TEXT, content TEXT, tags TEXT,
    evidence_url TEXT, evidence_verified INTEGER DEFAULT 0,
    confidence TEXT DEFAULT 'hypothesis', corroborations INTEGER DEFAULT 0,
    origin TEXT, created_at TIMESTAMP, updated_at TIMESTAMP
);
"""

_PUSH_COLUMNS = (
    "sync_key", "scope", "type", "content", "tags", "evidence_url",
    "evidence_verified", "confidence", "corroborations", "origin",
    "created_at", "updated_at",
)


def content_sync_key(scope: str, mem_type: str, content: str) -> str:
    raw = f"{scope}\x1f{mem_type}\x1f{content}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _push_rows(local_conn, remote_conn, origin: str) -> int:
    """Insert local rows the remote is missing (by content hash). Idempotent."""
    remote_conn.execute(_REMOTE_SCHEMA)
    local_cols = {r[1] for r in local_conn.execute("PRAGMA table_info(memories)").fetchall()}

    def col(name: str) -> str:
        return name if name in local_cols else "NULL"

    rows = local_conn.execute(
        f"SELECT scope, type, content, {col('tags')}, {col('evidence_url')}, "
        f"{col('evidence_verified')}, {col('confidence')}, {col('corroborations')}, "
        f"{col('created_at')}, {col('updated_at')} "
        f"FROM memories WHERE COALESCE(archived, 0) = 0"
    ).fetchall()
    existing = {r[0] for r in remote_conn.execute("SELECT sync_key FROM memories").fetchall()}
    placeholders = ",".join("?" for _ in _PUSH_COLUMNS)
    pushed = 0
    for scope, mem_type, content, *rest in rows:
        key = content_sync_key(scope, mem_type, content)
        if key in existing:
            continue
        remote_conn.execute(
            f"INSERT INTO memories ({','.join(_PUSH_COLUMNS)}) VALUES ({placeholders})",
            (key, scope, mem_type, content, *rest[:4], origin, *rest[4:]),
        )
        existing.add(key)
        pushed += 1
    remote_conn.commit()
    return pushed


class TursoWarmTier:
    def __init__(
        self,
        vault=None,
        local_db_path: str | None = None,
        sync_url: str | None = None,
        auth_token: str | None = None,
        replica_path: str = ".agent-neutral-warm-replica.db",
        accept_threshold: float = 0.65,
        origin: str | None = None,
        remote_conn=None,
    ):
        self.vault = vault
        self.local_db_path = local_db_path or (getattr(vault, "db_path", None))
        self.sync_url = sync_url or os.environ.get("TURSO_DATABASE_URL")
        self.auth_token = auth_token or os.environ.get("TURSO_AUTH_TOKEN")
        self.replica_path = replica_path
        self.accept_threshold = accept_threshold
        self.origin = origin or socket.gethostname()
        self._injected_remote = remote_conn
        if remote_conn is None and not self.sync_url:
            raise ValueError(
                "Turso sync_url not provided and not in TURSO_DATABASE_URL"
            )

    # ---------------------------------------------------------------- #
    def _remote(self):
        if self._injected_remote is not None:
            self._injected_remote.execute(_REMOTE_SCHEMA)
            return self._injected_remote
        try:
            import libsql_experimental as libsql
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The warm tier needs: pip install 'agent-neutral-harness[warm]'"
            ) from exc
        conn = libsql.connect(
            self.replica_path, sync_url=self.sync_url, auth_token=self.auth_token or ""
        )
        conn.execute(_REMOTE_SCHEMA)
        conn.sync()
        return conn

    def _sync_remote(self, conn):
        if self._injected_remote is None and hasattr(conn, "sync"):
            conn.sync()

    def _local(self):
        return sqlite3.connect(self.local_db_path)

    # ---------------------------------------------------------------- #
    def push(self) -> str:
        remote = self._remote()
        local = self._local()
        try:
            n = _push_rows(local, remote, self.origin)
        finally:
            local.close()
        self._sync_remote(remote)
        return f"Pushed {n} new memories to the warm tier."

    def pull(self) -> str:
        """Bring down remote rows this vault doesn't have, through save_memory.

        Requires a ``vault``. Divergent near-duplicates surface as
        ``needs_review`` rather than being silently merged or dropped.
        """
        if self.vault is None:
            raise RuntimeError("pull() needs a MemoryVault (pass vault=...)")
        remote = self._remote()
        self._sync_remote(remote)
        rows = remote.execute(
            "SELECT scope, type, content, tags, evidence_url FROM memories"
        ).fetchall()

        inserted = corroborated = flagged = 0
        with self.vault._conn() as conn:
            have = {
                (r["scope"], r["type"], r["content"])
                for r in conn.execute("SELECT scope, type, content FROM memories")
            }
        for scope, mem_type, content, tags, evidence_url in rows:
            if (scope, mem_type, content) in have:
                continue
            msg = self.vault.save_memory(scope, mem_type, content, tags or "", evidence_url or "")
            if "Corroborated" in msg:
                corroborated += 1
            elif "needs_review" in msg:
                flagged += 1
            else:
                inserted += 1

        note = f" ({flagged} flagged needs_review)" if flagged else ""
        return (
            f"Pulled from the warm tier: {inserted} new, "
            f"{corroborated} corroborated existing{note}."
        )

    def scan(self, query: str, embed_fn, threshold: float | None = None, limit: int = 3) -> list[dict]:
        threshold = self.accept_threshold if threshold is None else threshold
        qv = embed_fn(query)
        if qv is None:
            return []
        remote = self._remote()
        self._sync_remote(remote)
        rows = remote.execute(
            "SELECT sync_key, scope, type, content, confidence FROM memories"
        ).fetchall()
        scored = []
        for key, scope, mem_type, content, confidence in rows:
            sim = cosine_similarity(qv, embed_fn(content))
            if sim >= threshold:
                scored.append((sim, {
                    "sync_key": key, "scope": scope, "type": mem_type,
                    "content": content, "confidence": confidence,
                }))
        scored.sort(key=lambda t: -t[0])
        return [d for _, d in scored[:limit]]
