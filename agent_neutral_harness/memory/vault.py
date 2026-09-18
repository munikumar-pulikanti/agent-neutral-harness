"""Persistent memory vault with evidence-gated confidence scoring and a
hot -> warm -> cold search cascade.

Hot tier (always available):
- Local SQLite + FTS5 for keyword search.
- ChromaDB (``all-MiniLM-L6-v2``) for semantic vector search, used when
  the optional ``memory`` extra is installed; the vault degrades to
  keyword-only search when it is not.

Warm tier (optional, ``agent_neutral_harness.memory.warm``): a shared
libSQL/Turso replica. Cold tier (optional,
``agent_neutral_harness.memory.cold``): an S3-compatible object store for
archived rows. Both are injected into :class:`MemoryVault`; when absent
the cascade simply stops at the hot tier.

Confidence lifecycle on save:
- A brand-new claim is ``hypothesis``.
- A save whose content closely matches an existing memory is treated as
  *corroboration of that memory* (its counter is bumped and its
  confidence promoted) rather than a duplicate insert.
- Promotion is evidence-gated: without a verified, reachable evidence URL
  a memory is capped at ``suspected`` no matter how often it is restated.
  ``confirmed`` needs >= ``CONFIRM_MIN_CORROBORATIONS`` and evidence.
- A save that is *related but not clearly the same* (the review band) is
  inserted with ``needs_review = 1`` for a human to disambiguate.
"""

import ipaddress
import json
import logging
import os
import socket
import sqlite3
from contextlib import contextmanager
from urllib.parse import urljoin, urlparse

import requests

try:
    import chromadb
    from chromadb.utils import embedding_functions

    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _CHROMA_AVAILABLE = False

log = logging.getLogger(__name__)

DB_DIR = (
    os.environ.get("AI_MEMORY_VAULT_DIR")
    or os.environ.get("AGENT_NEUTRAL_HARNESS_HOME")
    or os.path.expanduser("~/.ai-memory-vault")
)
DB_PATH = os.path.join(DB_DIR, "global_brain.db")
CHROMA_PATH = os.path.join(DB_DIR, "chroma")

EMBED_MODEL = "all-MiniLM-L6-v2"

# Save-time confidence-lifecycle thresholds (cosine similarity, 1 = identical).
CORROBORATION_THRESHOLD = 0.85  # "this is the same finding restated"
REVIEW_BAND_LOW = 0.5           # below this: unrelated, insert fresh
REVIEW_BAND_HIGH = 0.85         # [LOW, HIGH): related but same-or-conflicting is unclear
CONFIRM_MIN_CORROBORATIONS = 3  # corroboration_count needed for "confirmed" (with evidence)

MAX_EVIDENCE_REDIRECTS = 5  # hops walked manually before giving up as unverified

# Search-cascade acceptance thresholds per tier.
HOT_ACCEPT_THRESHOLD = 0.4      # your own curated data, lowest bar
WARM_ACCEPT_THRESHOLD = 0.65    # shared / less-curated data, higher bar
COLD_ACCEPT_THRESHOLD = 0.5     # archived, last resort

VALID_CONFIDENCE = ("hypothesis", "suspected", "confirmed")

_NO_KEYWORD_MATCH = "No matching memories found (checked keyword FTS)."
_NO_SEMANTIC_MATCH = "No semantic matches found (checked hot, warm, and cold tiers)."
NO_RESULT_MARKERS = (_NO_KEYWORD_MATCH, _NO_SEMANTIC_MATCH, "No semantic matches found")


def _url_is_safe(url: str) -> bool:
    """Reject non-HTTP(S) URLs and any host that resolves to a private,
    loopback, link-local or reserved address. Evidence URLs come from
    arbitrary MCP clients, so this HEAD request is an SSRF sink without it.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except (ValueError, socket.gaierror, OSError):
        return False
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            return False
    return True


def _sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each whitespace-delimited term is wrapped as a quoted string (which
    disables FTS operator parsing), so a query containing ``"``, ``-``,
    ``NEAR``, ``*`` etc. can never raise ``sqlite3.OperationalError``.
    """
    terms = [t.replace('"', '""') for t in query.split() if t.strip()]
    return " ".join(f'"{t}"' for t in terms)


class MemoryVault:
    def __init__(
        self,
        db_path: str = DB_PATH,
        chroma_path: str = CHROMA_PATH,
        warm=None,
        cold=None,
    ):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.warm = warm
        self.cold = cold
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._collection = None
        self._embed_fn = None
        self._chroma_ready = False
        self._init_sqlite()

    # ------------------------------------------------------------------ #
    # storage init
    # ------------------------------------------------------------------ #
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_sqlite(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    evidence_url TEXT,
                    evidence_verified INTEGER DEFAULT 0,
                    confidence TEXT DEFAULT 'hypothesis',
                    corroborations INTEGER DEFAULT 0,
                    needs_review INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    cold_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags, scope, content='memories', content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts (rowid, content, tags, scope)
                    VALUES (new.id, new.content, new.tags, new.scope);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts (memories_fts, rowid, content, tags, scope)
                    VALUES ('delete', old.id, old.content, old.tags, old.scope);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts (memories_fts, rowid, content, tags, scope)
                    VALUES ('delete', old.id, old.content, old.tags, old.scope);
                    INSERT INTO memories_fts (rowid, content, tags, scope)
                    VALUES (new.id, new.content, new.tags, new.scope);
                END;
                """
            )
            # Backfill columns for vaults created by an earlier version.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
            for col, ddl in (
                ("evidence_verified", "INTEGER DEFAULT 0"),
                ("needs_review", "INTEGER DEFAULT 0"),
                ("archived", "INTEGER DEFAULT 0"),
                ("cold_key", "TEXT"),
            ):
                if col not in cols:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")

    @property
    def collection(self):
        """Lazily create the Chroma collection on first use.

        Import-time model download / index open is deferred so importing
        this module (and unit-testing the SQLite path) stays cheap.
        """
        if self._chroma_ready:
            return self._collection
        self._chroma_ready = True
        if not _CHROMA_AVAILABLE:
            log.info("chromadb not installed; semantic search disabled (keyword only)")
            self._collection = None
            return None
        try:
            os.makedirs(self.chroma_path, exist_ok=True)
            client = chromadb.PersistentClient(path=self.chroma_path)
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL
            )
            self._collection = client.get_or_create_collection(
                name="memories", embedding_function=self._embed_fn, metadata={"hnsw:space": "cosine"}
            )
            self._ensure_cosine_space(client)
        except Exception:  # pragma: no cover - environment dependent
            log.exception("failed to initialise ChromaDB; falling back to keyword search")
            self._collection = None
        return self._collection

    def _ensure_cosine_space(self, client):
        """Guard against a collection created by an older version with L2 space.

        Similarity scoring here assumes cosine distance (``sim = 1 - dist``);
        an L2 collection silently mis-scores every corroboration and cascade
        decision. If a mismatch is found, recreate the collection as cosine
        and re-embed from SQLite (the source of truth). Set
        ``AGENT_NEUTRAL_HARNESS_NO_CHROMA_MIGRATE=1`` to raise instead.
        """
        space = (getattr(self._collection, "metadata", None) or {}).get("hnsw:space", "cosine")
        if space == "cosine":
            return
        if os.environ.get("AGENT_NEUTRAL_HARNESS_NO_CHROMA_MIGRATE"):
            raise RuntimeError(
                f"ChromaDB collection 'memories' uses '{space}' distance, not cosine. "
                f"Delete {self.chroma_path} and run sync_embeddings(), or unset "
                f"AGENT_NEUTRAL_HARNESS_NO_CHROMA_MIGRATE to auto-migrate."
            )
        log.warning(
            "migrating ChromaDB collection from '%s' to cosine space (re-embedding from SQLite)",
            space,
        )
        client.delete_collection("memories")
        self._collection = client.get_or_create_collection(
            name="memories", embedding_function=self._embed_fn, metadata={"hnsw:space": "cosine"}
        )
        self._reembed_all()

    def _reembed_all(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, scope, type, content, confidence, evidence_verified FROM memories "
                "WHERE COALESCE(archived, 0) = 0"
            ).fetchall()
        if not rows:
            return
        self._collection.upsert(
            ids=[str(r["id"]) for r in rows],
            documents=[r["content"] for r in rows],
            metadatas=[
                {"scope": r["scope"], "type": r["type"], "confidence": r["confidence"],
                 "evidence_verified": bool(r["evidence_verified"])}
                for r in rows
            ],
        )

    def embed(self, text: str):
        """Embed one string with the same model Chroma uses, or ``None``."""
        if not self.collection or not self._embed_fn:
            return None
        return list(self._embed_fn([text])[0])

    # ------------------------------------------------------------------ #
    # evidence
    # ------------------------------------------------------------------ #
    def _verify_evidence_url(self, url: str) -> bool:
        """HEAD-check that ``url`` is reachable, re-validating every redirect hop.

        ``requests(..., allow_redirects=True)`` follows a redirect chain
        with no safety check on the *target* -- a URL that passes
        ``_url_is_safe`` can 302 to a private/internal host (e.g. a cloud
        metadata endpoint) and be followed anyway, defeating the SSRF
        guard entirely. Redirects are therefore walked manually, one hop
        at a time, and each target is re-checked with ``_url_is_safe``
        before it is ever requested.

        Known residual gap: ``_url_is_safe`` and the actual request each
        do their own DNS resolution, moments apart -- a DNS-rebinding
        attacker controlling the resolved name could in principle answer
        differently between the two lookups. Not closed here (would need
        pinning the checked IP and connecting to it directly); acceptable
        for this project's threat model (evidence URLs come from your own
        local/MCP clients, not the open internet), and noted in
        SECURITY.md as a documented limitation rather than silently
        ignored.
        """
        if not url or not _url_is_safe(url):
            return False
        current = url
        for _ in range(MAX_EVIDENCE_REDIRECTS):
            try:
                resp = requests.head(current, timeout=5, allow_redirects=False)
                if resp.status_code in (405, 501):  # HEAD unsupported; fall back to GET
                    resp = requests.get(current, timeout=5, allow_redirects=False, stream=True)
                    resp.close()
            except requests.RequestException:
                return False
            if 300 <= resp.status_code < 400 and resp.headers.get("Location"):
                next_url = urljoin(current, resp.headers["Location"])
                if not _url_is_safe(next_url):
                    return False
                current = next_url
                continue
            return resp.status_code < 400
        return False  # too many redirects -- treat as unverified, not an error

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search_keyword(self, query: str, limit: int = 5) -> str:
        match_expr = _sanitize_fts_query(query)
        if not match_expr:
            return _NO_KEYWORD_MATCH
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.scope, m.type, m.content, m.confidence, m.evidence_url
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ? AND COALESCE(m.archived, 0) = 0
                ORDER BY rank LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()
        if not rows:
            return _NO_KEYWORD_MATCH
        return "\n".join(
            f"[{r['scope']}] ({r['type']}) [Confidence: {r['confidence']}] {r['content']}"
            for r in rows
        )

    def search_semantic(self, query: str, top: int = 5) -> str:
        """Cascading semantic search: hot -> warm -> cold.

        Anything found in a colder tier is transparently restored into the
        hot tier before the result is returned.
        """
        if not self.collection:
            return self.search_keyword(query, top)

        results = self.collection.query(
            query_texts=[query], n_results=top,
            include=["documents", "metadatas", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        hot = [
            (d, m or {}, 1.0 - float(dist))
            for d, m, dist in zip(docs, metas, dists, strict=False)
            if 1.0 - float(dist) >= HOT_ACCEPT_THRESHOLD
        ]
        if hot:
            return "\n".join(
                f"[{m.get('scope', 'global')}/{m.get('type', 'fact')}, sim={sim:.2f}] {d}"
                for d, m, sim in hot
            )

        for tier, label, threshold in (
            (self.warm, "warm", WARM_ACCEPT_THRESHOLD),
            (self.cold, "cold", COLD_ACCEPT_THRESHOLD),
        ):
            if not tier or self.embed(query) is None:
                continue
            try:
                hits = tier.scan(query, self.embed, threshold=threshold)
            except Exception:  # pragma: no cover - network/tier dependent
                log.exception("%s tier scan failed", label)
                continue
            if hits:
                for row in hits:
                    self._restore(row)
                return "\n".join(
                    f"[{r.get('scope', 'global')}/{r.get('type', 'fact')}] {r['content']} "
                    f"(restored from {label} tier)"
                    for r in hits
                )
        return _NO_SEMANTIC_MATCH

    def _restore(self, row: dict):
        """Bring a row found in a colder tier back into the hot tier."""
        content = row["content"]
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM memories WHERE content = ?", (content,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memories SET archived = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (existing["id"],),
                )
                mem_id = existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO memories (scope, type, content, confidence) VALUES (?, ?, ?, ?)",
                    (row.get("scope", "global"), row.get("type", "fact"), content,
                     row.get("confidence", "hypothesis")),
                )
                mem_id = cur.lastrowid
        if self.collection:
            try:
                self.collection.upsert(
                    ids=[str(mem_id)], documents=[content],
                    metadatas=[{"scope": row.get("scope", "global"),
                               "type": row.get("type", "fact")}],
                )
            except Exception:  # pragma: no cover
                log.exception("failed to re-embed restored memory #%s", mem_id)

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    def _most_similar_active(self, content: str):
        """``(id, similarity)`` of the single most similar non-archived
        memory, or ``None`` when there is no semantic tier / no rows."""
        if not self.collection:
            return None
        try:
            res = self.collection.query(
                query_texts=[content], n_results=1, include=["distances"]
            )
        except Exception:  # pragma: no cover - environment dependent
            log.exception("similarity query failed")
            return None
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not ids:
            return None
        return int(ids[0]), 1.0 - float(dists[0])

    def save_memory(
        self,
        scope: str,
        mem_type: str,
        content: str,
        tags: str = "",
        evidence_url: str = "",
    ) -> str:
        if not content or not content.strip():
            return "Refused: empty memory content."

        evidence_verified = self._verify_evidence_url(evidence_url)
        match = self._most_similar_active(content)

        if match and match[1] >= CORROBORATION_THRESHOLD:
            return self._corroborate(match, content, scope, mem_type, evidence_url, evidence_verified)

        needs_review = bool(match and REVIEW_BAND_LOW <= match[1] < REVIEW_BAND_HIGH)
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories
                    (scope, type, content, tags, evidence_url, evidence_verified,
                     confidence, corroborations, needs_review)
                VALUES (?, ?, ?, ?, ?, ?, 'hypothesis', 1, ?)
                """,
                (scope, mem_type, content, tags, evidence_url or None,
                 int(evidence_verified), int(needs_review)),
            )
            mem_id = cur.lastrowid

        self._embed(mem_id, content, scope, mem_type, "hypothesis", evidence_verified)

        ev = " (evidence verified)" if evidence_verified else \
             " (no verified evidence -- capped at 'suspected')"
        if needs_review:
            return (
                f"Saved memory #{mem_id} as hypothesis, flagged needs_review=1 "
                f"(similar to #{match[0]}, sim={match[1]:.2f} -- unclear if it agrees or conflicts)."
            )
        return f"Saved memory #{mem_id} [Confidence: hypothesis]{ev}"

    def _corroborate(self, match, content, scope, mem_type, evidence_url, evidence_verified) -> str:
        existing_id, sim = match
        with self._conn() as conn:
            row = conn.execute(
                "SELECT corroborations, evidence_verified FROM memories WHERE id = ?",
                (existing_id,),
            ).fetchone()
            if row is None:  # chroma/sqlite drift -- fall back to a plain insert
                cur = conn.execute(
                    "INSERT INTO memories (scope, type, content, evidence_url, evidence_verified, "
                    "confidence, corroborations) VALUES (?, ?, ?, ?, ?, 'hypothesis', 1)",
                    (scope, mem_type, content, evidence_url or None, int(evidence_verified)),
                )
                new_id = cur.lastrowid
                self._embed(new_id, content, scope, mem_type, "hypothesis", evidence_verified)
                return f"Saved memory #{new_id} [Confidence: hypothesis] (stale index; inserted fresh)"

            new_count = (row["corroborations"] or 1) + 1
            has_evidence = bool(evidence_verified or row["evidence_verified"])
            if has_evidence:
                confidence = "confirmed" if new_count >= CONFIRM_MIN_CORROBORATIONS else "suspected"
            else:
                confidence = "suspected"  # capped, permanently, without evidence

            attach_evidence = evidence_verified and not row["evidence_verified"]
            conn.execute(
                "UPDATE memories SET corroborations = ?, confidence = ?, "
                "updated_at = CURRENT_TIMESTAMP"
                + (", evidence_url = ?, evidence_verified = 1" if attach_evidence else "")
                + " WHERE id = ?",
                ([new_count, confidence, evidence_url, existing_id] if attach_evidence
                 else [new_count, confidence, existing_id]),
            )

        if self.collection:
            try:
                self.collection.upsert(
                    ids=[str(existing_id)], documents=[content],
                    metadatas=[{"scope": scope, "type": mem_type, "confidence": confidence}],
                )
            except Exception:  # pragma: no cover
                log.exception("failed to update embedding for corroborated memory #%s", existing_id)

        ev = " (evidence-backed)" if has_evidence else " (no verified evidence -- capped at suspected)"
        return (
            f"Corroborated memory #{existing_id} (sim={sim:.2f}) -- "
            f"confidence={confidence}, corroborations={new_count}{ev}"
        )

    def _embed(self, mem_id, content, scope, mem_type, confidence, evidence_verified):
        if not self.collection:
            return
        try:
            self.collection.upsert(
                ids=[str(mem_id)], documents=[content],
                metadatas=[{
                    "scope": scope, "type": mem_type,
                    "confidence": confidence, "evidence_verified": bool(evidence_verified),
                }],
            )
        except Exception:  # pragma: no cover - environment dependent
            log.exception("failed to embed memory #%s", mem_id)

    def list_needs_review(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, scope, type, content, corroborations FROM memories "
                "WHERE needs_review = 1 AND COALESCE(archived, 0) = 0 ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_review(self, mem_id: int) -> str:
        with self._conn() as conn:
            conn.execute("UPDATE memories SET needs_review = 0 WHERE id = ?", (mem_id,))
        return f"Cleared needs_review on memory #{mem_id}."

    def sync_embeddings(self) -> str:
        """Re-index every non-archived SQLite memory row into ChromaDB."""
        if not self.collection:
            return "ChromaDB not available (install the 'memory' extra)."
        with self._conn() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE COALESCE(archived, 0) = 0"
            ).fetchone()[0]
        if not n:
            return "No active rows to sync."
        self._reembed_all()
        return f"Synced {n} active memories into ChromaDB."

    # exposed for the cold tier
    def _rows_for_archive(self, cutoff_epoch: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE COALESCE(archived, 0) = 0 AND CAST(COALESCE("
                "strftime('%s', updated_at), strftime('%s', created_at), '0') AS INTEGER) < ?",
                (int(cutoff_epoch),),
            ).fetchall()
        return [dict(r) for r in rows]

    def _mark_archived(self, mem_id: int, cold_key: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET archived = 1, cold_key = ? WHERE id = ?", (cold_key, mem_id)
            )
        if self.collection:
            try:
                self.collection.delete(ids=[str(mem_id)])
            except Exception:  # pragma: no cover
                log.exception("failed to drop archived memory #%s from ChromaDB", mem_id)

    def _serialize_row(self, row: dict) -> bytes:
        return json.dumps(row, default=str).encode("utf-8")
