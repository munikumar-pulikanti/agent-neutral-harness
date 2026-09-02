"""Persistent memory vault with evidence-gated confidence scoring.

What is implemented here (the "hot" tier):
- Local SQLite + FTS5 for keyword search.
- ChromaDB (``all-MiniLM-L6-v2``) for semantic vector search, used when
  the optional ``memory`` extra is installed; the vault degrades to
  keyword-only search when it is not.
- Evidence-gated confidence: a new memory is ``hypothesis`` until it is
  corroborated by an existing similar memory, and only reaches
  ``confirmed`` when corroboration is backed by a *verified* evidence URL
  (checked with a real HTTP request, with SSRF protection).

Warm (Turso) and cold (object-store) tiers are a design goal, not part of
this module yet -- see HANDOFF.md.
"""

import ipaddress
import logging
import os
import socket
import sqlite3
from contextlib import contextmanager
from urllib.parse import urlparse

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

# Corroboration / confidence tuning.
CORROBORATION_MIN_SIMILARITY = 0.65   # cosine similarity for a semantic match
CONFIRM_MIN_CORROBORATIONS = 1        # corroborating neighbours needed for "confirmed"

VALID_CONFIDENCE = ("hypothesis", "suspected", "confirmed")

_NO_KEYWORD_MATCH = "No matching memories found (checked keyword FTS)."
_NO_SEMANTIC_MATCH = "No semantic matches found (hot tier)."
NO_RESULT_MARKERS = (_NO_KEYWORD_MATCH, _NO_SEMANTIC_MATCH)


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
    def __init__(self, db_path: str = DB_PATH, chroma_path: str = CHROMA_PATH):
        self.db_path = db_path
        self.chroma_path = chroma_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._collection = None
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
            if "evidence_verified" not in cols:
                conn.execute("ALTER TABLE memories ADD COLUMN evidence_verified INTEGER DEFAULT 0")

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
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            self._collection = client.get_or_create_collection(
                name="memories", embedding_function=ef, metadata={"hnsw:space": "cosine"}
            )
        except Exception:  # pragma: no cover - environment dependent
            log.exception("failed to initialise ChromaDB; falling back to keyword search")
            self._collection = None
        return self._collection

    # ------------------------------------------------------------------ #
    # evidence
    # ------------------------------------------------------------------ #
    def _verify_evidence_url(self, url: str) -> bool:
        if not url or not _url_is_safe(url):
            return False
        try:
            resp = requests.head(url, timeout=5, allow_redirects=True)
            return resp.status_code < 400
        except requests.RequestException:
            return False

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
                WHERE memories_fts MATCH ?
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
        if not self.collection:
            return self.search_keyword(query, top)
        results = self.collection.query(query_texts=[query], n_results=top)
        docs = (results.get("documents") or [[]])[0]
        if not docs:
            return _NO_SEMANTIC_MATCH
        metas = (results.get("metadatas") or [[{}] * len(docs)])[0]
        out = []
        for doc, meta in zip(docs, metas, strict=False):
            meta = meta or {}
            conf = meta.get("confidence", "hypothesis")
            scope = meta.get("scope", "global")
            out.append(f"[{scope}] [Confidence: {conf}] {doc}")
        return "\n".join(out)

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    def _find_corroborators(self, content: str, top: int = 5) -> list[str]:
        """IDs of existing memories similar enough to corroborate ``content``."""
        if not self.collection:
            return []
        try:
            res = self.collection.query(
                query_texts=[content], n_results=top, include=["distances"]
            )
        except Exception:  # pragma: no cover - environment dependent
            log.exception("corroboration query failed")
            return []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits = []
        for mem_id, dist in zip(ids, dists, strict=False):
            similarity = 1.0 - float(dist)  # cosine space
            if similarity >= CORROBORATION_MIN_SIMILARITY:
                hits.append(mem_id)
        return hits

    def save_memory(
        self,
        scope: str,
        mem_type: str,
        content: str,
        tags: str = "",
        evidence_url: str = "",
    ) -> str:
        """Persist a memory, scoring its confidence from corroboration + evidence.

        - ``confirmed``  : corroborated by >= CONFIRM_MIN_CORROBORATIONS
                           existing memories AND backed by a verified
                           evidence URL.
        - ``suspected``  : corroborated but without verified evidence.
        - ``hypothesis`` : no corroboration (the default for genuinely new
                           information).
        """
        if not content or not content.strip():
            return "Refused: empty memory content."

        has_evidence = self._verify_evidence_url(evidence_url)
        corroborators = self._find_corroborators(content)
        n = len(corroborators)

        if n >= CONFIRM_MIN_CORROBORATIONS and has_evidence:
            confidence = "confirmed"
        elif n >= 1:
            confidence = "suspected"
        else:
            confidence = "hypothesis"

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories
                    (scope, type, content, tags, evidence_url, evidence_verified,
                     confidence, corroborations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (scope, mem_type, content, tags, evidence_url, int(has_evidence), confidence, n),
            )
            mem_id = cur.lastrowid
            if corroborators:
                conn.executemany(
                    "UPDATE memories SET corroborations = corroborations + 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    [(cid,) for cid in corroborators],
                )

        if self.collection:
            try:
                self.collection.upsert(
                    ids=[str(mem_id)],
                    documents=[content],
                    metadatas=[{
                        "scope": scope,
                        "type": mem_type,
                        "confidence": confidence,
                        "evidence_verified": has_evidence,
                    }],
                )
            except Exception:  # pragma: no cover - environment dependent
                log.exception("failed to embed memory #%s", mem_id)

        note = "" if not corroborators else f", {n} corroborator(s)"
        return f"Saved memory #{mem_id} [Confidence: {confidence}{note}]"

    def sync_embeddings(self) -> str:
        """Re-index every SQLite memory row into ChromaDB."""
        if not self.collection:
            return "ChromaDB not available (install the 'memory' extra)."
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, scope, type, content, confidence, evidence_verified FROM memories"
            ).fetchall()
        if not rows:
            return "No rows to sync."
        self.collection.upsert(
            ids=[str(r["id"]) for r in rows],
            documents=[r["content"] for r in rows],
            metadatas=[
                {
                    "scope": r["scope"],
                    "type": r["type"],
                    "confidence": r["confidence"],
                    "evidence_verified": bool(r["evidence_verified"]),
                }
                for r in rows
            ],
        )
        return f"Synced {len(rows)} memories into ChromaDB."
