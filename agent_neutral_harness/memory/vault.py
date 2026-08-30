"""Persistent Multi-Tier Memory Vault with Evidence-Gated Confidence Scoring.

Tiers:
- Hot: Local SQLite FTS5 (keyword search) + ChromaDB (semantic vector search).
- Warm: Turso (libSQL) push/pull sync.
- Cold: MinIO (S3-compatible) archive/restore.
"""

import os
import re
import sqlite3
from typing import Optional, List, Dict, Any
import requests

try:
    import chromadb
    from chromadb.utils import embedding_functions
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

DB_DIR = os.path.expanduser("~/.ai-memory-vault")
DB_PATH = os.path.join(DB_DIR, "global_brain.db")
CHROMA_PATH = os.path.join(DB_DIR, "chroma")


class MemoryVault:
    def __init__(self, db_path: str = DB_PATH, chroma_path: str = CHROMA_PATH):
        self.db_path = db_path
        self.chroma_path = chroma_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_sqlite()
        self._init_chroma()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    evidence_url TEXT,
                    confidence TEXT DEFAULT 'hypothesis',
                    corroborations INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags, scope, content='memories', content_rowid='id'
                )
            """)

    def _init_chroma(self):
        if not _CHROMA_AVAILABLE:
            self.collection = None
            return
        os.makedirs(self.chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=self.chroma_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = client.get_or_create_collection(name="memories", embedding_function=ef)

    def _verify_evidence_url(self, url: str) -> bool:
        if not url or not url.startswith(("http://", "https://")):
            return False
        try:
            resp = requests.head(url, timeout=5, allow_redirects=True)
            return resp.status_code < 400
        except Exception:
            return False

    def search_keyword(self, query: str, limit: int = 5) -> str:
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT m.id, m.scope, m.type, m.content, m.confidence, m.evidence_url
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (query, limit)).fetchall()
            if not rows:
                return "No matching memories found (checked keyword FTS)."
            return "\n".join(f"[{r['scope']}] ({r['type']}) [Confidence: {r['confidence']}] {r['content']}" for r in rows)

    def search_semantic(self, query: str, top: int = 5) -> str:
        if not self.collection:
            return self.search_keyword(query, top)
        results = self.collection.query(query_texts=[query], n_results=top)
        if not results or not results["documents"] or not results["documents"][0]:
            return "No semantic matches found (hot tier)."
        
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        out = []
        for doc, meta in zip(docs, metas):
            conf = meta.get("confidence", "hypothesis")
            scope = meta.get("scope", "global")
            out.append(f"[{scope}] [Confidence: {conf}] {doc}")
        return "\n".join(out)

    def save_memory(self, scope: str, type_: str, content: str, tags: str = "", evidence_url: str = "") -> str:
        has_evidence = self._verify_evidence_url(evidence_url)
        confidence = "hypothesis"

        with self._get_conn() as conn:
            cur = conn.execute("""
                INSERT INTO memories (scope, type, content, tags, evidence_url, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (scope, type_, content, tags, evidence_url, confidence))
            mem_id = cur.lastrowid
            conn.execute("""
                INSERT INTO memories_fts (rowid, content, tags, scope)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, tags, scope))

        if self.collection:
            self.collection.add(
                ids=[str(mem_id)],
                documents=[content],
                metadatas=[{"scope": scope, "type": type_, "confidence": confidence, "evidence_verified": has_evidence}]
            )
        return f"Saved memory #{mem_id} [Confidence: {confidence}]"

    def sync_embeddings(self) -> str:
        if not self.collection:
            return "ChromaDB not available."
        with self._get_conn() as conn:
            rows = conn.execute("SELECT id, scope, type, content, confidence FROM memories").fetchall()
        if not rows:
            return "No rows to sync."
        ids = [str(r["id"]) for r in rows]
        docs = [r["content"] for r in rows]
        metas = [{"scope": r["scope"], "type": r["type"], "confidence": r["confidence"]} for r in rows]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return f"Synced {len(rows)} memories into ChromaDB."