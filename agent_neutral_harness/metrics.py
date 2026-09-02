"""Usage metrics for agent-neutral-harness.

Tracks routing decisions, model usage, cascade cost data, and eval
baselines/results in a local SQLite database. Model names are always
passed in as plain parameters, never hardcoded, so this stays usable
regardless of which local or cloud models a deployment actually uses.

The database location resolves in this order:
  1. ``$AGENT_NEUTRAL_HARNESS_METRICS_DB`` (full path to the .db file)
  2. ``$AGENT_NEUTRAL_HARNESS_HOME/metrics.db``
  3. ``~/.agent-neutral-harness/metrics.db`` (default)
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


def _default_home() -> Path:
    return Path(os.environ.get("AGENT_NEUTRAL_HARNESS_HOME", Path.home() / ".agent-neutral-harness"))


def metrics_db_path() -> Path:
    explicit = os.environ.get("AGENT_NEUTRAL_HARNESS_METRICS_DB")
    if explicit:
        return Path(explicit)
    return _default_home() / "metrics.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    task_snippet TEXT,
    category TEXT,
    model TEXT,
    duration_seconds REAL,
    tool_call_count INTEGER DEFAULT 0,
    memory_pre_hit INTEGER DEFAULT 0,
    memory_tier TEXT DEFAULT 'none',
    error_occurred INTEGER DEFAULT 0,
    assertion_flags TEXT DEFAULT '',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cascade_tier TEXT DEFAULT '',
    escalated INTEGER DEFAULT 0,
    cheap_attempt_tokens INTEGER,
    capable_attempt_tokens INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_category ON turns (category, id DESC);

CREATE TABLE IF NOT EXISTS eval_baselines (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    task TEXT,
    expected_summary TEXT,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    baseline_id INTEGER,
    passed INTEGER,
    judge_reasoning TEXT,
    actual_summary TEXT
);
"""

ESCALATION_RATE_WINDOW = 50  # most recent qualifying turns per category, not all-time

_initialized: set = set()


@contextmanager
def _connect():
    path = metrics_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    key = str(path)
    if key not in _initialized:
        conn.executescript(SCHEMA)
        _initialized.add(key)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the schema if it does not yet exist (idempotent)."""
    with _connect():
        pass


def log_turn(task_snippet, category, model, duration_seconds, tool_call_count=0,
             memory_pre_hit=False, memory_tier="none", error_occurred=False,
             assertion_flags="", input_tokens=0, output_tokens=0,
             cascade_tier="", escalated=False, cheap_attempt_tokens=0,
             capable_attempt_tokens=0):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO turns (task_snippet, category, model, duration_seconds, "
            "tool_call_count, memory_pre_hit, memory_tier, error_occurred, assertion_flags, "
            "input_tokens, output_tokens, cascade_tier, escalated, cheap_attempt_tokens, "
            "capable_attempt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ((task_snippet or "")[:200], category, model, duration_seconds, tool_call_count,
             int(memory_pre_hit), memory_tier, int(error_occurred), assertion_flags,
             input_tokens, output_tokens, cascade_tier, int(escalated),
             cheap_attempt_tokens, capable_attempt_tokens),
        )


def all_turns():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM turns ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def category_escalation_rate(category: str) -> dict:
    """Rolling recent-window escalation rate for a category.

    Deliberately NOT an all-time average (that dilutes fresh evidence as
    history grows) and deliberately excludes shortcut-skipped turns
    (``cheap_attempt_tokens IS NULL``). Those turns are logged as
    ``escalated=True`` by construction -- not because the cheap tier was
    tried and failed -- so counting them would make the rate
    self-reinforcing: the shortcut fires, produces more fake-escalated
    rows, and the rate stays pinned high forever even if the underlying
    cheap model improves.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT escalated FROM turns "
            "WHERE category = ? AND cheap_attempt_tokens IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (category, ESCALATION_RATE_WINDOW),
        ).fetchall()
    if not rows:
        return {"sample_size": 0, "escalation_rate": None}
    escalated_count = sum(1 for r in rows if r["escalated"])
    return {"sample_size": len(rows), "escalation_rate": escalated_count / len(rows)}


def save_eval_baseline(task: str, expected_summary: str, git_commit: str = "") -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO eval_baselines (task, expected_summary, git_commit) VALUES (?, ?, ?)",
            (task, expected_summary, git_commit),
        )
        return cursor.lastrowid


def all_baselines():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM eval_baselines ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def log_eval_result(baseline_id: int, passed: bool, judge_reasoning: str, actual_summary: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO eval_results (baseline_id, passed, judge_reasoning, actual_summary) "
            "VALUES (?, ?, ?, ?)",
            (baseline_id, int(passed), judge_reasoning, actual_summary),
        )


def all_eval_results():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM eval_results ORDER BY id").fetchall()
    return [dict(r) for r in rows]
