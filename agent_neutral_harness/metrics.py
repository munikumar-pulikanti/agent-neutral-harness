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
    capable_attempt_tokens INTEGER DEFAULT 0,
    config_fingerprint TEXT DEFAULT ''
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
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(turns)")}
        if "config_fingerprint" not in cols:  # migrate a v0.1 database
            conn.execute("ALTER TABLE turns ADD COLUMN config_fingerprint TEXT DEFAULT ''")
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
             capable_attempt_tokens=0, config_fingerprint=""):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO turns (task_snippet, category, model, duration_seconds, "
            "tool_call_count, memory_pre_hit, memory_tier, error_occurred, assertion_flags, "
            "input_tokens, output_tokens, cascade_tier, escalated, cheap_attempt_tokens, "
            "capable_attempt_tokens, config_fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ((task_snippet or "")[:200], category, model, duration_seconds, tool_call_count,
             int(memory_pre_hit), memory_tier, int(error_occurred), assertion_flags,
             input_tokens, output_tokens, cascade_tier, int(escalated),
             cheap_attempt_tokens, capable_attempt_tokens, config_fingerprint or ""),
        )


def all_turns():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM turns ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score interval lower bound for a proportion.

    Using this instead of the raw fraction means a small sample can't
    thrash a threshold decision by chance alone: a true rate of ~65% can
    easily show 80%+ in 20 samples. The lower bound asks "am I actually
    confident the true rate clears the threshold", not "did this one
    sample clear it".
    """
    if n == 0:
        return 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (center - margin) / denom)


def _window(conn, category: str, config_fingerprint):
    """Most-recent qualifying turns for a category (list of sqlite Rows).

    Excludes shortcut-skipped turns (``cheap_attempt_tokens IS NULL``):
    those are logged ``escalated=True`` by construction, so counting them
    would make the rate self-reinforcing. Filters by ``config_fingerprint``
    when one is supplied, so a model/prompt/tool-schema change naturally
    invalidates stale history instead of dragging the decision for a
    whole window.
    """
    if config_fingerprint:
        sql = (
            "SELECT escalated FROM turns WHERE category = ? "
            "AND cheap_attempt_tokens IS NOT NULL AND config_fingerprint = ? "
            "ORDER BY id DESC LIMIT ?"
        )
        params = (category, config_fingerprint, ESCALATION_RATE_WINDOW)
    else:
        sql = (
            "SELECT escalated FROM turns WHERE category = ? "
            "AND cheap_attempt_tokens IS NOT NULL "
            "ORDER BY id DESC LIMIT ?"
        )
        params = (category, ESCALATION_RATE_WINDOW)
    return conn.execute(sql, params).fetchall()


def category_escalation_rate(category: str, config_fingerprint: str = "") -> dict:
    """Rolling recent-window escalation rate for a category.

    Returns ``sample_size``, ``escalation_rate`` (raw fraction) and
    ``escalation_rate_lower_bound`` (95% Wilson lower bound — the value a
    threshold decision should compare against).
    """
    with _connect() as conn:
        rows = _window(conn, category, config_fingerprint)
    if not rows:
        return {"sample_size": 0, "escalation_rate": None, "escalation_rate_lower_bound": None}
    escalated_count = sum(1 for r in rows if r["escalated"])
    n = len(rows)
    return {
        "sample_size": n,
        "escalation_rate": escalated_count / n,
        "escalation_rate_lower_bound": _wilson_lower_bound(escalated_count, n),
    }


def detect_within_window_drift(
    category: str,
    config_fingerprint: str = "",
    min_half_size: int = 10,
    drift_threshold: float = 0.3,
) -> dict:
    """Split the escalation-rate window into newer/older halves and flag a jump.

    Catches a behaviour change that leaves the config fingerprint
    unchanged (e.g. a meaningful tool-description rewrite): old and new
    behaviour accumulate in the same window, and the newer half shows a
    measurable rise in escalation rate. This is a **flag only** — it
    never changes routing.
    """
    with _connect() as conn:
        rows = _window(conn, category, config_fingerprint)
    total = len(rows)
    half = total // 2
    if half < min_half_size:
        return {"drift_detected": False, "reason": "insufficient_data",
                "newer_half_size": half, "older_half_size": total - half}
    newer, older = rows[:half], rows[half:2 * half]
    newer_rate = sum(1 for r in newer if r["escalated"]) / len(newer)
    older_rate = sum(1 for r in older if r["escalated"]) / len(older)
    drift = newer_rate - older_rate
    return {
        "drift_detected": drift >= drift_threshold,
        "reason": "escalation_rate_increased" if drift >= drift_threshold else "stable",
        "newer_half_rate": newer_rate,
        "older_half_rate": older_rate,
        "drift_magnitude": drift,
        "newer_half_size": len(newer),
        "older_half_size": len(older),
    }


# --------------------------------------------------------------------- #
# aggregates for the dashboard
# --------------------------------------------------------------------- #
def _count_by(field: str) -> dict:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {field} AS k, COUNT(*) AS n FROM turns GROUP BY {field} ORDER BY n DESC"
        ).fetchall()
    return {(r["k"] or "unknown"): r["n"] for r in rows}


def category_breakdown() -> dict:
    return _count_by("category")


def model_breakdown() -> dict:
    return _count_by("model")


def memory_tier_breakdown() -> dict:
    base = {"hot": 0, "warm": 0, "cold": 0, "none": 0}
    for tier, n in _count_by("memory_tier").items():
        base[tier if tier in base else "none"] += n
    return base


# --------------------------------------------------------------------- #
# eval baselines / results
# --------------------------------------------------------------------- #
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
        rows = conn.execute(
            "SELECT r.*, b.task FROM eval_results r "
            "LEFT JOIN eval_baselines b ON b.id = r.baseline_id ORDER BY r.id"
        ).fetchall()
    return [dict(r) for r in rows]
