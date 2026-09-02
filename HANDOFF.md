# agent-neutral-harness — Project State & Handoff

Complete brief for continuing development with a tool/agent that has no prior
context. Read this fully before making changes.

## What this is

A reliability + memory + model-routing harness that is **agnostic to the agent
runtime**. It is the framework-neutral distillation of
[`hearthagent-pro`](https://github.com/munikumar-pulikanti/hearthagent-pro) (a
local-first coding agent): same ideas — cheap→capable cascade, deterministic
reliability assertions, multi-tier memory with evidence-gated confidence,
golden-baseline evals — but with every model name and every execution runtime
passed in as a parameter instead of hardcoded to Ollama/LangGraph.

The thesis: the *routing and reliability logic* is the same whether you drive
LangGraph, a raw tool loop, or an IDE assistant. Only `execute_fn(model, task)`
changes. So that is the one thing the caller supplies.

## Repo layout

```
agent_neutral_harness/
  __init__.py            -- public API (lazy re-exports)
  cascade.py             -- run_cascade: cheap-first, verify-real-output, escalate-on-observed-failure
  metrics.py             -- SQLite per-turn log; escalation rate (Wilson-bounded, fingerprint-scoped); drift; breakdowns
  fingerprint.py         -- config_fingerprint(), ollama_model_digest()
  dashboard.py           -- Streamlit metrics view ([dashboard]; console script: agent-neutral-dashboard)
  reliability/
    assertions.py        -- deterministic checks + run_all_checks() aggregator
  routing/
    classifier.py        -- classify_task() -> one of 5 categories; injectable generate_fn
    curator.py           -- curate_memories() -> relevance filter before prompt injection
  memory/
    vault.py             -- MemoryVault: SQLite/FTS5 + optional ChromaDB + confidence lifecycle + hot->warm->cold cascade
    warm.py              -- TursoWarmTier ([warm])
    cold.py              -- ObjectStoreColdTier ([cold])
    _similarity.py       -- pure-python cosine (no numpy in the warm/cold path)
    mcp_server.py        -- FastMCP server ([mcp]; console script: agent-neutral-memory)
  evals/
    runner.py            -- run_eval_judge() (one answer) + run_regression_suite() (all baselines)
tests/                   -- pytest; fast suite has no network / no model calls (fns are stubbed).
                            tests/test_semantic_integration.py is @pytest.mark.semantic (real ChromaDB)
.github/workflows/ci.yml -- fast job (ruff + pytest, 3.11-3.13) + semantic job ([memory], -m semantic)
```

## Design decisions worth knowing

1. **Escalation is observed, never predicted.** `run_cascade` runs the cheap
   model, runs `run_all_checks` on its *actual* output, and only escalates if a
   check fails. There is no confidence threshold on the model's self-report.

2. **The shortcut is self-correcting.** Once a category's escalation rate over the
   last ≥20 qualifying turns is ≥80%, the cascade skips the cheap tier ~80% of the
   time. The other ~20% still runs the cheap tier so the rate can fall again if the
   cheap model improves. Shortcut-skipped turns are logged with
   `cheap_attempt_tokens = NULL` and **excluded** from `category_escalation_rate`
   so the signal can't feed on itself.

3. **`execute_fn` exceptions are contained.** `_safe_execute` turns any raised
   exception into an `error` result so one bad runtime call can't crash the
   cascade or skip metrics logging.

4. **Confidence tiers are evidence-gated.** New memory = `hypothesis`. Corroborated
   by a semantically-similar existing memory = `suspected`. Corroborated **and**
   backed by a verified, reachable evidence URL = `confirmed`. URL verification is
   SSRF-guarded (rejects private/loopback/link-local/reserved IPs).

5. **Optional deps are actually optional.** Core install is just `requests`.
   ChromaDB (`[memory]`) and `mcp` (`[mcp]`) are extras. The vault degrades to
   keyword-only search when ChromaDB is absent; importing the package never pulls
   an extra or triggers a model download (lazy `__getattr__` + lazy Chroma init).

6. **Routing is provider-neutral by injection.** `classify_task`, `curate_memories`,
   `run_eval_judge` default to a local Ollama call but accept a `generate_fn`
   callable. That is also how the tests avoid the network.

7. **The shortcut decision is defended twice.** (a) It compares the threshold
   against the Wilson score lower bound of the escalation rate, not the raw
   fraction, so a lucky small sample can't flip it. (b) It's scoped to a
   `config_fingerprint` (model digest + system prompt + structural tool signature,
   cwd normalised out) so any behaviour-affecting change drops stale rows from the
   window. Tool *description prose* is deliberately not hashed (it caused phantom
   invalidations on typo fixes); a meaningful description rewrite that keeps the
   signature is instead caught by `metrics.detect_within_window_drift`.

## What changed porting from hearthagent-pro (2026-09-02, second pass)

`hearthagent-pro` turned out to have working versions of most of the v0.1 open
items. Ported and adapted to the agent-neutral architecture (injectable fns,
optional-dep extras, unit-tested with fakes rather than live services):

- **Corroboration re-promotion** (`vault.save_memory`): a restated claim now
  UPDATEs the matched memory (bump count, promote confidence) instead of
  inserting a duplicate. `confirmed` needs `CONFIRM_MIN_CORROBORATIONS` (3) + a
  verified evidence URL; without evidence a memory is permanently capped at
  `suspected`. A "review band" (similarity in `[0.5, 0.85)`) inserts with
  `needs_review = 1` (`vault.list_needs_review()` / `resolve_review()`).
- **`fingerprint.py`** — `config_fingerprint()` + `ollama_model_digest()`.
- **`metrics`** — `config_fingerprint` column + filtering, `_wilson_lower_bound`,
  `detect_within_window_drift`, `category_breakdown` / `model_breakdown` /
  `memory_tier_breakdown`.
- **`memory/warm.py`** — `TursoWarmTier` push/pull (id-based, insert-only) + a
  cascade `scan()`. `[warm]` extra (`libsql-experimental`).
- **`memory/cold.py`** — `ObjectStoreColdTier` archive/restore/scan against any
  S3-compatible store; injectable `client` for testing. `[cold]` extra (`boto3`).
- **hot → warm → cold search cascade** in `vault.search_semantic`, restoring
  colder hits into hot.
- **`dashboard.py`** — Streamlit, `[dashboard]` extra, `agent-neutral-dashboard`
  console script.
- **CI** — added a second `semantic` job (`-m semantic`, `[memory]` extra, HF
  cache) so the real ChromaDB path is covered; the fast job runs `-m 'not
  semantic'`.

Not ported: `hearthagent-pro`'s LangGraph `graph.py` wiring, `run_shell`
allowlist/injection guard, `ddgs` web search, voice input, Flask UI — all
app-scope, not harness-scope.

## What changed in the pre-release pass (2026-09-02, first pass)

Fixed:
- `check_tool_result_fidelity` corrupted filenames starting with `f`/`d`
  (`lstrip("fd")`) and flagged version numbers (`2.31.0`) as fabricated files.
- `run_eval_judge` returned a raw JSON string on success but a dict on error;
  now always `{"passed": bool, "reasoning": str}`.
- `save_memory` never actually scored confidence (hardcoded `hypothesis`,
  `_verify_evidence_url` result unused). Now implements the tiered logic.
- `search_keyword` raised `OperationalError` on FTS5 metacharacters in the query.
- `MemoryVault()` did a model download at import time.
- No SSRF protection on evidence-URL verification.
- FTS index had no delete/update sync (now trigger-based).

Added:
- `tests/` (38 tests), GitHub Actions CI, `LICENSE` (MIT), `py.typed`.
- `ruff` config; `pyproject.toml` metadata, `[project.scripts]`, optional-dep
  extras. Dropped deps that nothing imported: fastapi, uvicorn, pydantic, pyyaml,
  langgraph, langchain-* (boto3 later returned as the `[cold]` extra).
- `logging` throughout (was silent `except: pass`).
- `run_regression_suite`, `metrics.all_eval_results`, configurable state dirs.
- Public API surface in `agent_neutral_harness/__init__.py`.

## Third pass (2026-09-02) -- the last three gaps

- **Warm-tier multi-writer safety.** Sync is now by **content hash**
  (`content_sync_key(scope, type, content)` = `sha256(...)[:24]`), not local row
  id, so two machines that both use local id 1 for different content no longer
  collide or lose data. `pull()` runs every incoming row through
  `vault.save_memory()`, so a divergent near-duplicate surfaces as
  corroboration or `needs_review` -- the review queue is the conflict log.
  Remote schema is keyed on `sync_key` and records an `origin`.
- **Cosine-space migration guard.** `MemoryVault._ensure_cosine_space()` detects a
  Chroma collection created with L2 space, and by default recreates it as cosine
  and re-embeds from SQLite (the source of truth), logging a warning. Set
  `AGENT_NEUTRAL_HARNESS_NO_CHROMA_MIGRATE=1` to raise with instructions instead.
- **Integration tests vs live backends.** `tests/test_integration_backends.py`
  (`@pytest.mark.integration`) runs the real `ObjectStoreColdTier` against MinIO
  and the real `TursoWarmTier` push/pull against a libSQL server. CI has an
  `integration` job with `bitnami/minio` + `libsql-server` service containers.
  Locally: set `MINIO_ENDPOINT` / `LIBSQL_URL` (see the module docstring). Verified
  green against real containers on 2026-09-02.

Test lanes now: `pytest` (fast), `-m semantic` (real ChromaDB), `-m integration`
(real MinIO + libSQL). 65 tests total.

## Known gaps / open items

1. **`classify_task` loose-matches** the category name as a token in the model's
   reply. A verbose reply mentioning two category words returns the first in
   `CATEGORIES` order.
2. **Corroboration re-promotion is one-directional.** A new save promotes the
   memory it matches, but a memory promoted to `suspected` earlier is not
   re-checked for `confirmed` if evidence is attached to a *different* corroborator
   later. Revisit only if it bites.
3. **Warm `push()` doesn't detect a divergent edit at push time** -- only `pull()`
   does (via `save_memory`). If machine A edits a memory and pushes, and nobody
   ever pulls on A, A won't see B's conflicting version. Acceptable: pull is the
   sync direction where review belongs.
4. **libSQL integration test tolerates an unreachable server** (skips rather than
   fails) so a flaky image pull doesn't red the build. If the warm tier silently
   regressed and the server also failed to start, that job would go green.

## Working conventions

- Verify every edit landed (the `hearthagent` lineage has a history of silent
  string-replace failures).
- Never silently degrade or auto-modify anything the user hasn't approved — flag
  for review instead. Applies to confidence tiers, prompt templates, routing.
- Keep the README honest: the status table is load-bearing. When a 🔨 becomes ✅,
  update it in both README.md and this file.
- `uv run ruff check . && uv run pytest` must be green before any push;
  `uv run pytest -m semantic` too when touching the memory vault.

## Related

- `hearthagent-pro` — the full local-first agent this was distilled from.
- `~/Downloads/hearthagent-pro-handoff.md` — that project's handoff doc.
- `~/Downloads/eval-and-adaptive-routing-spec-v8.md` — design spec lineage.
