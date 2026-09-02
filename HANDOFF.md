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
  metrics.py             -- SQLite per-turn log + eval baselines/results; category_escalation_rate()
  reliability/
    assertions.py        -- deterministic checks + run_all_checks() aggregator
  routing/
    classifier.py        -- classify_task() -> one of 5 categories; injectable generate_fn
    curator.py           -- curate_memories() -> relevance filter before prompt injection
  memory/
    vault.py             -- MemoryVault: SQLite/FTS5 + optional ChromaDB + confidence scoring
    mcp_server.py         -- FastMCP server (console script: agent-neutral-memory)
  evals/
    runner.py            -- run_eval_judge() (one answer) + run_regression_suite() (all baselines)
tests/                   -- pytest; no network, no model calls (generate_fn is stubbed)
.github/workflows/ci.yml -- ruff + pytest on 3.11 / 3.12 / 3.13
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

## What changed in the pre-release pass (2026-09-02)

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
  extras (removed 8 unused deps: fastapi, uvicorn, pydantic, boto3, pyyaml,
  langgraph, langchain-*).
- `logging` throughout (was silent `except: pass`).
- `run_regression_suite`, `metrics.all_eval_results`, configurable state dirs.
- Public API surface in `agent_neutral_harness/__init__.py`.

## Known gaps / open items

1. **Warm (Turso) and cold (object-store) tiers are not built.** `vault.py`
   docstring describes them; only the hot tier exists. The `hearthagent-pro`
   repo has a working version to port.
2. **Corroboration doesn't re-promote existing memories.** When a new memory
   corroborates an old one, the old one's `corroborations` counter increments but
   its confidence tier is not re-evaluated. Deliberate (re-verifying old evidence
   URLs on every write is too expensive) — revisit if it matters.
3. **Cosine-space assumption.** `_find_corroborators` assumes the Chroma
   collection uses cosine distance (`similarity = 1 - distance`). New collections
   are created with `hnsw:space=cosine`; a pre-existing collection created with the
   default L2 space would score corroboration wrong. Migrate by deleting
   `~/.ai-memory-vault/chroma` and running `sync_embeddings`.
4. **No dashboard in this repo.** The metrics schema supports one; the Streamlit
   dashboard lives in `hearthagent-pro`.
5. **The semantic path is not covered by CI** (ChromaDB + sentence-transformers +
   model download is too heavy for the matrix). Only the SQLite path is tested in
   CI; test the semantic path manually with `uv sync --extra memory`.
6. **`classify_task` loose-matches** the category name as a token in the model's
   reply. A verbose model that mentions two category words returns the first in
   `CATEGORIES` order.

## Working conventions

- Verify every edit landed (the `hearthagent` lineage has a history of silent
  string-replace failures).
- Never silently degrade or auto-modify anything the user hasn't approved — flag
  for review instead. Applies to confidence tiers, prompt templates, routing.
- Keep the README honest: the status table is load-bearing. If you build the warm
  tier, move it from 🔨 to ✅ in both README.md and this file.
- `uv run ruff check . && uv run pytest` must be green before any push.

## Related

- `hearthagent-pro` — the full local-first agent this was distilled from.
- `~/Downloads/hearthagent-pro-handoff.md` — that project's handoff doc.
- `~/Downloads/eval-and-adaptive-routing-spec-v8.md` — design spec lineage.
