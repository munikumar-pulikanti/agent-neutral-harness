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
(real MinIO + libSQL). 69 tests total (66 fast lane).

## Fourth pass (pre-launch hardening, this session) -- 0.1.0 -> 0.2.0

Prompted by a Staff-level portfolio review before the public/LinkedIn launch.
Full findings and rationale in [`DESIGN.md`](DESIGN.md) and
[`docs/owasp-agentic-mapping.md`](docs/owasp-agentic-mapping.md); summary:

- **Fixed a real SSRF gap** in `memory.vault._verify_evidence_url`: it used
  `requests.head(..., allow_redirects=True)`, which follows a redirect to
  *any* target with no safety re-check -- a URL that passed `_url_is_safe`
  could 302 to a private/internal host (e.g. cloud metadata) and be
  followed anyway. Now walks redirects manually, re-validating each hop.
  4 new tests in `test_vault.py`. Residual known gap (DNS-rebinding TOCTOU)
  documented in `SECURITY.md`, not silently ignored.
- **Fixed known gap #1** (`classify_task` picking the first matching
  category in declaration order on an ambiguous reply) -- now falls back to
  `DEFAULT_CATEGORY` on 0 or 2+ matches instead of guessing. Test added.
- **Breaking API change, `run_cascade`:** errors from the capable tier used
  to return as an `"Error: ..."` string indistinguishable from real model
  output. Now raises `CascadeError` instead. Done now (v0.1.0 has ~zero
  external adoption -- cheapest time to break this). Bumped to **0.2.0**.
  `examples/*.py` checked -- none string-matched `"Error:"`, none needed
  updating.
- **Added `mypy` as a CI job** (`typecheck`, alongside `test`/`semantic`/
  `integration`). Codebase was already clean except one real type-narrowing
  bug in `metrics._window` (fixed, `params: tuple[object, ...]`).
- **Added `SECURITY.md`**, **`DESIGN.md`**, and
  **`docs/owasp-agentic-mapping.md`** (honest mapping against the OWASP Top
  10 for Agentic Applications, published 2025-12-09 -- three categories
  are genuine strengths already in the design, ASI01 goal-hijack is an
  honestly-stated gap, not glossed over).
- **README**: added a "Why not just use X?" section (LiteLLM, Mem0/Zep/
  Letta) so a skeptical reader isn't left to guess the differentiation, and
  a "More reading" section linking the new docs.
- **LongMemEval benchmark: done.** Real run, `bench/longmemeval_bench.py`
  (tracked, portable -- fetches the dataset via `huggingface_hub`, not a
  hardcoded local cache path), full per-question output in
  `bench/longmemeval_bench_results.json`. **9/12 (75%)**, 12 questions
  stratified across all 6 LongMemEval question types (2/type, seed=42,
  oracle split). Judge was `llama3.2:1b` (swapped down from the planned
  `llama3.1:8b` for CPU speed -- a real methodology caveat, stated in
  README/DESIGN.md/the script's own docstring, not hidden). Genuine,
  category-consistent miss on temporal-reasoning (0/2) -- stated as an
  honest limitation, not smoothed over. Two false starts before this
  number (a too-short judge timeout gave a bogus 0%, then chromadb got
  transiently uninstalled by an unrelated parallel `uv sync`) were caught
  and rejected before landing on the real result -- worth knowing this
  wasn't the first number produced, in case it comes up.
- **Before going public:** this pass needs its own tag (`v0.2.0`) same as
  the prior note about `v0.1.0`'s artifacts predating later commits --
  re-cut before `gh repo edit --visibility public`.

## Known gaps / open items

1. ~~`classify_task` loose-matches...~~ Fixed this pass (see above).
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
5. **No A2A (agent-to-agent) protocol support.** `[mcp]` shares memory across
   tools on one machine; there's no A2A transport. A2A reached Linux
   Foundation v1.0 and real production use across multiple industries in
   2026 -- industry sources now describe MCP + A2A as the two protocols a
   cross-vendor agent stack is expected to speak. Worth a real look, not
   implemented.

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
