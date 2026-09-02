# Agent-Neutral Harness

[![CI](https://github.com/munikumar-pulikanti/agent-neutral-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/munikumar-pulikanti/agent-neutral-harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A reliability, memory, and model-routing harness for AI coding workflows that does
**not** own your agent loop.

Most agent frameworks make you adopt their runtime. This one is the opposite: you
keep your agent (LangGraph, a raw tool loop, an IDE assistant, whatever), and pass
it in as a plain `execute_fn(model, task) -> dict`. The harness handles the parts
that are the same everywhere — deciding when a cheap model is good enough, catching
a class of model failures deterministically, and sharing a memory vault across
tools over MCP.

Runs fully local by default (Ollama, CPU, no API key). Model names are always
parameters — nothing here hardcodes a model or a provider.

---

## What's here

| Module | Purpose | Status |
|---|---|---|
| `cascade` | Run a cheap model, verify its **real** output, escalate to a capable model only on an **observed** failure. Learns per-category escalation rates and shortcuts hopeless categories (while still sampling them). | ✅ Working, tested |
| `reliability.assertions` | Deterministic checks on a response: leaked tool-call JSON, empty output, "tests passed" with nothing run, filenames in the summary that aren't in the real tool output. | ✅ Working, tested |
| `metrics` | Per-turn SQLite log (model, tokens, tier, escalation, flags) + eval baselines/results. | ✅ Working, tested |
| `routing.classifier` | Lightweight task-intent classifier (`investigate` / `implement` / `unit_tests` / `extract` / `general`). Ollama by default; inject any `generate_fn`. | ✅ Working, tested |
| `routing.curator` | Filter retrieved memories for *actual task relevance* before prompt injection, not just semantic similarity. | ✅ Working, tested |
| `memory.vault` | SQLite + FTS5 keyword search, optional ChromaDB semantic search, **evidence-gated confidence** (`hypothesis` → `suspected` → `confirmed`, where `confirmed` requires a verified evidence URL). | ✅ Hot tier working |
| `memory.mcp_server` | FastMCP server exposing the vault to any MCP client (Claude Desktop, Cursor, Windsurf, …). | ✅ Working |
| `evals.runner` | Re-run golden baselines through your agent, grade with an LLM judge, record pass/fail. | ✅ Working, tested |
| Warm tier (Turso) / cold tier (object store) | Cross-machine sync + archival. | 🔨 Designed, not built — see [HANDOFF.md](HANDOFF.md) |
| Dashboard | Visualize the metrics DB. | 🔨 Not in this repo yet |

---

## Install

```bash
pip install "agent-neutral-harness[all]"        # everything
pip install "agent-neutral-harness"             # core only (cascade + assertions + metrics + routing)
pip install "agent-neutral-harness[memory]"     # + semantic memory search (ChromaDB)
pip install "agent-neutral-harness[mcp]"        # + MCP server
```

Local development:

```bash
uv sync --group dev --extra all
uv run pytest
uv run ruff check .
```

---

## Use it

### Cascade — cheap-first, verify, escalate on real failure

```python
from agent_neutral_harness import run_cascade

def execute_fn(model: str, task: str) -> dict:
    # YOUR agent runtime. Return the real result of running `task` on `model`.
    return {
        "final_content": "...",
        "tools_invoked": ["run_shell"],
        "tool_results": {"run_shell": "..."},
        "input_tokens": 1200, "output_tokens": 300,
        "error": None,
    }

answer = run_cascade(
    task="add a retry to the upload helper",
    category="implement",
    cheap_model="llama3.2:3b",
    capable_model="llama3.1:8b",
    execute_fn=execute_fn,
)
```

The cheap tier is only trusted if its output passes every deterministic check.
Escalation happens on **observed** failure, never a confidence guess. Once a
category's recent escalation rate passes 80% (min 20 samples), the cascade starts
skipping the cheap tier for it — but still samples it 20% of the time so it can
notice the cheap model getting better.

### Reliability checks standalone

```python
from agent_neutral_harness import run_all_checks

flags = run_all_checks(
    response_text=model_output,
    tools_invoked=["read_file"],
    tool_results={"read_file": actual_file_contents},
)
# [] means clean; e.g. ["fabricated_items_in_summary"] means the summary
# named a file that wasn't in the real tool output.
```

### Cross-tool memory over MCP

```bash
agent-neutral-memory          # runs the MCP server on stdio
```

Claude Desktop / Cursor / Windsurf MCP config:

```json
{
  "mcpServers": {
    "agent-neutral-memory": {
      "command": "agent-neutral-memory"
    }
  }
}
```

Tools exposed: `search_memory`, `search_memory_semantic`, `save_memory`,
`sync_embeddings`. A saved memory starts as `hypothesis` and is only promoted to
`confirmed` when a similar memory already exists **and** the supplied
`evidence_url` is real and reachable (checked with an SSRF-guarded HTTP request).

---

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `AGENT_NEUTRAL_HARNESS_HOME` | `~/.agent-neutral-harness` | Base dir for state |
| `AGENT_NEUTRAL_HARNESS_METRICS_DB` | `<home>/metrics.db` | Metrics DB path |
| `AI_MEMORY_VAULT_DIR` | `~/.ai-memory-vault` | Memory vault dir (SQLite + Chroma) |
| `AGENT_NEUTRAL_HARNESS_LOG` | `INFO` | Log level for the MCP server |

---

## License

MIT — see [LICENSE](LICENSE).
