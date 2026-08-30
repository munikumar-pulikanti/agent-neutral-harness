# Agent-Neutral Harness (`agent-neutral-harness`)

A universal, agent-neutral reliability, memory, and model-routing harness for AI coding workflows.

Plugs directly into **Windsurf**, **Cursor**, **Antigravity**, **Claude Desktop**, or custom Python agents without forcing you into a single editor or runtime.

---

## Key Modules

* **`agent_neutral_harness.memory`**: Multi-tier persistent memory vault (Hot SQLite FTS5/ChromaDB → Warm Turso → Cold MinIO) with automated evidence URL verification.
* **`agent_neutral_harness.memory.mcp_server`**: FastMCP server enabling any MCP-compatible client to read/write the shared memory vault.
* **`agent_neutral_harness.routing`**: Task-aware intent classifier and candidate memory curator powered by lightweight local models (`llama3.2:1b`).
* **`agent_neutral_harness.reliability`**: Deterministic session assertions (catching leaked tool-call JSON, unverified test claims, empty responses).
* **`agent_neutral_harness.evals`**: Golden baseline regression runner with LLM-as-a-judge grading.

---

## Quick Start

```bash
uv sync
```

### Run MCP Server (Cross-Tool Memory)
```bash
uv run python3 -m agent_neutral_harness.memory.mcp_server
```

Add to your MCP config (`mcp_config.json`):
```json
{
  "mcpServers": {
    "agent-neutral-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/agent-neutral-harness", "python3", "-m", "agent_neutral_harness.memory.mcp_server"]
    }
  }
}
```