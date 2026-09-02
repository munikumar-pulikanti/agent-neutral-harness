"""FastMCP server exposing the MemoryVault to any MCP-compatible client.

Run it with::

    python -m agent_neutral_harness.memory.mcp_server

or via the installed console script ``agent-neutral-memory``.
"""

import logging
import os

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "The MCP server needs the 'mcp' extra:\n"
        "    pip install 'agent-neutral-harness[mcp]'"
    ) from exc

from agent_neutral_harness.memory.vault import MemoryVault

logging.basicConfig(level=os.environ.get("AGENT_NEUTRAL_HARNESS_LOG", "INFO"))

mcp = FastMCP("agent-neutral-harness-memory")

_vault: MemoryVault | None = None


def vault() -> MemoryVault:
    """Instantiate the vault on first tool call, not at import time."""
    global _vault
    if _vault is None:
        _vault = MemoryVault()
    return _vault


@mcp.tool()
def search_memory(query: str) -> str:
    """Fast keyword search over the persistent cross-tool memory vault."""
    return vault().search_keyword(query)


@mcp.tool()
def search_memory_semantic(query: str) -> str:
    """Meaning-based (vector) search over the persistent memory vault."""
    return vault().search_semantic(query)


@mcp.tool()
def save_memory(
    scope: str, memory_type: str, content: str, tags: str = "", evidence_url: str = ""
) -> str:
    """Save a finding to the shared memory vault.

    Confidence is scored automatically from corroboration by existing
    memories and whether ``evidence_url`` is a real, reachable URL.
    """
    return vault().save_memory(scope, memory_type, content, tags, evidence_url)


@mcp.tool()
def sync_embeddings() -> str:
    """Re-index all SQLite memory rows into ChromaDB."""
    return vault().sync_embeddings()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
