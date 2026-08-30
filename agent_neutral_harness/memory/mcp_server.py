"""FastMCP Server exposing MemoryVault to any MCP-compatible client."""

from mcp.server.fastmcp import FastMCP
from agent_neutral_harness.memory.vault import MemoryVault

mcp = FastMCP("agent-neutral-harness-memory")
vault = MemoryVault()


@mcp.tool()
def search_memory(query: str) -> str:
    """Fast keyword search over the persistent cross-tool memory vault."""
    return vault.search_keyword(query)


@mcp.tool()
def search_memory_semantic(query: str) -> str:
    """Meaning-based search over the persistent memory vault."""
    return vault.search_semantic(query)


@mcp.tool()
def save_memory(scope: str, type_: str, content: str, tags: str = "", evidence_url: str = "") -> str:
    """Save a finding to the shared memory vault with optional evidence URL."""
    return vault.save_memory(scope, type_, content, tags, evidence_url)


@mcp.tool()
def sync_embeddings() -> str:
    """Re-index all active SQLite memory rows into ChromaDB."""
    return vault.sync_embeddings()


def main():
    mcp.run()


if __name__ == "__main__":
    main()