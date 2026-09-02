"""Agent-Neutral Harness: universal reliability, memory, and model-routing
for AI coding agents.

Public API:
    run_cascade            -- cheap-first, verify-then-escalate execution
    run_all_checks         -- deterministic reliability assertions
    MemoryVault            -- persistent memory with evidence-gated confidence
    classify_task          -- lightweight task-intent classifier
    curate_memories        -- relevance filter for retrieved memories
    run_eval_judge         -- LLM-as-a-judge grading of one answer
    run_regression_suite   -- golden-baseline regression runner
"""

__version__ = "0.1.0"

from agent_neutral_harness.cascade import run_cascade
from agent_neutral_harness.fingerprint import config_fingerprint, ollama_model_digest
from agent_neutral_harness.reliability.assertions import run_all_checks

__all__ = [
    "__version__",
    "run_cascade",
    "run_all_checks",
    "config_fingerprint",
    "ollama_model_digest",
    "MemoryVault",
    "classify_task",
    "curate_memories",
    "run_eval_judge",
    "run_regression_suite",
]


def __getattr__(name: str):
    # Lazy re-exports so importing the package never eagerly pulls optional
    # deps (chromadb, mcp) or triggers model downloads.
    if name == "MemoryVault":
        from agent_neutral_harness.memory.vault import MemoryVault

        return MemoryVault
    if name == "classify_task":
        from agent_neutral_harness.routing.classifier import classify_task

        return classify_task
    if name == "curate_memories":
        from agent_neutral_harness.routing.curator import curate_memories

        return curate_memories
    if name in ("run_eval_judge", "run_regression_suite"):
        from agent_neutral_harness.evals import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
