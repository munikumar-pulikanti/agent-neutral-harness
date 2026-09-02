"""Curation engine: filter retrieved memory candidates for actual task
relevance before they are injected into a prompt.

Semantic similarity alone lets topically-close-but-irrelevant memories
through; a cheap model deciding "is this actually useful for THIS task"
is a better gate. Defaults to local Ollama; inject ``generate_fn`` for
any other provider.
"""

import logging
from collections.abc import Callable

import requests

from agent_neutral_harness.memory.vault import NO_RESULT_MARKERS

log = logging.getLogger(__name__)

CURATION_PROMPT = """Task: {task}

Retrieved memory entries:
{candidates}

Which of these entries, if any, are actually relevant and useful for answering the task above?
Return ONLY the relevant entries, copied exactly as given, one per line.
If none are relevant, respond with exactly: NONE

Do not explain your reasoning. Do not add anything not in the original entries."""

GenerateFn = Callable[[str], str]


def _ollama_generate(model: str, base_url: str, timeout: float) -> GenerateFn:
    def _gen(prompt: str) -> str:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    return _gen


def _is_empty_result(text: str) -> bool:
    return not text or not text.strip() or any(marker in text for marker in NO_RESULT_MARKERS)


def curate_memories(
    task: str,
    raw_memory_text: str,
    model: str = "llama3.2:1b",
    base_url: str = "http://localhost:11434",
    timeout: float = 15.0,
    generate_fn: GenerateFn | None = None,
) -> str:
    """Return only the memory entries relevant to ``task``.

    Passes an empty/"no results" input straight through. On model error,
    fails open (returns the raw candidates unchanged) rather than dropping
    potentially useful context.
    """
    if _is_empty_result(raw_memory_text):
        return raw_memory_text
    gen = generate_fn or _ollama_generate(model, base_url, timeout)
    try:
        curated = gen(CURATION_PROMPT.format(task=task, candidates=raw_memory_text)).strip()
    except Exception:  # noqa: BLE001 - curation must never be fatal
        log.warning("memory curation failed; passing raw candidates through", exc_info=True)
        return raw_memory_text
    if not curated or curated.upper().startswith("NONE"):
        return ""
    return curated
