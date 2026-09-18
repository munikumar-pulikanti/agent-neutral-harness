"""Task classifier.

Defaults to a local Ollama model (``llama3.2:1b``) but any text-in /
text-out callable can be injected via ``generate_fn`` to keep routing
provider-neutral.
"""

import logging
import re
from collections.abc import Callable

import requests

log = logging.getLogger(__name__)

CATEGORIES = ("investigate", "implement", "unit_tests", "extract", "general")
DEFAULT_CATEGORY = "general"

CLASSIFIER_PROMPT = """You are a task classifier. Given the following user request, determine the primary category.
Categories:
- investigate (reading logs, exploring code, debugging errors)
- implement (writing new code, modifying functions, building features)
- unit_tests (writing or fixing unit tests)
- extract (extracting data, parsing text, transforming structure)
- general (explanations, architecture queries, small talk)

Respond with ONLY the single category name, in lowercase.

User request: {task}
Category:"""

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


def _coerce_category(raw: str) -> str:
    # Collapse every non-letter run to a single "_" so "Category: implement."
    # and "unit tests" both normalise cleanly ("category_implement",
    # "unit_tests"), then match a category as a whole token.
    normalized = re.sub(r"[^a-z]+", "_", raw.lower()).strip("_")
    matches = [c for c in CATEGORIES if re.search(rf"(^|_){c}($|_)", normalized)]
    if len(matches) == 1:
        return matches[0]
    # Zero matches: unrecognised reply. More than one: a verbose reply named
    # two categories and there's no honest way to prefer one -- guessing the
    # first in CATEGORIES order (the old behaviour) is arbitrary, not a real
    # classification. Both fail safe to the default rather than guess.
    log.debug("classifier reply %r resolved to %d category matches %r; defaulting to %r",
               raw, len(matches), matches, DEFAULT_CATEGORY)
    return DEFAULT_CATEGORY


def classify_task(
    task: str,
    model: str = "llama3.2:1b",
    base_url: str = "http://localhost:11434",
    timeout: float = 10.0,
    generate_fn: GenerateFn | None = None,
) -> str:
    """Return one of :data:`CATEGORIES`, falling back to ``general`` on any error."""
    gen = generate_fn or _ollama_generate(model, base_url, timeout)
    try:
        raw = gen(CLASSIFIER_PROMPT.format(task=task))
    except Exception:  # noqa: BLE001 - classification must never be fatal
        log.warning("task classification failed; defaulting to %r", DEFAULT_CATEGORY, exc_info=True)
        return DEFAULT_CATEGORY
    return _coerce_category(raw or "")
