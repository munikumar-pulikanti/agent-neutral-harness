"""Config fingerprinting for escalation-rate history invalidation.

Accumulated per-category escalation-rate stats are only meaningful while
the thing being measured hasn't changed. A model weight swap (same Ollama
tag, new digest), a system-prompt edit, or a tool being added / removed /
retyped all shift the cheap tier's win rate just as much as each other.

``config_fingerprint`` combines those into one short hash. Pass it to
:func:`agent_neutral_harness.metrics.category_escalation_rate` and
:func:`agent_neutral_harness.cascade.run_cascade` so a change to any input
naturally drops stale rows out of the window instead of dragging the
routing decision for ~50 turns.

Deliberately structural: tool *description prose* is not hashed. Hashing
prose caused phantom invalidations on typo fixes; the bet is that
structural changes (name / parameter names / parameter types) are the
dominant driver of real win-rate shifts. A meaningful description rewrite
that leaves the signature intact is instead caught by
:func:`agent_neutral_harness.metrics.detect_within_window_drift`.
"""

import hashlib
import logging
import os
from collections.abc import Iterable, Mapping

import requests

log = logging.getLogger(__name__)

_digest_cache: dict[str, str] = {}


def ollama_model_digest(
    model: str, base_url: str = "http://localhost:11434", timeout: float = 5.0
) -> str:
    """Short digest of a model's weights from Ollama's ``/api/tags`` (cached).

    Returns ``""`` if Ollama isn't reachable or the model isn't found —
    callers treat an empty component as "unknown", not "changed".
    """
    if model in _digest_cache:
        return _digest_cache[model]
    digest = ""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if model in (m.get("name"), m.get("model")):
                digest = (m.get("digest") or "")[:12]
                break
    except requests.RequestException:
        log.debug("could not fetch model digest for %r", model, exc_info=True)
    _digest_cache[model] = digest
    return digest


def _tool_signature(tools: Iterable) -> str:
    """Structural signature of a tool set: name + sorted (param, type) pairs.

    Accepts anything with a ``name`` and an ``args`` mapping (LangChain
    tools), or plain dicts shaped ``{"name": ..., "parameters": {...}}``.
    """
    parts = []
    for t in tools or []:
        if isinstance(t, Mapping):
            name = t.get("name", "")
            params = t.get("parameters") or t.get("args") or {}
        else:
            name = getattr(t, "name", "")
            params = getattr(t, "args", None) or {}
        param_structure = sorted(
            (str(k), str((v or {}).get("type", "") if isinstance(v, Mapping) else ""))
            for k, v in dict(params).items()
        )
        parts.append(f"{name}|{param_structure}")
    return "||".join(sorted(parts))


def config_fingerprint(
    *,
    model_digest: str = "",
    system_prompt: str = "",
    tools: Iterable | None = None,
    normalize_cwd: bool = True,
    extra: str = "",
) -> str:
    """Combine the behaviour-affecting inputs into one 16-char hash.

    ``model_digest``  : e.g. from :func:`ollama_model_digest`.
    ``system_prompt`` : the exact prompt the model will see.
    ``tools``         : the tool objects/dicts exposed to the model.
    ``normalize_cwd`` : replace ``os.getcwd()`` in the prompt with a
                        placeholder (paths vary per machine without being
                        a real behaviour change).
    ``extra``         : any additional string a deployment wants folded in.
    """
    prompt = system_prompt
    if normalize_cwd and prompt:
        prompt = prompt.replace(os.getcwd(), "<CWD>")
    combined = f"{model_digest}::{prompt}::{_tool_signature(tools or [])}::{extra}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
