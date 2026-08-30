"""Curation engine to filter retrieved memory candidates before prompt injection."""

import requests

CURATION_PROMPT = """Task: {task}

Retrieved memory entries:
{candidates}

Which of these entries, if any, are actually relevant and useful for answering the task above?
Return ONLY the relevant entries, copied exactly as given, one per line.
If none are relevant, respond with exactly: NONE

Do not explain your reasoning. Do not add anything not in the original entries."""


def curate_memories(task: str, raw_memory_text: str, model: str = "llama3.2:1b", base_url: str = "http://localhost:11434") -> str:
    if not raw_memory_text or "No matching memories" in raw_memory_text or "No semantic matches" in raw_memory_text:
        return raw_memory_text
    prompt = CURATION_PROMPT.format(task=task, candidates=raw_memory_text)
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=15,
        ).json()
        curated = resp.get("response", "").strip()
        if not curated or curated.upper().startswith("NONE"):
            return ""
        return curated
    except Exception:
        return raw_memory_text