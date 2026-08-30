"""Task classifier using lightweight local LLM (e.g., llama3.2:1b)."""

import requests

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


def classify_task(task: str, model: str = "llama3.2:1b", base_url: str = "http://localhost:11434") -> str:
    prompt = CLASSIFIER_PROMPT.format(task=task)
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=10,
        ).json()
        category = resp.get("response", "").strip().lower()
        if category in {"investigate", "implement", "unit_tests", "extract", "general"}:
            return category
    except Exception:
        pass
    return "general"