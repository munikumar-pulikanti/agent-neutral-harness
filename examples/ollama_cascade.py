"""Real local run — cascade over two Ollama models, no API key.

    ollama pull llama3.2:1b qwen2.5-coder:3b qwen2.5-coder:7b
    python examples/ollama_cascade.py

Same shape as ``quickstart.py``, but ``execute_fn`` actually calls Ollama instead
of returning a canned dict. This is a one-shot text generation per tier (no tool
loop) — enough to show the cascade verifying a *real* model response and
escalating only when a deterministic check fails on it.
"""

import os
import tempfile

os.environ["AGENT_NEUTRAL_HARNESS_METRICS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="anh-ollama-"), "metrics.db"
)

import requests  # noqa: E402

from agent_neutral_harness import metrics, run_cascade  # noqa: E402
from agent_neutral_harness.routing.classifier import classify_task  # noqa: E402

OLLAMA = "http://localhost:11434"
CHEAP = "qwen2.5-coder:3b"
CAPABLE = "qwen2.5-coder:7b"


def execute_fn(model: str, task: str) -> dict:
    """Run `task` on `model` via Ollama, return the harness result shape.

    A real agent would put its tool loop here and populate tools_invoked /
    tool_results. This example does a single generation, so those stay empty
    and the reliability checks that apply are the text-only ones
    (leaked tool-call JSON, empty output, unbacked "tests pass" claims).
    """
    try:
        r = requests.post(
            f"{OLLAMA}/api/generate",
            json={"model": model, "prompt": task, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ollama_call_failed: {exc}"}

    return {
        "final_content": body.get("response", ""),
        "tools_invoked": [],
        "tool_results": {},
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
        "error": None,
    }


def main() -> None:
    task = (
        "Write a Python function `retry(fn, attempts=3)` that retries `fn` with "
        "exponential backoff and re-raises the last exception. Return only code."
    )

    category = classify_task(task, model="llama3.2:1b")
    print(f"category: {category}")

    answer = run_cascade(
        task=task,
        category=category,
        cheap_model=CHEAP,
        capable_model=CAPABLE,
        execute_fn=execute_fn,
        random_fn=lambda: 0.0,  # deterministic: never take the sampling shortcut
    )

    turn = metrics.all_turns()[0]
    print(f"tier used      : {turn['cascade_tier']}")
    print(f"escalated      : {bool(turn['escalated'])}")
    print(f"assertion_flags: {turn['assertion_flags'] or '(none)'}")
    print(f"\n--- answer ---\n{answer[:600]}")


if __name__ == "__main__":
    main()
