"""Runnable end-to-end example — no Ollama, no API key, no extras.

    python examples/quickstart.py

Everything the harness needs from you is a plain ``execute_fn(model, task) -> dict``.
This file fakes one so the example runs anywhere, then walks the three things the
harness does on top of it:

  1. classify a task's intent          (routing.classifier)
  2. run the cheap model, verify its   (cascade + reliability.assertions)
     REAL output, escalate only on an
     observed failure
  3. show what got logged              (metrics)

Swap ``fake_execute`` for a call into your own agent (LangGraph, a raw tool loop,
an IDE assistant) and nothing else changes — see ``langgraph_cascade.py``.
"""

import os
import tempfile

# Point the metrics DB at a throwaway file so the example never touches your real
# ~/.agent-neutral-harness state. Must be set before importing the package.
os.environ["AGENT_NEUTRAL_HARNESS_METRICS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="anh-quickstart-"), "metrics.db"
)

from agent_neutral_harness import metrics, run_all_checks, run_cascade  # noqa: E402
from agent_neutral_harness.routing.classifier import classify_task  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Your runtime, faked.
# --------------------------------------------------------------------------- #
# The cheap model here "reads a file", then names a second file in its summary
# that was never in the tool output — a real failure mode (fabricated_items).
# The capable model does the same task honestly.
def fake_execute(model: str, task: str) -> dict:
    if "3b" in model:  # the cheap tier
        return {
            "final_content": "Updated upload_helper.py and retry_config.py to add the retry.",
            "tools_invoked": ["read_file"],
            "tool_results": {"read_file": "# upload_helper.py\ndef upload(...): ..."},
            "input_tokens": 900,
            "output_tokens": 120,
            "error": None,
        }
    return {  # the capable tier
        "final_content": "Added a 3-attempt exponential backoff to upload() in upload_helper.py.",
        "tools_invoked": ["read_file"],
        "tool_results": {"read_file": "# upload_helper.py\ndef upload(...): ..."},
        "input_tokens": 1600,
        "output_tokens": 240,
        "error": None,
    }


def fake_classifier(prompt: str) -> str:
    # Stand-in for a 1B local model. Your real one gets injected the same way.
    return "implement"


def main() -> None:
    task = "add a retry to the upload helper"

    # 1. intent routing -------------------------------------------------------
    category = classify_task(task, generate_fn=fake_classifier)
    print(f"task       : {task}")
    print(f"category   : {category}\n")

    # 2. standalone reliability check (what the cascade runs internally) ------
    bad = "I refactored the parser. All tests pass."
    print("reliability check on an 'all tests pass' claim with no test tool run:")
    print(f"  flags    : {run_all_checks(bad, tools_invoked=[], tool_results={})}\n")

    # 3. the cascade --------------------------------------------------------
    # random_fn=lambda: 0.0 makes the holdout-sampling deterministic for the demo.
    answer = run_cascade(
        task=task,
        category=category,
        cheap_model="llama3.2:3b",
        capable_model="llama3.1:8b",
        execute_fn=fake_execute,
        random_fn=lambda: 0.0,
    )
    print(f"cascade answer: {answer}\n")

    # 4. what got logged --------------------------------------------------
    turn = metrics.all_turns()[0]
    print("logged turn:")
    print(f"  tier used        : {turn['cascade_tier']}")
    print(f"  escalated        : {bool(turn['escalated'])}")
    print(f"  assertion_flags  : {turn['assertion_flags'] or '(none)'}")
    print(
        "\nThe cheap tier fabricated a filename, the deterministic check caught it,\n"
        "and the cascade escalated — no model was asked to grade another model."
    )


if __name__ == "__main__":
    main()
