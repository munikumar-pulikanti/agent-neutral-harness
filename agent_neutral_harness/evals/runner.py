"""Golden-baseline regression evals with an LLM-as-a-judge.

``run_eval_judge`` grades one actual answer against a stored golden
answer. ``run_regression_suite`` re-runs every saved baseline through a
caller-supplied agent and records pass/fail in the metrics DB.
"""

import json
import logging
from collections.abc import Callable

import requests

from agent_neutral_harness import metrics

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an impartial AI evaluator.
Task: {task}
Golden Expected Answer: {expected}
Actual Agent Response: {actual}

Did the actual agent response fulfill the task accurately according to the golden baseline?
Respond with JSON only: {{"passed": true or false, "reasoning": "one or two sentences"}}"""

# generate_fn(prompt) -> raw model text (expected to be JSON here)
GenerateFn = Callable[[str], str]
# agent_fn(task) -> the agent's answer for that task
AgentFn = Callable[[str], str]


def _ollama_json_generate(model: str, base_url: str, timeout: float) -> GenerateFn:
    def _gen(prompt: str) -> str:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    return _gen


def _parse_verdict(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"passed": False, "reasoning": f"unparseable judge output: {raw!r}"}
    return {
        "passed": bool(data.get("passed", False)),
        "reasoning": str(data.get("reasoning", "")),
    }


def run_eval_judge(
    task: str,
    expected_answer: str,
    actual_answer: str,
    judge_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 30.0,
    generate_fn: GenerateFn | None = None,
) -> dict:
    """Grade ``actual_answer`` against ``expected_answer``.

    Always returns ``{"passed": bool, "reasoning": str}``.
    """
    gen = generate_fn or _ollama_json_generate(judge_model, base_url, timeout)
    prompt = JUDGE_PROMPT.format(task=task, expected=expected_answer, actual=actual_answer)
    try:
        raw = gen(prompt)
    except Exception as exc:  # noqa: BLE001 - judge failure is a fail, not a crash
        log.warning("eval judge call failed", exc_info=True)
        return {"passed": False, "reasoning": f"eval judge error: {exc}"}
    return _parse_verdict(raw)


def run_regression_suite(
    agent_fn: AgentFn,
    judge_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    generate_fn: GenerateFn | None = None,
) -> dict:
    """Re-run every saved baseline through ``agent_fn`` and judge each one.

    Records each result via :func:`agent_neutral_harness.metrics.log_eval_result`
    and returns a summary dict.
    """
    baselines = metrics.all_baselines()
    passed = 0
    details = []
    for b in baselines:
        actual = agent_fn(b["task"])
        verdict = run_eval_judge(
            b["task"], b["expected_summary"], actual,
            judge_model=judge_model, base_url=base_url, generate_fn=generate_fn,
        )
        metrics.log_eval_result(b["id"], verdict["passed"], verdict["reasoning"], actual)
        passed += int(verdict["passed"])
        details.append({"baseline_id": b["id"], "task": b["task"], **verdict})
        log.info("baseline #%s: %s", b["id"], "PASS" if verdict["passed"] else "FAIL")
    return {"total": len(baselines), "passed": passed, "failed": len(baselines) - passed, "details": details}
