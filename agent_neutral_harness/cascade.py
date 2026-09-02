"""Cascade execution: attempt a cheap model first, verify its real output
with deterministic checks, and escalate to a capable model only on an
*observed* failure -- never a predicted or confidence-based guess.

Model names are always passed in as parameters. Nothing here assumes a
specific local or cloud model stack -- that is a deployment-time choice,
not something this module should hardcode. The caller supplies
``execute_fn`` so the cascade stays agnostic to which agent framework or
runtime actually executes the task. That is the "agent-neutral" part.
"""

import logging
import random
import time
from collections.abc import Callable

from agent_neutral_harness import metrics
from agent_neutral_harness.reliability.assertions import run_all_checks

log = logging.getLogger(__name__)

# Shortcut tuning -- module-level so a deployment can override if needed.
SHORTCUT_MIN_SAMPLES = 20
SHORTCUT_ESCALATION_THRESHOLD = 0.8
SHORTCUT_SAMPLE_RATE = 0.2
# Compare the threshold against the Wilson lower bound, not the raw rate,
# so a lucky small sample can't switch the shortcut on. Set False to use
# the raw fraction.
SHORTCUT_USE_LOWER_BOUND = True

# Keys ``execute_fn`` may return. Missing keys are tolerated and defaulted.
_RESULT_DEFAULTS = {
    "final_content": "",
    "tool_call_count": 0,
    "tools_invoked": [],
    "tool_results": {},
    "input_tokens": 0,
    "output_tokens": 0,
    "error": None,
}

ExecuteFn = Callable[[str, str], dict]


def _safe_execute(execute_fn: ExecuteFn, model: str, task: str) -> dict:
    """Call ``execute_fn`` and always return a normalized result dict.

    An exception raised by ``execute_fn`` is turned into an ``error`` entry
    rather than propagating, so a single bad runtime call cannot crash the
    cascade or leave metrics unlogged.
    """
    try:
        raw = execute_fn(model, task) or {}
    except Exception as exc:  # noqa: BLE001 -- runtime is caller-defined
        log.exception("execute_fn raised for model=%s", model)
        raw = {"error": f"execute_fn_exception: {exc}"}
    return {**_RESULT_DEFAULTS, **raw}


def _tokens(result: dict) -> int:
    return int(result.get("input_tokens", 0)) + int(result.get("output_tokens", 0))


def run_cascade(
    task: str,
    category: str,
    cheap_model: str,
    capable_model: str,
    execute_fn: ExecuteFn,
    random_fn: Callable[[], float] | None = None,
    config_fingerprint: str = "",
) -> str:
    """Run the cascade for one task and return the final content string.

    ``execute_fn(model_name, task)`` returns a dict with keys:
        final_content: str
        tool_call_count: int
        tools_invoked: list[str]
        tool_results: dict[str, str]
        input_tokens: int
        output_tokens: int
        error: str | None   (e.g. "recursion_limit", or an error message)

    ``config_fingerprint`` (see :mod:`agent_neutral_harness.fingerprint`)
    scopes the escalation-rate history to the current model + prompt +
    tool schema, so any change to those drops stale rows from the window.

    Every turn is logged to :mod:`agent_neutral_harness.metrics`. A
    shortcut-skipped cheap tier is recorded with ``cheap_attempt_tokens``
    NULL so it does not pollute the escalation-rate signal.
    """
    random_fn = random_fn or random.random
    start = time.time()

    stats = metrics.category_escalation_rate(category, config_fingerprint)
    if SHORTCUT_USE_LOWER_BOUND and stats["escalation_rate_lower_bound"] is not None:
        rate_signal = stats["escalation_rate_lower_bound"]
    else:
        rate_signal = stats["escalation_rate"]
    shortcut_eligible = (
        stats["sample_size"] >= SHORTCUT_MIN_SAMPLES
        and rate_signal is not None
        and rate_signal >= SHORTCUT_ESCALATION_THRESHOLD
    )
    skip_cheap = shortcut_eligible and random_fn() > SHORTCUT_SAMPLE_RATE

    # ----- Shortcut path: category has a proven-high escalation rate -----
    if skip_cheap:
        log.info(
            "shortcut: skipping cheap tier for category=%s (escalation_rate=%.2f, n=%d)",
            category, stats["escalation_rate"], stats["sample_size"],
        )
        result = _safe_execute(execute_fn, capable_model, task)
        duration = time.time() - start
        if result["error"]:
            metrics.log_turn(
                task_snippet=task, category=category, model=capable_model,
                duration_seconds=duration, error_occurred=True,
                assertion_flags=f"capable_tier_error:{result['error']}",
                cascade_tier="capable", escalated=True,
                cheap_attempt_tokens=None, capable_attempt_tokens=0,
                config_fingerprint=config_fingerprint,
            )
            return f"Error: {result['error']}"

        flags = run_all_checks(
            result["final_content"], result["tools_invoked"], result["tool_results"]
        )
        metrics.log_turn(
            task_snippet=task, category=category, model=capable_model,
            duration_seconds=duration, tool_call_count=result["tool_call_count"],
            error_occurred=False,
            assertion_flags="shortcut_skip" + (":" + ",".join(flags) if flags else ""),
            input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
            cascade_tier="capable", escalated=True,
            cheap_attempt_tokens=None, capable_attempt_tokens=_tokens(result),
            config_fingerprint=config_fingerprint,
        )
        return result["final_content"]

    # ----- Normal path: try the cheap tier, verify its real output -----
    cheap_result = _safe_execute(execute_fn, cheap_model, task)
    if cheap_result["error"]:
        cheap_flags = [f"cheap_tier_error:{cheap_result['error']}"]
        cheap_tokens = 0
    else:
        cheap_flags = run_all_checks(
            cheap_result["final_content"],
            cheap_result["tools_invoked"],
            cheap_result["tool_results"],
        )
        cheap_tokens = _tokens(cheap_result)

    if not cheap_flags:
        duration = time.time() - start
        metrics.log_turn(
            task_snippet=task, category=category, model=cheap_model,
            duration_seconds=duration, tool_call_count=cheap_result["tool_call_count"],
            error_occurred=False, assertion_flags="",
            input_tokens=cheap_result["input_tokens"], output_tokens=cheap_result["output_tokens"],
            cascade_tier="cheap", escalated=False,
            cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=0,
            config_fingerprint=config_fingerprint,
        )
        return cheap_result["final_content"]

    # ----- Observed failure -> escalate to the capable tier -----
    log.info("escalating category=%s cheap->capable due to: %s", category, cheap_flags)
    capable_result = _safe_execute(execute_fn, capable_model, task)
    duration = time.time() - start

    if capable_result["error"]:
        metrics.log_turn(
            task_snippet=task, category=category, model=capable_model,
            duration_seconds=duration, error_occurred=True,
            assertion_flags=(
                f"escalated_due_to:{','.join(cheap_flags)};"
                f"capable_tier_error:{capable_result['error']}"
            ),
            cascade_tier="capable", escalated=True,
            cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=0,
            config_fingerprint=config_fingerprint,
        )
        return f"Error: {capable_result['error']}"

    capable_flags = run_all_checks(
        capable_result["final_content"],
        capable_result["tools_invoked"],
        capable_result["tool_results"],
    )
    assertion_note = "escalated_due_to:" + ",".join(cheap_flags)
    if capable_flags:
        assertion_note += ";capable_flags:" + ",".join(capable_flags)

    metrics.log_turn(
        task_snippet=task, category=category, model=capable_model,
        duration_seconds=duration, tool_call_count=capable_result["tool_call_count"],
        error_occurred=False, assertion_flags=assertion_note,
        input_tokens=capable_result["input_tokens"], output_tokens=capable_result["output_tokens"],
        cascade_tier="capable", escalated=True,
        cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=_tokens(capable_result),
        config_fingerprint=config_fingerprint,
    )
    return capable_result["final_content"]
