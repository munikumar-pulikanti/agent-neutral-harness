import pytest

from agent_neutral_harness import cascade


def make_execute(responses):
    """responses: dict model_name -> result dict (or callable(task)->dict)."""
    calls = []

    def execute_fn(model, task):
        calls.append(model)
        r = responses[model]
        return r(task) if callable(r) else dict(r)

    execute_fn.calls = calls
    return execute_fn


GOOD = {
    "final_content": "All good.", "tool_call_count": 0, "tools_invoked": [],
    "tool_results": {}, "input_tokens": 5, "output_tokens": 5, "error": None,
}
LEAKY = {**GOOD, "final_content": '{"name": "run", "parameters": {}}'}


@pytest.fixture(autouse=True)
def _isolated_metrics(metrics_mod):
    # ensures every cascade test writes to a temp metrics db
    yield


def test_cheap_success_does_not_escalate():
    ex = make_execute({"cheap": GOOD, "capable": GOOD})
    out = cascade.run_cascade("t", "general", "cheap", "capable", ex, random_fn=lambda: 0.0)
    assert out == "All good."
    assert ex.calls == ["cheap"]


def test_cheap_failure_escalates_to_capable():
    ex = make_execute({"cheap": LEAKY, "capable": GOOD})
    out = cascade.run_cascade("t", "general", "cheap", "capable", ex, random_fn=lambda: 0.0)
    assert out == "All good."
    assert ex.calls == ["cheap", "capable"]


def test_cheap_runtime_error_escalates():
    ex = make_execute({"cheap": {**GOOD, "error": "recursion_limit"}, "capable": GOOD})
    out = cascade.run_cascade("t", "general", "cheap", "capable", ex, random_fn=lambda: 0.0)
    assert out == "All good."
    assert ex.calls == ["cheap", "capable"]


def test_execute_fn_exception_is_contained():
    def boom(model, task):
        raise RuntimeError("kaboom")

    ex = make_execute({"cheap": boom, "capable": GOOD})
    out = cascade.run_cascade("t", "general", "cheap", "capable", ex, random_fn=lambda: 0.0)
    assert out == "All good."


def test_capable_error_raises_cascade_error_after_escalation():
    ex = make_execute({"cheap": LEAKY, "capable": {**GOOD, "error": "boom"}})
    with pytest.raises(cascade.CascadeError) as exc_info:
        cascade.run_cascade("t", "general", "cheap", "capable", ex, random_fn=lambda: 0.0)
    assert str(exc_info.value) == "boom"
    assert exc_info.value.tier == "capable"
    assert exc_info.value.cheap_error  # cheap tier's own flags carried along


def test_capable_error_on_shortcut_path_raises_cascade_error(metrics_mod):
    for _ in range(cascade.SHORTCUT_MIN_SAMPLES):
        metrics_mod.log_turn(
            task_snippet="t", category="implement", model="capable",
            duration_seconds=0.1, escalated=True, cheap_attempt_tokens=1,
        )
    ex = make_execute({"cheap": GOOD, "capable": {**GOOD, "error": "boom"}})
    with pytest.raises(cascade.CascadeError) as exc_info:
        cascade.run_cascade("t", "implement", "cheap", "capable", ex, random_fn=lambda: 0.99)
    assert str(exc_info.value) == "boom"
    assert exc_info.value.cheap_error is None  # shortcut path never ran the cheap tier


def test_shortcut_skips_cheap_when_category_escalation_is_high(metrics_mod):
    for _ in range(cascade.SHORTCUT_MIN_SAMPLES):
        metrics_mod.log_turn(
            task_snippet="t", category="implement", model="capable",
            duration_seconds=0.1, escalated=True, cheap_attempt_tokens=1,
        )
    ex = make_execute({"cheap": GOOD, "capable": GOOD})
    # random_fn > SHORTCUT_SAMPLE_RATE -> take the shortcut
    out = cascade.run_cascade("t", "implement", "cheap", "capable", ex, random_fn=lambda: 0.99)
    assert out == "All good."
    assert ex.calls == ["capable"]


def test_shortcut_history_is_scoped_to_config_fingerprint(metrics_mod):
    # 20 escalated turns, but all under a STALE fingerprint
    for _ in range(cascade.SHORTCUT_MIN_SAMPLES):
        metrics_mod.log_turn(
            task_snippet="t", category="implement", model="capable", duration_seconds=0.1,
            escalated=True, cheap_attempt_tokens=1, config_fingerprint="STALE",
        )
    ex = make_execute({"cheap": GOOD, "capable": GOOD})
    out = cascade.run_cascade(
        "t", "implement", "cheap", "capable", ex,
        random_fn=lambda: 0.99, config_fingerprint="CURRENT",
    )
    # stale history doesn't count -> cheap tier is still tried
    assert out == "All good."
    assert ex.calls == ["cheap"]
