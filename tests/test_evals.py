from agent_neutral_harness.evals.runner import _parse_verdict, run_eval_judge, run_regression_suite


def test_parse_verdict_valid_json():
    v = _parse_verdict('{"passed": true, "reasoning": "matches"}')
    assert v == {"passed": True, "reasoning": "matches"}


def test_parse_verdict_garbage():
    v = _parse_verdict("not json at all")
    assert v["passed"] is False
    assert "unparseable" in v["reasoning"]


def test_run_eval_judge_with_injected_generate_fn():
    v = run_eval_judge("t", "expected", "actual", generate_fn=lambda p: '{"passed": false, "reasoning": "off"}')
    assert v == {"passed": False, "reasoning": "off"}


def test_run_eval_judge_error_is_a_fail():
    def boom(_):
        raise RuntimeError("no judge")

    v = run_eval_judge("t", "e", "a", generate_fn=boom)
    assert v["passed"] is False


def test_run_regression_suite(metrics_mod):
    metrics_mod.save_eval_baseline("what is 2+2", "4", "")
    metrics_mod.save_eval_baseline("capital of France", "Paris", "")
    summary = run_regression_suite(
        agent_fn=lambda task: "an answer",
        generate_fn=lambda p: '{"passed": true, "reasoning": "ok"}',
    )
    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert len(metrics_mod.all_eval_results()) == 2
