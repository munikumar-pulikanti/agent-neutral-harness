def _log(m, category, escalated, cheap_tokens=10):
    m.log_turn(
        task_snippet="t", category=category, model="x", duration_seconds=0.1,
        escalated=escalated, cheap_attempt_tokens=cheap_tokens,
    )


def test_escalation_rate_empty(metrics_mod):
    assert metrics_mod.category_escalation_rate("implement") == {
        "sample_size": 0, "escalation_rate": None,
    }


def test_escalation_rate_counts(metrics_mod):
    for _ in range(3):
        _log(metrics_mod, "implement", escalated=True)
    _log(metrics_mod, "implement", escalated=False)
    stats = metrics_mod.category_escalation_rate("implement")
    assert stats["sample_size"] == 4
    assert stats["escalation_rate"] == 0.75


def test_escalation_rate_excludes_shortcut_skipped_turns(metrics_mod):
    # shortcut-skipped turns are logged with cheap_attempt_tokens = NULL
    _log(metrics_mod, "extract", escalated=False)
    metrics_mod.log_turn(
        task_snippet="t", category="extract", model="x", duration_seconds=0.1,
        escalated=True, cheap_attempt_tokens=None,
    )
    stats = metrics_mod.category_escalation_rate("extract")
    assert stats["sample_size"] == 1
    assert stats["escalation_rate"] == 0.0


def test_escalation_rate_window(metrics_mod):
    metrics_mod.ESCALATION_RATE_WINDOW = 5
    for _ in range(5):
        _log(metrics_mod, "general", escalated=False)
    for _ in range(5):
        _log(metrics_mod, "general", escalated=True)
    stats = metrics_mod.category_escalation_rate("general")
    assert stats["sample_size"] == 5
    assert stats["escalation_rate"] == 1.0  # only the 5 most recent (all escalated)


def test_eval_baseline_roundtrip(metrics_mod):
    bid = metrics_mod.save_eval_baseline("task", "expected", "abc123")
    metrics_mod.log_eval_result(bid, True, "looks right", "actual")
    baselines = metrics_mod.all_baselines()
    results = metrics_mod.all_eval_results()
    assert baselines[0]["task"] == "task"
    assert results[0]["passed"] == 1
