from agent_neutral_harness.metrics import _wilson_lower_bound


def _log(m, category, escalated, cheap_tokens=10):
    m.log_turn(
        task_snippet="t", category=category, model="x", duration_seconds=0.1,
        escalated=escalated, cheap_attempt_tokens=cheap_tokens,
    )


def test_escalation_rate_empty(metrics_mod):
    stats = metrics_mod.category_escalation_rate("implement")
    assert stats["sample_size"] == 0
    assert stats["escalation_rate"] is None
    assert stats["escalation_rate_lower_bound"] is None


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


def test_wilson_lower_bound_is_below_point_estimate():
    lb = _wilson_lower_bound
    assert lb(0, 0) == 0.0
    # 18/20 successes: raw 0.9, lower bound noticeably less
    assert 0.6 < lb(18, 20) < 0.9
    # a big confident sample sits close to the point estimate
    assert lb(900, 1000) > 0.87


def test_escalation_rate_filtered_by_config_fingerprint(metrics_mod):
    for _ in range(4):
        metrics_mod.log_turn(task_snippet="t", category="implement", model="x",
                             duration_seconds=0.1, escalated=True,
                             cheap_attempt_tokens=1, config_fingerprint="OLD")
    for _ in range(2):
        metrics_mod.log_turn(task_snippet="t", category="implement", model="x",
                             duration_seconds=0.1, escalated=False,
                             cheap_attempt_tokens=1, config_fingerprint="NEW")
    stale = metrics_mod.category_escalation_rate("implement", config_fingerprint="OLD")
    fresh = metrics_mod.category_escalation_rate("implement", config_fingerprint="NEW")
    assert stale["sample_size"] == 4 and stale["escalation_rate"] == 1.0
    assert fresh["sample_size"] == 2 and fresh["escalation_rate"] == 0.0
    # no fingerprint filter -> all rows
    assert metrics_mod.category_escalation_rate("implement")["sample_size"] == 6


def test_detect_within_window_drift(metrics_mod):
    # older half (logged first) mostly not escalated, newer half all escalated
    for _ in range(10):
        _log(metrics_mod, "implement", escalated=False)
    for _ in range(10):
        _log(metrics_mod, "implement", escalated=True)
    drift = metrics_mod.detect_within_window_drift("implement")
    assert drift["drift_detected"] is True
    assert drift["drift_magnitude"] >= 0.3
    assert "window_span_seconds" in drift


def test_detect_within_window_drift_insufficient_data(metrics_mod):
    for _ in range(6):
        _log(metrics_mod, "implement", escalated=True)
    result = metrics_mod.detect_within_window_drift("implement")
    assert result["reason"] == "insufficient_data"
    assert result["window_span_seconds"] == 0


def test_detect_within_window_drift_max_age_excludes_stale_rows(metrics_mod):
    for _ in range(10):
        _log(metrics_mod, "implement", escalated=False)
    for _ in range(10):
        _log(metrics_mod, "implement", escalated=True)
    # backdate every logged turn well past the age cutoff
    with metrics_mod._connect() as conn:
        conn.execute("UPDATE turns SET timestamp = timestamp - 100000")
    result = metrics_mod.detect_within_window_drift("implement", max_age_seconds=3600)
    assert result["reason"] == "insufficient_data"


def test_breakdowns(metrics_mod):
    _log(metrics_mod, "implement", escalated=True)
    _log(metrics_mod, "general", escalated=False)
    assert metrics_mod.category_breakdown()["implement"] == 1
    assert set(metrics_mod.memory_tier_breakdown()) == {"hot", "warm", "cold", "none"}


def test_eval_baseline_roundtrip(metrics_mod):
    bid = metrics_mod.save_eval_baseline("task", "expected", "abc123")
    metrics_mod.log_eval_result(bid, True, "looks right", "actual")
    baselines = metrics_mod.all_baselines()
    results = metrics_mod.all_eval_results()
    assert baselines[0]["task"] == "task"
    assert results[0]["passed"] == 1
