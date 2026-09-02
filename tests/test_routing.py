from agent_neutral_harness.routing.classifier import classify_task
from agent_neutral_harness.routing.curator import curate_memories


def test_classify_task_with_injected_generate_fn():
    assert classify_task("fix the flaky test", generate_fn=lambda p: "unit_tests") == "unit_tests"
    assert classify_task("x", generate_fn=lambda p: "Category: implement.") == "implement"
    assert classify_task("x", generate_fn=lambda p: "unit tests") == "unit_tests"


def test_classify_task_unknown_falls_back_to_general():
    assert classify_task("x", generate_fn=lambda p: "banana") == "general"


def test_classify_task_error_falls_back_to_general():
    def boom(_):
        raise RuntimeError("no ollama")

    assert classify_task("x", generate_fn=boom) == "general"


def test_curate_passes_through_no_results():
    txt = "No matching memories found (checked keyword FTS)."
    assert curate_memories("task", txt, generate_fn=lambda p: "should not be called") == txt


def test_curate_returns_empty_on_none():
    out = curate_memories("task", "[global] some memory", generate_fn=lambda p: "NONE")
    assert out == ""


def test_curate_fails_open_on_error():
    raw = "[global] some memory"

    def boom(_):
        raise RuntimeError("down")

    assert curate_memories("task", raw, generate_fn=boom) == raw
