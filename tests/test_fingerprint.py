from agent_neutral_harness.fingerprint import _tool_signature, config_fingerprint


def test_fingerprint_stable_and_sensitive():
    a = config_fingerprint(model_digest="abc", system_prompt="you are helpful", tools=[])
    assert a == config_fingerprint(model_digest="abc", system_prompt="you are helpful", tools=[])
    assert a != config_fingerprint(model_digest="DEF", system_prompt="you are helpful", tools=[])
    assert a != config_fingerprint(model_digest="abc", system_prompt="you are terse", tools=[])
    assert len(a) == 16


def test_fingerprint_normalizes_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prompt = f"work in {tmp_path}"
    fp1 = config_fingerprint(system_prompt=prompt)
    monkeypatch.chdir(tmp_path.parent)
    # same logical prompt, different cwd baked in -> still same fingerprint
    fp2 = config_fingerprint(system_prompt=f"work in {tmp_path.parent}")
    assert fp1 == fp2


def test_tool_signature_ignores_description_prose():
    tool_a = {"name": "run", "parameters": {"cmd": {"type": "string", "description": "the command"}}}
    tool_b = {"name": "run", "parameters": {"cmd": {"type": "string", "description": "REWORDED"}}}
    tool_c = {"name": "run", "parameters": {"cmd": {"type": "string"}, "cwd": {"type": "string"}}}
    assert _tool_signature([tool_a]) == _tool_signature([tool_b])
    assert _tool_signature([tool_a]) != _tool_signature([tool_c])
