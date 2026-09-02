from agent_neutral_harness.reliability import assertions as a


def test_tool_call_leak_detects_json_and_markup():
    assert a.check_tool_call_leak('Sure: {"name": "run", "parameters": {"x": 1}}')
    assert a.check_tool_call_leak("<tool_call>do_thing()</tool_call>")
    assert not a.check_tool_call_leak("I ran the tool and it worked.")
    assert not a.check_tool_call_leak("")


def test_empty_response():
    assert a.check_empty_response("")
    assert a.check_empty_response("   \n\t")
    assert not a.check_empty_response("done")


def test_unverified_success_claim():
    assert a.check_unverified_success_claim("All tests passed.", tools_invoked=[])
    assert not a.check_unverified_success_claim("All tests passed.", tools_invoked=["pytest"])
    assert not a.check_unverified_success_claim("Working on it.", tools_invoked=[])


def test_fidelity_flags_fabricated_filename():
    resp = "I found config.py and secrets.py in the directory."
    real = {"list_dir": "config.py\nutils.py"}
    assert a.check_tool_result_fidelity(resp, real)


def test_fidelity_passes_when_all_files_are_real():
    resp = "The directory contains config.py and utils.py."
    real = {"list_dir": "config.py\nutils.py\nREADME.md"}
    assert not a.check_tool_result_fidelity(resp, real)


def test_fidelity_ignores_version_numbers():
    # regression: "2.31.0" must not be treated as a fabricated filename
    resp = "requests 2.31.0 is installed and the run succeeded."
    real = {"pip": "requests 2.31.0"}
    assert not a.check_tool_result_fidelity(resp, real)


def test_fidelity_handles_filenames_starting_with_f_or_d():
    # regression: the old lstrip("fd") mangled these
    resp = "Edited foo.py and data.json."
    real = {"ls": "foo.py\ndata.json"}
    assert not a.check_tool_result_fidelity(resp, real)


def test_fidelity_skips_empty_tool_output():
    assert not a.check_tool_result_fidelity("mentions thing.py", {"search": "no results found"})


def test_run_all_checks_aggregates_flags():
    flags = a.run_all_checks("", tools_invoked=[])
    assert "empty_response" in flags
    assert a.run_all_checks("looks good", tools_invoked=[]) == []
