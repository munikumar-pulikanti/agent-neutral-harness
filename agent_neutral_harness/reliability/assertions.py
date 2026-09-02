"""Deterministic reliability assertions for validating agent responses.

These checks are intentionally cheap and deterministic: they run on the
*real* output an agent produced (and, where relevant, the *real* tool
output it received) and never call a model. They exist to catch a set of
concrete failure modes observed during real agent development, not
hypothetical ones. See ``run_all_checks`` for the aggregated entry point.
"""

import logging
import re

log = logging.getLogger(__name__)

# Tools whose invocation counts as "the agent actually verified something".
# Deployment-specific tool names can be added via ``run_all_checks``.
DEFAULT_VERIFICATION_TOOLS = frozenset(
    {"run_shell", "execute_test", "pytest", "run_command", "bash", "shell", "exec"}
)

_SUCCESS_PHRASES = (
    "tests passed",
    "all tests are passing",
    "all tests pass",
    "successfully verified",
    "build succeeded",
    "build passed",
)

_TOOL_CALL_LEAK_PATTERNS = (
    re.compile(r'\{\s*"name"\s*:\s*".+?"\s*,\s*"(?:parameters|arguments)"\s*:\s*[{\[]', re.DOTALL),
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL),
    re.compile(r"\[TOOL_CALL:.*?\]", re.DOTALL),
)

# A filename-ish token: at least one name char, a dot, then a 1-8 char
# extension that contains at least one letter (so "2.31.0" / "v1.2" are
# not treated as files). Trailing punctuation is trimmed by the caller.
_FILENAME_RE = re.compile(r"\b[\w][\w\-]*(?:\.[\w\-]+)*\.(?=[\w-]*[a-zA-Z])[\w-]{1,8}\b")


def check_tool_call_leak(response_text: str) -> bool:
    """True if raw tool-calling JSON/markup leaked into plain response text.

    This is the "the model printed the tool call instead of invoking it"
    failure -- the user sees machinery, and the intended action never ran.
    """
    if not response_text:
        return False
    return any(p.search(response_text) for p in _TOOL_CALL_LEAK_PATTERNS)


def check_empty_response(response_text: str) -> bool:
    """True if the agent produced an empty or whitespace-only response."""
    return not response_text or not response_text.strip()


def check_unverified_success_claim(
    response_text: str,
    tools_invoked: list[str],
    verification_tools: frozenset | None = None,
) -> bool:
    """True if the response claims success but no verification tool ran.

    Catches "tests passed" / "build succeeded" narration when nothing was
    actually executed to justify it.
    """
    if not response_text:
        return False
    verification_tools = verification_tools or DEFAULT_VERIFICATION_TOOLS
    lowered = response_text.lower()
    claims_success = any(phrase in lowered for phrase in _SUCCESS_PHRASES)
    if not claims_success:
        return False
    has_verified = any(t in verification_tools for t in (tools_invoked or []))
    return not has_verified


def _candidate_tokens(text: str) -> set:
    """Filename-like tokens mentioned in a narrative summary."""
    tokens = set()
    for match in _FILENAME_RE.finditer(text):
        tokens.add(match.group(0).strip(".,;:!?)('\"").lower())
    return {t for t in tokens if t}


def check_tool_result_fidelity(response_text: str, tool_results: dict[str, str]) -> bool:
    """True if the summary names concrete items absent from the real tool output.

    Observed for real: a model executes a tool call correctly, receives
    correct data back, then invents *additional* filenames/entries in its
    own prose summary of that real result. "Did the tool run" says nothing
    about whether the summary of its result stayed honest.

    ``tool_results`` maps tool name -> that tool's real output text, for
    tools that return concrete enumerable items (file listings, search
    results, memory entries, query rows). The check is deliberately
    conservative: a token counts as fabricated only if it appears nowhere
    in the real output, as a substring, case-insensitively.
    """
    if not response_text or not tool_results:
        return False

    mentioned = _candidate_tokens(response_text)
    if not mentioned:
        return False

    combined = "\n".join(v for v in tool_results.values() if v).lower()
    if not combined:
        return False
    if "no matches" in combined or "no results" in combined or "not found" in combined:
        # The tool reported nothing; the summary having nothing to match
        # against is not evidence of fabrication.
        return False

    fabricated = [tok for tok in mentioned if tok not in combined]
    if fabricated:
        log.debug("fidelity check: tokens absent from tool output: %s", fabricated)
        return True
    return False


def run_all_checks(
    response_text: str,
    tools_invoked: list[str],
    tool_results: dict[str, str] | None = None,
    verification_tools: frozenset | None = None,
) -> list[str]:
    """Run every reliability check, returning a list of flag strings.

    An empty list means the response passed all deterministic checks. The
    flags feed cascade escalation decisions and per-turn metrics logging.
    """
    tool_results = tool_results or {}
    flags: list[str] = []
    if check_tool_call_leak(response_text):
        flags.append("unexecuted_tool_call_leaked_to_user")
    if check_empty_response(response_text):
        flags.append("empty_response")
    if check_unverified_success_claim(response_text, tools_invoked, verification_tools):
        flags.append("claimed_success_without_real_verification")
    if check_tool_result_fidelity(response_text, tool_results):
        flags.append("fabricated_items_in_summary")
    return flags
