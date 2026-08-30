"""Deterministic reliability assertions for validating agent responses."""

import re
from typing import List, Dict, Any


def check_tool_call_leak(response_text: str) -> bool:
    """Checks if raw tool-calling JSON leaked into plain response text."""
    patterns = [
        r'\{\s*"name"\s*:\s*".*?"\s*,\s*"parameters"\s*:\s*\{',
        r'<tool_call>.*?</tool_call>',
        r'\[TOOL_CALL:.*?\]'
    ]
    return any(re.search(p, response_text, re.DOTALL) for p in patterns)


def check_empty_response(response_text: str) -> bool:
    """Checks if the agent generated an empty or whitespace-only response."""
    return not response_text or not response_text.strip()


def check_unverified_success_claim(response_text: str, tools_invoked: List[str]) -> bool:
    """Catches claims of 'tests passed' or 'implemented' when no verification tool was executed."""
    success_phrases = ["tests passed", "all tests are passing", "successfully verified", "build succeeded"]
    claims_success = any(phrase in response_text.lower() for phrase in success_phrases)
    verification_tools = {"run_shell", "execute_test", "pytest", "run_command", "bash"}
    has_verified = any(t in verification_tools for t in tools_invoked)
    return claims_success and not has_verified