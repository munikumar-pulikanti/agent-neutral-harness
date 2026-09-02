# Contributing

Thanks for taking a look. This is a small, focused project — issues and PRs welcome.

## Setup

```bash
uv sync --group dev --extra all
uv run pytest
uv run ruff check .
```

## Before opening a PR

- `uv run ruff check .` is clean.
- `uv run pytest` is green.
- New behaviour has a test. Tests must not hit the network or a real model —
  inject a `generate_fn` / `execute_fn` stub (see `tests/` for the pattern).
- If you change what's built vs designed, update the status table in `README.md`
  **and** `HANDOFF.md`.

## Scope

The harness stays agent-neutral: no hard dependency on a specific agent framework
or model provider in the core. Model names and runtimes are always parameters.
Anything heavier than `requests` goes behind an optional-dependency extra.

## Reporting bugs

Include the failing input, what you expected, and what happened. A reproducing
test case is the fastest path to a fix.
