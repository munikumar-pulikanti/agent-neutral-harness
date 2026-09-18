# Mapping to the OWASP Top 10 for Agentic Applications (2026)

OWASP's [GenAI Security Project published the Top 10 for Agentic
Applications](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
(ASI01–ASI10) on 2025-12-09 — the first peer-reviewed risk taxonomy specific
to agents that plan, hold memory, call tools, and act with delegated
authority, rather than the LLM-prompt-focused OWASP GenAI/LLM Top 10.

This harness is a reliability/memory/routing layer, not an agent runtime —
it doesn't own tool execution, credentials, or inter-agent transport (see
[README](../README.md), "not own your agent loop"). So this mapping is
honest about which categories it actually addresses versus which are the
wrapping runtime's responsibility. Overclaiming coverage here would be worse
than no mapping at all.

| Risk | Addressed here? | How |
| --- | --- | --- |
| **ASI01** Agent Goal Hijack | **No — gap.** | `reliability.assertions` checks *output*, not the reasoning process that produced it. A hijacked agent whose final output still passes every check (no leaked tool call, no unverified success claim, no fabricated summary item) is not caught. See `reliability/assertions.py` — an injection-pattern check is a real candidate addition, not yet built. |
| **ASI02** Tool Misuse & Exploitation | Partial | `check_unverified_success_claim` requires a real verification tool to have run before a "tests passed"-style claim is trusted; `check_tool_result_fidelity` catches a summary inventing items absent from the tool's real output. Neither validates *what arguments* a tool was actually called with. |
| **ASI03** Agent Identity & Privilege Abuse | Out of scope by design | No credential or privilege model — that belongs to `execute_fn`'s runtime, which the harness deliberately never owns. |
| **ASI04** Agentic Supply Chain Compromise | Partial, by design | Core install is `requests` only; every heavier dependency (chromadb, mcp, libsql-experimental, boto3, streamlit) is an opt-in extra, and importing the package never eagerly pulls one in (`__init__.py`'s lazy `__getattr__`). Smaller default surface, not active scanning. |
| **ASI05** Unexpected Code Execution | Out of scope by design | The harness never executes code or tools itself; that's `execute_fn`'s job. |
| **ASI06** Memory & Context Poisoning | **Yes — a real strength.** | `memory.vault`'s confidence lifecycle: a new claim starts at `hypothesis`; promotion past `suspected` requires **both** ≥3 corroborations **and** a verified, reachable evidence URL (SSRF-guarded — see [SECURITY.md](../SECURITY.md)); an ambiguous near-duplicate lands in a `needs_review` queue instead of silently merging or silently duplicating. A poisoned or repeated-but-unverified claim is structurally incapable of reaching `confirmed`. |
| **ASI07** Insecure Inter-Agent Communication | Out of scope, currently | The `mcp` extra shares memory across tools on one machine; there's no A2A (agent-to-agent) transport. A2A reached Linux Foundation v1.0 and production use across multiple industries in 2026 — worth a real look as a follow-up, not implemented yet (see `HANDOFF.md`). |
| **ASI08** Cascading Agent Failures | **Yes — the core design.** | `cascade.run_cascade` escalates only on an *observed* deterministic-check failure, never a predicted one; the Wilson-lower-bound-gated shortcut and `config_fingerprint` scoping exist specifically so one bad signal (a lucky small sample, a stale model swap) can't silently compound into a wrong routing decision across hundreds of turns. |
| **ASI09** Human-Agent Trust Exploitation | Partial | The evidence-gating in ASI06 applies here too: a memory can't reach `confirmed` just because it's been *restated* confidently or often — corroboration count alone caps at `suspected`. Doesn't address a human being persuaded by an agent's *live* output, only whether the harness's own memory layer will treat repetition as proof. |
| **ASI10** Rogue Agents | **Yes.** | `metrics.detect_within_window_drift` splits a category's recent escalation-rate window into older/newer halves and flags a rate jump even when `config_fingerprint` hasn't changed — the case where behavior drifted (a model update, a quiet prompt regression) but nothing declared that it should have. Flag only, never auto-changes routing; surfaced per-category in the dashboard. |

## Honest summary

Three categories (memory poisoning, cascading failures, rogue-agent drift)
are squarely what this harness is built for, and the design predates this
mapping — it isn't retrofitted to look good against it. Three (identity,
code execution, inter-agent transport) are explicitly not this project's
job. The rest are partial, and ASI01 (goal hijack) is the most honest gap:
nothing here currently distinguishes "the agent was manipulated into a bad
plan" from "the agent's own output happens to be clean" — that's a real
direction for future work, not a solved problem dressed up as one.
