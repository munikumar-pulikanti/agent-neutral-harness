# Design

The reasoning behind the harder decisions in this codebase, pulled together
in one place instead of scattered across docstrings. If you're evaluating
whether the architecture holds up, start here.

## The one constraint everything else follows from

You keep your agent loop. The harness never owns tool execution, model
selection at the call site, or credentials — every entry point takes the
model name and an `execute_fn(model, task) -> dict` as plain parameters.
That's not a purity rule for its own sake: it's what makes the same
cascade/reliability/memory logic usable from a raw tool loop, LangGraph, or
an IDE assistant without adapting to any of their internals. The cost is
real — the harness can't optimize across the boundary it doesn't own — and
is accepted deliberately (see [`docs/owasp-agentic-mapping.md`](docs/owasp-agentic-mapping.md)
for where that boundary puts real risk categories out of this project's
scope, not swept under it).

## Cascade: escalate on observed failure, never a prediction

`run_cascade` runs the cheap model, runs `reliability.run_all_checks` on its
**real** output, and escalates only if a check actually fails. There's no
confidence threshold on the model's self-report — a model claiming
certainty is not evidence of correctness, and building a threshold around
that self-report would just be trusting the thing you're trying to verify.

Two decisions make the cost-saving shortcut trustworthy instead of just fast:

- **Wilson lower bound, not the raw rate.** A cheap model's true escalation
  rate of ~65% can show 80%+ in a small sample by chance. Comparing the
  shortcut threshold against the 95% Wilson lower bound instead of the raw
  fraction means a lucky streak can't flip the shortcut on — you need to be
  genuinely confident the rate clears the bar, not just have gotten unlucky
  with your last 20 calls.
- **`config_fingerprint` scoping.** A model digest, system prompt, and tool
  schema get hashed into one fingerprint. Escalation-rate history is scoped
  to it, so a model swap, a prompt edit, or a tool signature change drops
  stale rows out of the window automatically — the alternative was dragging
  a routing decision for ~50 turns on data that no longer describes the
  current setup.

**A senior AI engineer reviewing this asked a sharp question during that
work: is the drift window sized by turn count or wall clock?** It matters —
a busy day and a quiet week shouldn't be compared on the same footing. The
answer landed on turn count (`ESCALATION_RATE_WINDOW = 50`), which
self-normalizes for volume, but that answer was incomplete on its own: a
low-traffic category can take weeks to fill 50 turns, and at that point the
"older half" of the drift comparison is a weeks-old baseline being compared
against yesterday. `detect_within_window_drift` now reports
`window_span_seconds` alongside the drift verdict — so the comparison's own
staleness is visible, not just the number — and accepts a `max_age_seconds`
bound so a caller can refuse to compare against a baseline that's too old to
mean anything, which then honestly reports `insufficient_data` rather than a
number nobody should trust.

Tool *description prose* is deliberately excluded from the fingerprint hash
— hashing it caused phantom invalidations on pure typo fixes. The bet: a
tool's name/parameter *structure* is what actually shifts a model's success
rate; a meaningful prose rewrite that keeps the same structure is instead
caught by drift detection, which measures behavior directly rather than
guessing from the diff.

## Memory: evidence-gated confidence, not repetition-gated confidence

The lifecycle is `hypothesis → suspected → confirmed`, and the gate on the
last step is deliberately strict: `confirmed` requires **both** three or
more corroborations **and** a verified, reachable evidence URL. Repetition
alone — a claim restated five, ten, fifty times — caps permanently at
`suspected`. This is the design's actual thesis about memory systems: an
agent's confidence in its own repeated claim is not evidence, and a memory
system that treats "restated often" as "probably true" is building
consensus out of an echo chamber, not verification. See
[`docs/owasp-agentic-mapping.md`](docs/owasp-agentic-mapping.md) (ASI06,
ASI09) for how this maps to memory-poisoning and trust-exploitation risk
categories directly.

The corroboration match itself uses a similarity **band**, not a single
threshold: above `CORROBORATION_THRESHOLD` (0.85) is treated as the same
finding restated; between 0.5 and 0.85 is genuinely ambiguous — related, but
unclear whether it agrees or conflicts — and lands in a `needs_review` queue
instead of the system silently guessing either way. That queue is a
deliberate admission that some disagreements need a human, not an algorithm
picking a side.

The same principle carries into the warm tier: sync identity is a
**content hash** (`sha256(scope|type|content)`), not a local row id.
Two machines independently using local id `1` for different content used
to be a real collision risk under naive id-based sync; hashing the content
itself makes the sync key stable across machines regardless of local
numbering. A genuine divergent edit of "the same" memory on two machines
produces two different hashes, and `pull()` runs every incoming row back
through `save_memory` — so that divergence surfaces through the same
corroboration/review-band machinery as any other ambiguous match. The
review queue **is** the conflict log; there's no separate conflict-resolution
system to keep in sync with the confidence lifecycle.

## Reliability assertions: deterministic only, never another model call

Every check in `reliability.assertions` runs on the agent's *actual* output
and, where relevant, the *actual* tool output it received — never a model
judging another model's work. That's a scope choice, not an oversight: an
LLM-judge check belongs in `evals.runner` (where the judge is the explicit
point, and its own failure mode — "unparseable judge output" — is handled
as a graded failure, not a crash). Mixing the two would make it unclear
which failures are "the agent messed up" versus "the judge is unreliable
today." Keeping deterministic checks deterministic is what makes them cheap
enough to run on every turn without another API call, and honest about what
they can and can't catch — see `docs/owasp-agentic-mapping.md` for the
category (ASI01, goal hijack) this scope choice explicitly does not cover.

## What's deliberately not here

`hearthagent-pro` (the local-first coding agent this harness was distilled
from) has a LangGraph wiring, a `run_shell` allowlist/injection guard, web
search, voice input, and a Flask UI. None of that ported over, on purpose —
those are application-scope decisions a specific agent runtime makes, not
things a reliability/memory/routing harness should have an opinion about.
Everything here stays useful to a raw tool loop, LangGraph, or anything
else precisely because it doesn't assume any of them.
