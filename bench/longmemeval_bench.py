"""Scoped LongMemEval (oracle split) benchmark against agent-neutral-harness's
MemoryVault. NOT part of the installable package -- a standalone, reproducible
script behind the number quoted in README.md / DESIGN.md.

Requires: pip install "agent-neutral-harness[memory]" huggingface_hub
Also requires a local Ollama with `nomic-embed-text` (or your embedding
model of choice already configured) and `llama3.2:1b` pulled.

Run: python bench/longmemeval_bench.py

Methodology:
- Dataset: xiaowu0162/longmemeval on HuggingFace, "oracle" split (the
  benchmark's own official evaluation mode: haystack_sessions contains only
  the ground-truth-relevant sessions for each question, not the full
  multi-thousand-turn distractor haystack -- isolates retrieval-given-
  relevant-context from needle-in-haystack retrieval at full scale).
  Downloaded via huggingface_hub, not a hardcoded local path, so this is
  actually runnable by someone else, not just reproducible-in-theory.
- Sample: 2 questions per question_type x 6 types = 12 questions (oracle
  split has 500 total across 6 types; this is a scoped sample chosen for
  time budget, not the full set -- treat the headline number accordingly).
- For each question: a FRESH MemoryVault (isolated temp dir, real semantic
  search via the [memory] extra) ingests every turn (user + assistant) of
  its haystack_sessions via save_memory(), one memory per turn.
- Retrieval: vault.search_semantic(question).
- Grading: agent_neutral_harness.evals.runner.run_eval_judge (already in the
  repo) with llama3.2:1b as judge -- swapped down from the originally
  planned llama3.1:8b for CPU-inference speed. This is a materially weaker
  grader than the original plan; a re-run with a stronger judge would give
  a tighter number, this one should be read as directional, not final.
"""

import json
import random
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from agent_neutral_harness.evals.runner import run_eval_judge
from agent_neutral_harness.memory.vault import MemoryVault

PER_TYPE = 2
SEED = 42


def _data_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id="xiaowu0162/longmemeval",
            filename="longmemeval_oracle",
            repo_type="dataset",
        )
    )


def sample_questions():
    data = json.loads(_data_path().read_text())
    by_type = {}
    for d in data:
        by_type.setdefault(d["question_type"], []).append(d)
    rng = random.Random(SEED)
    sampled = []
    for _qtype, items in sorted(by_type.items()):
        rng.shuffle(items)
        sampled.extend(items[:PER_TYPE])
    return sampled


def run_one(example: dict) -> dict:
    tmpdir = tempfile.mkdtemp(prefix="anh-bench-")
    try:
        vault = MemoryVault(
            db_path=str(Path(tmpdir) / "bench.db"),
            chroma_path=str(Path(tmpdir) / "chroma"),
        )
        turn_count = 0
        for session in example["haystack_sessions"]:
            for turn in session:
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                vault.save_memory(
                    scope="bench",
                    mem_type="fact" if turn["role"] == "user" else "note",
                    content=content[:2000],
                )
                turn_count += 1

        retrieved = vault.search_semantic(example["question"], top=5)
        verdict = run_eval_judge(
            task=example["question"],
            expected_answer=example["answer"],
            actual_answer=retrieved,
            judge_model="llama3.2:1b",
            timeout=180.0,
        )
        return {
            "question_id": example["question_id"],
            "question_type": example["question_type"],
            "question": example["question"],
            "answer": example["answer"],
            "retrieved": retrieved,
            "turn_count": turn_count,
            "passed": verdict["passed"],
            "reasoning": verdict["reasoning"],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    questions = sample_questions()
    print(f"Sampled {len(questions)} questions across "
          f"{len(set(q['question_type'] for q in questions))} types "
          f"(seed={SEED}, {PER_TYPE}/type)")

    results = []
    start = time.time()
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        r = run_one(q)
        dt = time.time() - t0
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{i}/{len(questions)}] {status} ({dt:.1f}s, {r['turn_count']} turns) "
              f"{r['question_type']}: {r['question'][:70]}")

    total_dt = time.time() - start
    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== {passed}/{len(results)} passed ({100*passed/len(results):.0f}%) "
          f"in {total_dt:.0f}s ===")

    by_type = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["passed"])
    for qtype, flags in sorted(by_type.items()):
        print(f"  {qtype}: {sum(flags)}/{len(flags)}")

    out_path = Path(__file__).with_name("longmemeval_bench_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
