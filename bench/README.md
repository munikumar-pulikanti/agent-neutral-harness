# Benchmarks

Standalone scripts that produce the numbers quoted in the main README and
`DESIGN.md`. Not part of the installable package (no `agent_neutral_harness`
code depends on anything here) -- these exist so a claim in the docs is
reproducible, not just asserted.

## `longmemeval_bench.py`

Real [LongMemEval](https://arxiv.org/abs/2410.10813) run (oracle split)
against `MemoryVault`. See the script's own docstring for full methodology
(sample size, judge model, what "oracle split" means here) -- it's
deliberately kept there, next to the code that implements it, rather than
duplicated in this file.

```bash
pip install "agent-neutral-harness[memory]" huggingface_hub
# Ollama running locally with nomic-embed-text and llama3.2:1b pulled
python bench/longmemeval_bench.py
```

`longmemeval_bench_results.json` is the full per-question output from the
run actually cited in the docs (question, official answer, what the vault
retrieved, judge verdict and reasoning) -- committed so the headline number
isn't the only thing a reader can check.
