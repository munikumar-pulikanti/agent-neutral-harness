# Examples

Each file is self-contained and points its metrics DB at a throwaway temp file,
so running one never touches your real `~/.agent-neutral-harness` state.

| File | Needs | Shows |
|---|---|---|
| [`quickstart.py`](quickstart.py) | nothing (core install only) | The whole flow with a faked `execute_fn` — classify, run cheap, catch a fabricated filename, escalate. Runs anywhere. |
| [`ollama_cascade.py`](ollama_cascade.py) | Ollama + 3 pulled models | The same flow with `execute_fn` actually calling Ollama — the cascade verifying a *real* model response. |
| [`langgraph_cascade.py`](langgraph_cascade.py) | `langchain`, `langchain-ollama`, Ollama | The one integration point: wrapping a compiled LangChain/LangGraph agent as `execute_fn`. The harness never imports LangGraph. |

```bash
# from the repo root
python examples/quickstart.py

ollama pull llama3.2:1b qwen2.5-coder:3b qwen2.5-coder:7b
python examples/ollama_cascade.py

pip install langchain langchain-ollama
python examples/langgraph_cascade.py
```
