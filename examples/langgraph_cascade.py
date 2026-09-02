"""Wrap a real LangGraph agent as an ``execute_fn`` for the cascade.

    pip install "agent-neutral-harness" langchain langchain-ollama
    ollama pull llama3.2:1b qwen2.5-coder:3b qwen2.5-coder:7b
    python examples/langgraph_cascade.py

The harness never sees LangGraph. The only integration point is the adapter
below: take whatever your compiled graph returns and map it onto the result
dict the cascade expects. Everything the cascade does — verify the real output,
escalate on an observed failure, learn the per-category escalation rate — works
the same whether the runtime underneath is a graph, a raw loop, or an IDE agent.
"""

import os
import tempfile

os.environ["AGENT_NEUTRAL_HARNESS_METRICS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="anh-langgraph-"), "metrics.db"
)

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

from agent_neutral_harness import metrics, run_cascade  # noqa: E402
from agent_neutral_harness.routing.classifier import classify_task  # noqa: E402


@tool
def run_shell(command: str) -> str:
    """Run a shell command and return its output. (Stubbed for the example.)"""
    return f"$ {command}\n3 passed in 0.12s"


def build_graph(model: str):
    return create_agent(ChatOllama(model=model, temperature=0), [run_shell])


def make_execute_fn():
    """Return an execute_fn(model, task) that runs a LangChain/LangGraph agent.

    The adapter: the agent gives back a list of messages; the cascade wants
    final_content + which tools ran + their real output. Pull tool names and
    ToolMessage contents out of the message history so the reliability checks
    can compare the summary against what the tools actually returned.
    """

    def execute_fn(model: str, task: str) -> dict:
        graph = build_graph(model)
        try:
            state = graph.invoke({"messages": [HumanMessage(content=task)]})
        except Exception as exc:  # noqa: BLE001
            return {"error": f"graph_invoke_failed: {exc}"}

        messages = state["messages"]
        tools_invoked, tool_results = [], {}
        for m in messages:
            if m.__class__.__name__ == "ToolMessage":
                name = getattr(m, "name", "tool")
                tools_invoked.append(name)
                tool_results[name] = str(m.content)

        return {
            "final_content": messages[-1].content,
            "tool_call_count": len(tools_invoked),
            "tools_invoked": tools_invoked,
            "tool_results": tool_results,
            "input_tokens": 0,   # fill from response_metadata if you need cost data
            "output_tokens": 0,
            "error": None,
        }

    return execute_fn


def main() -> None:
    task = "Run the test suite with `pytest -q` and tell me if it is green."

    category = classify_task(task, model="llama3.2:1b")
    print(f"category: {category}")

    answer = run_cascade(
        task=task,
        category=category,
        cheap_model="qwen2.5-coder:3b",
        capable_model="qwen2.5-coder:7b",
        execute_fn=make_execute_fn(),
        random_fn=lambda: 0.0,
    )

    turn = metrics.all_turns()[0]
    print(f"tier used      : {turn['cascade_tier']}")
    print(f"escalated      : {bool(turn['escalated'])}")
    print(f"assertion_flags: {turn['assertion_flags'] or '(none)'}")
    print(f"\n--- answer ---\n{answer[:600]}")


if __name__ == "__main__":
    main()
