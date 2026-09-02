"""Streamlit dashboard for the metrics DB -- routing, reliability, cost, evals.

Run it with either::

    agent-neutral-dashboard
    streamlit run -m agent_neutral_harness.dashboard

Requires the ``dashboard`` extra: pip install "agent-neutral-harness[dashboard]"
"""

import sys


def _render():
    import pandas as pd
    import streamlit as st

    from agent_neutral_harness import metrics

    st.set_page_config(page_title="agent-neutral-harness", layout="wide")
    st.title("agent-neutral-harness — reliability & usage")
    st.caption("Real per-turn data from the local metrics DB.")

    turns = metrics.all_turns()
    if not turns:
        st.warning("No turns logged yet. Run some cascades, then refresh.")
    else:
        df = pd.DataFrame(turns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        total_tok = int(df["input_tokens"].sum() + df["output_tokens"].sum())
        esc_rate = df["escalated"].mean() * 100 if "escalated" in df else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Turns", len(df))
        c2.metric("Tokens", f"{total_tok:,}")
        c3.metric("Avg latency", f"{df['duration_seconds'].mean():.1f}s")
        c4.metric("Escalation rate", f"{esc_rate:.0f}%")

        g1, g2, g3 = st.columns(3)
        g1.subheader("Category")
        g1.bar_chart(pd.Series(metrics.category_breakdown()))
        g2.subheader("Model")
        g2.bar_chart(pd.Series(metrics.model_breakdown()))
        g3.subheader("Memory tier")
        g3.bar_chart(pd.Series(metrics.memory_tier_breakdown()))

        st.subheader("Assertion flags")
        flagged = df[df["assertion_flags"].fillna("") != ""]
        if flagged.empty:
            st.info("No assertion flags triggered.")
        else:
            flags: list[str] = []
            for f in flagged["assertion_flags"]:
                flags.extend(part for part in str(f).replace(";", ",").split(",") if part)
            st.bar_chart(pd.Series(flags).value_counts())
            with st.expander("Flagged turns"):
                st.dataframe(
                    flagged[["task_snippet", "category", "model", "assertion_flags"]],
                    use_container_width=True,
                )

        st.subheader("Recent turns")
        st.dataframe(
            df[["timestamp", "category", "model", "cascade_tier", "escalated",
                "duration_seconds", "assertion_flags"]]
            .sort_values("timestamp", ascending=False).head(25),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Eval results (LLM-judged)")
    results = metrics.all_eval_results()
    if not results:
        st.info("No evals run yet.")
    else:
        edf = pd.DataFrame(results)
        st.metric("Pass rate", f"{edf['passed'].mean() * 100:.0f}% "
                               f"({int(edf['passed'].sum())}/{len(edf)})")
        st.dataframe(edf[["task", "passed", "judge_reasoning"]], use_container_width=True)


def main():
    """Console-script entry point: (re-)launch under ``streamlit run``."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit("The dashboard needs: pip install 'agent-neutral-harness[dashboard]'")
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    _render()
