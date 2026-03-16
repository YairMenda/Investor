"""
Streamlit UI for the Investor AI Research Agent.

Features:
- Sidebar: company dropdown, filing type filter, user profile
- Chat interface with streaming support
- Agent Reasoning expander showing intermediate steps
"""

import json

import streamlit as st

from agents.graph import graph
from agents.state import initial_state
from config.companies import MAG7_COMPANIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investor — SEC Filing AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Investor")
    st.caption("AI Research Agent for Mag7 SEC Filings")

    st.divider()
    st.subheader("Filters")

    company_options = ["All Companies"] + [
        f"{ticker} — {info['name']}" for ticker, info in MAG7_COMPANIES.items()
    ]
    selected_company = st.selectbox("Company", company_options, index=0)

    filing_type_options = ["All Filings", "10-K (Annual)", "10-Q (Quarterly)"]
    selected_filing = st.selectbox("Filing Type", filing_type_options, index=0)

    st.divider()
    st.subheader("User Profile")
    st.caption("Help the AI personalize responses")

    expertise = st.selectbox(
        "Investment Expertise",
        ["General Investor", "Retail Investor", "Institutional Analyst", "Fund Manager", "Student"],
        index=0,
    )
    focus = st.multiselect(
        "Areas of Focus",
        ["Revenue Growth", "Profitability", "AI/Technology", "Risk", "Governance", "Valuation"],
        default=[],
    )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Claude Sonnet 4.6 + Pinecone + Cohere")

# ── Helper functions ─────────────────────────────────────────────────────────

def _render_trace(trace: list[dict]) -> None:
    """Render agent trace steps in a readable format."""
    for step in trace:
        node = step.get("node", "unknown")

        if node == "orchestrator":
            st.markdown(f"**Orchestrator** routed to: `{step.get('selected_agents', [])}`")
            st.markdown(f"- Company: `{step.get('company_filter', 'All')}`")
            st.markdown(f"- Filing: `{step.get('filing_type_filter', 'All')}`")
            if step.get("routing_explanation"):
                st.caption(step["routing_explanation"])

        elif node == "reranker":
            st.markdown(
                f"**Retrieval** — queried {len(step.get('namespaces_queried', []))} namespace(s), "
                f"retrieved {step.get('retrieved_count', 0)}, reranked to {step.get('reranked_count', 0)}"
            )
            if step.get("retry_count", 0) > 0:
                st.caption(f"Retry #{step['retry_count']}")

        elif node in ("finance_agent", "tech_agent", "legal_agent", "persona_agent"):
            label = node.replace("_agent", "").capitalize()
            st.markdown(f"**{label} Agent** — used {step.get('chunks_used', 0)} context chunks")
            if step.get("output_preview"):
                st.caption(step["output_preview"])

        elif node == "validator":
            passed = step.get("validation_passed", True)
            icon = "✅" if passed else "⚠️"
            st.markdown(f"**Validator** {icon}")
            if not passed and step.get("unsupported_claims"):
                st.caption(f"Issues: {'; '.join(step['unsupported_claims'][:2])}")

        st.divider()


# ── Build state modifiers from sidebar ───────────────────────────────────────

def _resolve_company_filter() -> str | None:
    """Convert sidebar company selection to Pinecone namespace."""
    if selected_company == "All Companies":
        return None
    ticker = selected_company.split(" — ")[0]
    return MAG7_COMPANIES.get(ticker, {}).get("namespace")


def _resolve_filing_filter() -> str | None:
    if selected_filing == "10-K (Annual)":
        return "10-K"
    if selected_filing == "10-Q (Quarterly)":
        return "10-Q"
    return None


def _build_user_profile() -> dict:
    return {
        "expertise": expertise,
        "focus_areas": ", ".join(focus) if focus else "General",
    }


# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("SEC Filing Research Assistant")
st.caption("Ask questions about Apple, Microsoft, Alphabet, Amazon, Meta, NVIDIA, or Tesla SEC filings.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent_trace"):
            with st.expander("Agent Reasoning", expanded=False):
                _render_trace(msg["agent_trace"])


# ── Query input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Mag7 SEC filings..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare state
    state = initial_state(
        user_query=prompt,
        user_profile=_build_user_profile(),
    )
    # Apply sidebar filters (override orchestrator defaults when user explicitly filters)
    company_filter = _resolve_company_filter()
    filing_filter = _resolve_filing_filter()

    # Run the graph with streaming
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        trace_placeholder = st.empty()
        response_placeholder = st.empty()

        accumulated_trace: list[dict] = []
        final_response = ""

        try:
            for event in graph.stream(state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if not node_output:
                        continue

                    # Update status
                    status_placeholder.caption(f"Running: {node_name}...")

                    # Accumulate trace
                    new_trace = node_output.get("agent_trace", [])
                    if new_trace:
                        accumulated_trace = new_trace
                        with trace_placeholder.expander("Agent Reasoning", expanded=True):
                            _render_trace(accumulated_trace)

                    # Check for final response
                    if node_output.get("final_response"):
                        final_response = node_output["final_response"]
                        response_placeholder.markdown(final_response)

            # Clear status
            status_placeholder.empty()

            # If no final response was set during streaming, invoke synchronously
            if not final_response:
                result = graph.invoke(state)
                final_response = result.get("final_response", "No response generated.")
                accumulated_trace = result.get("agent_trace", [])
                response_placeholder.markdown(final_response)

        except Exception as exc:
            final_response = f"An error occurred: {str(exc)}"
            response_placeholder.error(final_response)

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response,
        "agent_trace": accumulated_trace,
    })
