"""
LangGraph AgentState definition.

All nodes in the graph read from and write to this shared state TypedDict.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    user_query: str

    # ── User context ──────────────────────────────────────────────────────
    user_profile: dict[str, Any]   # "knows me" context for personalization

    # ── Routing / filters ─────────────────────────────────────────────────
    selected_agents: list[str]     # ["finance", "tech", "legal", "persona"]
    company_filter: str | None     # Pinecone namespace (e.g. "apple")
    filing_type_filter: str | None # "10-K", "10-Q", or None for any

    # ── Retrieved context ─────────────────────────────────────────────────
    retrieved_chunks: list[dict]   # top-20 from Pinecone
    reranked_chunks: list[dict]    # top-5 after Cohere rerank

    # ── Specialist outputs ─────────────────────────────────────────────────
    finance_output: str | None
    tech_output: str | None
    legal_output: str | None
    persona_output: str | None

    # ── Validation ────────────────────────────────────────────────────────
    validation_passed: bool
    validation_issues: list[str]
    retry_count: int

    # ── Final output ──────────────────────────────────────────────────────
    final_response: str | None
    agent_trace: list[dict]         # step records for UI expander


def initial_state(user_query: str, user_profile: dict | None = None) -> AgentState:
    """Create a fresh AgentState for a new query."""
    return AgentState(
        messages=[],
        user_query=user_query,
        user_profile=user_profile or {},
        selected_agents=[],
        company_filter=None,
        filing_type_filter=None,
        retrieved_chunks=[],
        reranked_chunks=[],
        finance_output=None,
        tech_output=None,
        legal_output=None,
        persona_output=None,
        validation_passed=False,
        validation_issues=[],
        retry_count=0,
        final_response=None,
        agent_trace=[],
    )
