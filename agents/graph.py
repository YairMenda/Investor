"""
LangGraph StateGraph definition for the Investor agent workflow.

Flow:
    START → orchestrator → reranker → [fan-out to selected agents] → validator → END
                                                                          ↑        |
                                                                          └─retry──┘
"""

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.orchestrator import orchestrator_node
from agents.reranker import reranker_node
from agents.finance_agent import finance_agent_node
from agents.tech_agent import tech_agent_node
from agents.legal_agent import legal_agent_node
from agents.persona_agent import persona_agent_node
from agents.validator import validator_node, should_retry


def _route_after_orchestrator(state: AgentState) -> str:
    """
    If the orchestrator rejected the query (set final_response), skip to END.
    Otherwise proceed to reranker.
    """
    if state.get("final_response"):
        return "end"
    return "reranker"


def _route_after_validator(state: AgentState) -> str:
    """Route to retry (reranker) or END based on validator decision."""
    return should_retry(state)


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────────
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("finance_agent", finance_agent_node)
    workflow.add_node("tech_agent", tech_agent_node)
    workflow.add_node("legal_agent", legal_agent_node)
    workflow.add_node("persona_agent", persona_agent_node)
    workflow.add_node("validator", validator_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    # START → orchestrator
    workflow.add_edge(START, "orchestrator")

    # orchestrator → reranker (or END if query rejected)
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"reranker": "reranker", "end": END},
    )

    # reranker → all specialist agents in parallel (fan-out)
    # Each agent node internally checks if it was selected
    workflow.add_edge("reranker", "finance_agent")
    workflow.add_edge("reranker", "tech_agent")
    workflow.add_edge("reranker", "legal_agent")
    workflow.add_edge("reranker", "persona_agent")

    # All specialist agents → validator (fan-in)
    workflow.add_edge("finance_agent", "validator")
    workflow.add_edge("tech_agent", "validator")
    workflow.add_edge("legal_agent", "validator")
    workflow.add_edge("persona_agent", "validator")

    # validator → retry (back to reranker) or END
    workflow.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"retry": "reranker", "end": END},
    )

    return workflow.compile()


# Module-level compiled graph — import this in other modules
graph = build_graph()
