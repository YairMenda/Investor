"""
Finance specialist agent — analyzes financial data from SEC filings.
"""

from anthropic import Anthropic

from agents.state import AgentState
from agents.prompts.finance import FINANCE_SYSTEM, FINANCE_FEW_SHOTS
from config.settings import settings

client = Anthropic(api_key=settings.anthropic_api_key)

CATEGORY = "finance"


def _format_context(chunks: list[dict]) -> str:
    """Format reranked chunks as labeled context blocks."""
    if not chunks:
        return "No relevant context retrieved."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        label = (
            f"[{meta.get('company_name', '?')} | "
            f"{meta.get('filing_type', '?')} | "
            f"{meta.get('period', '?')} | "
            f"{meta.get('section_title', '?')}]"
        )
        parts.append(f"{label}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def _build_few_shot_messages() -> list[dict]:
    """Build few-shot conversation turns for Claude."""
    messages = []
    for ex in FINANCE_FEW_SHOTS:
        messages.append({
            "role": "user",
            "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
        })
        messages.append({"role": "assistant", "content": ex["answer"]})
    return messages


def finance_agent_node(state: AgentState) -> dict:
    """LangGraph node for financial analysis."""
    selected = state.get("selected_agents", [])
    if CATEGORY not in selected:
        return {}  # Not selected for this query

    reranked = state.get("reranked_chunks", [])
    query = state["user_query"]
    trace = list(state.get("agent_trace", []))

    # Filter chunks to finance category (prefer finance, but use all if none match)
    finance_chunks = [c for c in reranked if c.get("metadata", {}).get("category") == CATEGORY]
    context_chunks = finance_chunks if finance_chunks else reranked

    context_str = _format_context(context_chunks)
    few_shot_messages = _build_few_shot_messages()

    messages = few_shot_messages + [
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        }
    ]

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=FINANCE_SYSTEM,
        messages=messages,
    )

    output = response.content[0].text.strip()

    trace.append({
        "node": "finance_agent",
        "chunks_used": len(context_chunks),
        "output_preview": output[:200] + "..." if len(output) > 200 else output,
    })

    return {
        "finance_output": output,
        "agent_trace": trace,
    }
