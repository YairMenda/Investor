"""
Persona specialist agent — C-suite, board, compensation, and governance analysis.
"""

from anthropic import Anthropic

from agents.state import AgentState
from agents.prompts.persona import PERSONA_SYSTEM, PERSONA_FEW_SHOTS
from config.settings import settings

client = Anthropic(api_key=settings.anthropic_api_key)

CATEGORY = "personas"


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context retrieved."
    parts = []
    for chunk in chunks:
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
    messages = []
    for ex in PERSONA_FEW_SHOTS:
        messages.append({
            "role": "user",
            "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
        })
        messages.append({"role": "assistant", "content": ex["answer"]})
    return messages


def persona_agent_node(state: AgentState) -> dict:
    """LangGraph node for executive/governance analysis."""
    selected = state.get("selected_agents", [])
    if CATEGORY not in selected:
        return {}

    reranked = state.get("reranked_chunks", [])
    query = state["user_query"]

    persona_chunks = [c for c in reranked if c.get("metadata", {}).get("category") == CATEGORY]
    context_chunks = persona_chunks if persona_chunks else reranked

    context_str = _format_context(context_chunks)
    messages = _build_few_shot_messages() + [
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        }
    ]

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=PERSONA_SYSTEM,
        messages=messages,
    )

    output = response.content[0].text.strip()

    return {
        "persona_output": output,
        "agent_trace": [{
            "node": "persona_agent",
            "chunks_used": len(context_chunks),
            "output_preview": output[:200] + "..." if len(output) > 200 else output,
        }],
    }
