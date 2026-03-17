"""
Orchestrator node — routes user queries to the appropriate specialist agents.

Extracts:
  - Company filter (Pinecone namespace)
  - Filing type filter
  - Selected specialist agents
  - Query validity (guardrails)
"""

import json
import re

from anthropic import Anthropic

from agents.state import AgentState
from agents.prompts.orchestrator import ORCHESTRATOR_SYSTEM, ORCHESTRATOR_FEW_SHOTS
from config.settings import settings

client = Anthropic(api_key=settings.anthropic_api_key)


def _build_few_shot_text() -> str:
    """Format few-shot examples for the orchestrator prompt."""
    lines = ["\n## Examples\n"]
    for ex in ORCHESTRATOR_FEW_SHOTS:
        lines.append(f"Query: {ex['query']}")
        lines.append(f"Response: {json.dumps(ex['response'], indent=2)}\n")
    return "\n".join(lines)


def _personalize_context(user_profile: dict) -> str:
    """Add user profile context to the orchestrator prompt if available."""
    if not user_profile:
        return ""
    lines = ["\n## User Profile (personalization context)"]
    for key, value in user_profile.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def orchestrator_node(state: AgentState) -> dict:
    """
    LangGraph node: analyze the user query and set routing state.
    """
    query = state["user_query"]
    user_profile = state.get("user_profile", {})

    system_prompt = ORCHESTRATOR_SYSTEM + _build_few_shot_text() + _personalize_context(user_profile)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Route this query: {query}"}],
    )

    raw_text = response.content[0].text.strip()

    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    routing = json.loads(json_match.group()) if json_match else json.loads(raw_text)

    trace_entry = {
        "node": "orchestrator",
        "query": query,
        "company_filter": routing.get("company_filter"),
        "filing_type_filter": routing.get("filing_type_filter"),
        "selected_agents": routing.get("selected_agents", []),
        "routing_explanation": routing.get("routing_explanation", ""),
        "is_valid_query": routing.get("is_valid_query", True),
    }

    if not routing.get("is_valid_query", True):
        return {
            "selected_agents": [],
            "company_filter": None,
            "filing_type_filter": None,
            "final_response": routing.get("rejection_reason", "I can only answer questions about Mag7 SEC filings."),
            "agent_trace": [trace_entry],
        }

    return {
        "selected_agents": routing.get("selected_agents", ["finance"]),
        "company_filter": routing.get("company_filter"),
        "filing_type_filter": routing.get("filing_type_filter"),
        "agent_trace": [trace_entry],
    }
