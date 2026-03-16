"""
Validator node — checks specialist outputs for hallucinations against retrieved context.

If unsupported claims are found and retry_count < 3, routes back to reranker.
Otherwise assembles the final response and routes to END.
"""

import json
import re

from anthropic import Anthropic

from agents.state import AgentState
from config.settings import settings

client = Anthropic(api_key=settings.anthropic_api_key)

MAX_RETRIES = 3

VALIDATOR_SYSTEM = """You are a fact-checking assistant for investment research. Your job is to verify that all factual claims in a response are directly supported by the provided source context.

For each response you receive, check whether numerical figures, named individuals, product names, dates, and regulatory claims appear explicitly in the context.

Respond with valid JSON only:
{
  "validation_passed": true | false,
  "unsupported_claims": ["claim 1", "claim 2"],
  "explanation": "brief explanation"
}

Rules:
- ONLY flag claims that contradict or are absent from the context
- Do NOT flag claims that are reasonable inferences from the context
- If no context was retrieved (context is empty), validation_passed should be false
- Minor paraphrasing is acceptable
"""


def _collect_outputs(state: AgentState) -> str:
    """Aggregate all specialist agent outputs into one string for validation."""
    outputs = []
    if state.get("finance_output"):
        outputs.append(f"## Finance Analysis\n{state['finance_output']}")
    if state.get("tech_output"):
        outputs.append(f"## Technology Analysis\n{state['tech_output']}")
    if state.get("legal_output"):
        outputs.append(f"## Legal/Risk Analysis\n{state['legal_output']}")
    if state.get("persona_output"):
        outputs.append(f"## Executive/Governance Analysis\n{state['persona_output']}")
    return "\n\n".join(outputs)


def _format_context_summary(chunks: list[dict]) -> str:
    """Summarize retrieved context for the validator."""
    if not chunks:
        return "No context retrieved."
    parts = []
    for chunk in chunks[:5]:
        meta = chunk.get("metadata", {})
        parts.append(
            f"[{meta.get('company_name', '?')} | {meta.get('period', '?')} | "
            f"{meta.get('section_title', '?')}]\n{chunk['text'][:500]}"
        )
    return "\n\n---\n\n".join(parts)


def _assemble_final_response(state: AgentState) -> str:
    """Combine specialist outputs into a coherent final response."""
    outputs = []

    if state.get("finance_output"):
        outputs.append(state["finance_output"])
    if state.get("tech_output"):
        outputs.append(state["tech_output"])
    if state.get("legal_output"):
        outputs.append(state["legal_output"])
    if state.get("persona_output"):
        outputs.append(state["persona_output"])

    if not outputs:
        return "I was unable to find relevant information in the available SEC filings for your query."

    if len(outputs) == 1:
        return outputs[0]

    # Multiple outputs — combine with headers
    sections = []
    if state.get("finance_output"):
        sections.append(f"### Financial Analysis\n{state['finance_output']}")
    if state.get("tech_output"):
        sections.append(f"### Technology Analysis\n{state['tech_output']}")
    if state.get("legal_output"):
        sections.append(f"### Legal & Risk Analysis\n{state['legal_output']}")
    if state.get("persona_output"):
        sections.append(f"### Executive & Governance\n{state['persona_output']}")

    return "\n\n".join(sections)


def validator_node(state: AgentState) -> dict:
    """
    LangGraph node: validate outputs and assemble or trigger retry.
    """
    trace = list(state.get("agent_trace", []))
    retry_count = state.get("retry_count", 0)
    reranked = state.get("reranked_chunks", [])

    combined_output = _collect_outputs(state)
    context_summary = _format_context_summary(reranked)

    if not combined_output.strip():
        # Nothing to validate — probably invalid query was rejected upstream
        return {
            "validation_passed": True,
            "validation_issues": [],
            "final_response": state.get("final_response", "No analysis available."),
            "agent_trace": trace,
        }

    # Ask Claude to validate
    validation_prompt = (
        f"Source Context:\n{context_summary}\n\n"
        f"Response to Validate:\n{combined_output}"
    )

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=VALIDATOR_SYSTEM,
        messages=[{"role": "user", "content": validation_prompt}],
    )

    raw = response.content[0].text.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    result = json.loads(json_match.group()) if json_match else {"validation_passed": True, "unsupported_claims": []}

    passed = result.get("validation_passed", True)
    issues = result.get("unsupported_claims", [])

    trace.append({
        "node": "validator",
        "validation_passed": passed,
        "unsupported_claims": issues,
        "retry_count": retry_count,
        "explanation": result.get("explanation", ""),
    })

    if not passed and retry_count < MAX_RETRIES:
        return {
            "validation_passed": False,
            "validation_issues": issues,
            "retry_count": retry_count + 1,
            "agent_trace": trace,
        }

    # Passed or max retries reached — assemble final response
    final = _assemble_final_response(state)
    return {
        "validation_passed": True,
        "validation_issues": issues,
        "final_response": final,
        "agent_trace": trace,
    }


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function for the LangGraph retry loop.
    Returns "retry" or "end".
    """
    if (
        not state.get("validation_passed", True)
        and state.get("retry_count", 0) < MAX_RETRIES
        and not state.get("final_response")
    ):
        return "retry"
    return "end"
