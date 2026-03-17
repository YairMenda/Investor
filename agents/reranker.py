"""
Retrieval + Reranking node.

1. Embed user_query with OpenAI
2. Query Pinecone: top_k=20 with namespace and optional filing_type filter
3. Cohere Rerank on top-20 → top-5
4. Store results in retrieved_chunks and reranked_chunks
"""

import cohere

from agents.state import AgentState
from config.companies import MAG7_COMPANIES, ALL_TICKERS
from config.settings import settings
from pipeline.embedder import embed_query
from pipeline.indexer import query_index

PINECONE_TOP_K = 20
COHERE_TOP_N = 5


def _get_cohere_client() -> cohere.Client:
    return cohere.Client(api_key=settings.cohere_api_key)


def _query_single_namespace(
    query_vector: list[float],
    namespace: str,
    filing_type_filter: str | None,
    top_k: int,
) -> list[dict]:
    """Query a single Pinecone namespace and return match dicts."""
    metadata_filter = None
    if filing_type_filter:
        metadata_filter = {"filing_type": {"$eq": filing_type_filter}}

    matches = query_index(
        query_vector=query_vector,
        namespace=namespace,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    return [
        {
            "id": m.id,
            "score": m.score,
            "text": m.metadata.get("text", ""),
            "metadata": m.metadata,
        }
        for m in matches
    ]


def reranker_node(state: AgentState) -> dict:
    """
    LangGraph node: retrieve and rerank context for the current query.
    """
    query = state["user_query"]
    company_filter = state.get("company_filter")
    filing_type_filter = state.get("filing_type_filter")
    retry_count = state.get("retry_count", 0)

    # On retry, expand the query to pull broader context
    if retry_count > 0:
        validation_issues = state.get("validation_issues", [])
        issue_context = "; ".join(validation_issues[:2]) if validation_issues else ""
        query_for_retrieval = f"{query} {issue_context}".strip()
    else:
        query_for_retrieval = query

    # 1. Embed the query
    query_vector = embed_query(query_for_retrieval)

    # 2. Query Pinecone
    if company_filter:
        # Single company namespace
        namespaces = [company_filter]
        per_namespace_k = PINECONE_TOP_K
    else:
        # All companies — query each namespace and merge
        namespaces = [MAG7_COMPANIES[t]["namespace"] for t in ALL_TICKERS]
        per_namespace_k = max(3, PINECONE_TOP_K // len(namespaces))

    all_matches: list[dict] = []
    for ns in namespaces:
        try:
            matches = _query_single_namespace(
                query_vector, ns, filing_type_filter, per_namespace_k
            )
            all_matches.extend(matches)
        except Exception as exc:
            print(f"  WARNING: Pinecone query failed for namespace '{ns}': {exc}")

    # Sort by score descending; take top PINECONE_TOP_K across all namespaces
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = all_matches[:PINECONE_TOP_K]

    # 3. Cohere Rerank
    reranked: list[dict] = []
    if top_matches:
        try:
            co = _get_cohere_client()
            texts = [m["text"] for m in top_matches]
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=texts,
                top_n=COHERE_TOP_N,
            )
            for result in rerank_response.results:
                match = top_matches[result.index]
                match["rerank_score"] = result.relevance_score
                reranked.append(match)
        except Exception as exc:
            print(f"  WARNING: Cohere rerank failed: {exc}. Using top Pinecone matches.")
            reranked = top_matches[:COHERE_TOP_N]
    else:
        reranked = []

    return {
        "retrieved_chunks": top_matches,
        "reranked_chunks": reranked,
        "agent_trace": [{
            "node": "reranker",
            "namespaces_queried": namespaces,
            "retrieved_count": len(top_matches),
            "reranked_count": len(reranked),
            "query_used": query_for_retrieval,
            "retry_count": retry_count,
        }],
    }
