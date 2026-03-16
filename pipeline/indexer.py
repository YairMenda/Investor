"""
Index chunk embeddings into Pinecone.

Creates the serverless index on first run (dim=3072, cosine, aws us-east-1).
Upserts in batches of 100; grouped by namespace (one per company).
Chunk text is stored in vector metadata for retrieval without secondary lookups.
"""

from time import sleep
from typing import Iterator

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from config.settings import settings
from pipeline.chunker import Chunk

UPSERT_BATCH_SIZE = 100


def _get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _ensure_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it does not exist yet."""
    existing = [idx.name for idx in pc.list_indexes().indexes]
    if settings.pinecone_index_name in existing:
        return

    print(f"Creating Pinecone index '{settings.pinecone_index_name}'...")
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
    )

    # Wait for index to be ready
    for _ in range(30):
        status = pc.describe_index(settings.pinecone_index_name).status
        if status.get("ready"):
            break
        sleep(2)
    print("  Index ready.")


def _chunk_to_vector(chunk: Chunk, vector: list[float]) -> dict:
    """Convert a Chunk + embedding into a Pinecone vector dict."""
    return {
        "id": chunk.chunk_id,
        "values": vector,
        "metadata": {
            "text": chunk.text,
            "ticker": chunk.ticker,
            "company_name": chunk.company_name,
            "filing_type": chunk.filing_type,
            "period": chunk.period,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "category": chunk.category,
            "source_file": chunk.source_file,
            "accession_number": chunk.accession_number,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
        },
    }


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    namespace: str | None = None,
) -> int:
    """
    Upsert chunks and their embeddings into Pinecone.

    Args:
        chunks: List of Chunk objects.
        vectors: Aligned list of embedding vectors.
        namespace: Override namespace (defaults to chunk.namespace).

    Returns:
        Total number of vectors upserted.
    """
    assert len(chunks) == len(vectors), "chunks and vectors must be aligned"

    pc = _get_pinecone_client()
    _ensure_index(pc)
    index = pc.Index(settings.pinecone_index_name)

    # Group by namespace
    grouped: dict[str, list[tuple[Chunk, list[float]]]] = {}
    for chunk, vec in zip(chunks, vectors):
        ns = namespace or chunk.namespace
        grouped.setdefault(ns, []).append((chunk, vec))

    total_upserted = 0
    for ns, pairs in grouped.items():
        pinecone_vectors = [_chunk_to_vector(c, v) for c, v in pairs]

        for batch in tqdm(
            list(_batched(pinecone_vectors, UPSERT_BATCH_SIZE)),
            desc=f"Upserting {ns}",
        ):
            index.upsert(vectors=batch, namespace=ns)
            total_upserted += len(batch)

        print(f"  Namespace '{ns}': {len(pairs)} vectors upserted")

    return total_upserted


def index_all(
    chunks_by_ticker: dict[str, list[Chunk]],
    vectors_by_ticker: dict[str, list[list[float]]],
) -> int:
    """
    Index all tickers. Convenience wrapper over index_chunks.

    Returns total vectors upserted across all tickers.
    """
    total = 0
    for ticker, chunks in chunks_by_ticker.items():
        if not chunks:
            continue
        vectors = vectors_by_ticker.get(ticker, [])
        if not vectors:
            print(f"  WARNING: No vectors for {ticker}, skipping index.")
            continue
        total += index_chunks(chunks, vectors)
    return total


def query_index(
    query_vector: list[float],
    namespace: str,
    top_k: int = 20,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Query Pinecone for the most similar vectors.

    Returns list of matches with metadata (including 'text').
    """
    pc = _get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=metadata_filter,
    )
    return response.matches
