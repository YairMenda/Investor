"""
Batch-embed chunk texts using OpenAI text-embedding-3-large.

Batches at 100 texts per API call to stay well within rate limits.
Returns list of 3072-dimensional vectors aligned with input chunks.
"""

import time
from typing import Generator

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from pipeline.chunker import Chunk

BATCH_SIZE = 100


def _batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive batches of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# Max characters to embed — text-embedding-3-large limit is 8192 tokens.
# Roughly 4 chars/token; 7500 tokens * 4 = 30000 chars (safe ceiling).
MAX_EMBED_CHARS = 30_000


def _truncate_for_embedding(text: str) -> str:
    """Truncate text to fit within the embedding model's token limit."""
    if len(text) <= MAX_EMBED_CHARS:
        return text
    # Truncate at a word boundary
    truncated = text[:MAX_EMBED_CHARS]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry on failure."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    # Response is ordered by index
    vectors = sorted(response.data, key=lambda x: x.index)
    return [v.embedding for v in vectors]


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Embed all chunk texts in batches of 100.

    Args:
        chunks: List of Chunk objects.

    Returns:
        List of 3072-dim embedding vectors, aligned with input chunks.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    texts = [_truncate_for_embedding(c.text) for c in chunks]
    all_vectors: list[list[float]] = []

    for batch_num, batch in enumerate(_batched(texts, BATCH_SIZE)):
        vectors = _embed_batch(client, batch)
        all_vectors.extend(vectors)
        # Minimal progress output
        processed = min((batch_num + 1) * BATCH_SIZE, len(texts))
        print(f"  Embedded {processed}/{len(texts)} chunks")

    return all_vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding
