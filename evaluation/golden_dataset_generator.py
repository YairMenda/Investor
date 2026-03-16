"""
Golden dataset generator for RAGAS evaluation.

Samples chunks from each Pinecone namespace and uses Claude to generate
grounded Q&A pairs. Target: ~5 Q&A pairs per section category per company.

Output format (JSONL):
    {"question": "...", "ground_truth": "...", "contexts": ["..."], "metadata": {...}}
"""

import json
import random
from pathlib import Path

from anthropic import Anthropic
from pinecone import Pinecone

from config.companies import MAG7_COMPANIES, ALL_TICKERS
from config.settings import settings

client = Anthropic(api_key=settings.anthropic_api_key)
OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "golden_dataset.jsonl"

PAIRS_PER_CATEGORY = 5
CATEGORIES = ["finance", "legal", "tech", "personas"]

GENERATOR_SYSTEM = """You are an investment research expert generating evaluation questions for an AI system.

Given a passage from a company's SEC filing, generate a question-answer pair where:
1. The question is natural and requires understanding the passage
2. The answer is entirely grounded in the passage (no external knowledge needed)
3. The question is specific enough to have a definitive answer

Respond with valid JSON only:
{
  "question": "...",
  "ground_truth": "..."
}

Guidelines:
- Questions should be about specific financial figures, risks, products, or governance facts
- Avoid overly simple questions (e.g., "What company is this about?")
- Answers should be 1-3 sentences, factual, and fully supported by the passage"""


def _sample_chunks_from_namespace(
    pc: Pinecone,
    namespace: str,
    category: str,
    n: int = 10,
) -> list[dict]:
    """
    Sample random chunks from a Pinecone namespace filtered by category.
    Uses a random vector query to get diverse samples.
    """
    import random

    index = pc.Index(settings.pinecone_index_name)

    # Use random vector to sample diverse chunks
    random_vector = [random.gauss(0, 1) for _ in range(settings.embedding_dimensions)]

    try:
        response = index.query(
            vector=random_vector,
            top_k=n,
            namespace=namespace,
            include_metadata=True,
            filter={"category": {"$eq": category}},
        )
        return [
            {
                "text": m.metadata.get("text", ""),
                "metadata": dict(m.metadata),
            }
            for m in response.matches
            if m.metadata.get("text")
        ]
    except Exception as exc:
        print(f"  WARNING: Failed to sample {namespace}/{category}: {exc}")
        return []


def _generate_qa_pair(chunk: dict) -> dict | None:
    """Use Claude to generate a Q&A pair from a chunk."""
    text = chunk.get("text", "").strip()
    if len(text.split()) < 50:
        return None  # Too short to generate meaningful question

    prompt = f"SEC Filing Passage:\n{text[:1500]}\n\nGenerate a Q&A pair."

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=256,
            system=GENERATOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        import re
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return None
        qa = json.loads(json_match.group())
        return {
            "question": qa.get("question", ""),
            "ground_truth": qa.get("ground_truth", ""),
            "contexts": [text],
            "metadata": chunk.get("metadata", {}),
        }
    except Exception as exc:
        print(f"  WARNING: Q&A generation failed: {exc}")
        return None


def generate_golden_dataset(
    tickers: list[str] | None = None,
    pairs_per_category: int = PAIRS_PER_CATEGORY,
    output_file: Path = OUTPUT_FILE,
) -> list[dict]:
    """
    Generate golden Q&A dataset from Pinecone chunks.

    Args:
        tickers: Tickers to sample from (default: all Mag7).
        pairs_per_category: Target Q&A pairs per category per company.
        output_file: Where to save the JSONL dataset.

    Returns:
        List of Q&A pair dicts.
    """
    tickers = tickers or ALL_TICKERS
    pc = Pinecone(api_key=settings.pinecone_api_key)

    dataset: list[dict] = []
    OUTPUT_DIR.mkdir(exist_ok=True)

    for ticker in tickers:
        namespace = MAG7_COMPANIES[ticker]["namespace"]
        print(f"\nGenerating Q&A for {ticker} ({namespace})...")

        for category in CATEGORIES:
            # Sample more chunks than needed (some may fail generation)
            chunks = _sample_chunks_from_namespace(
                pc, namespace, category, n=pairs_per_category * 2
            )
            random.shuffle(chunks)

            generated = 0
            for chunk in chunks:
                if generated >= pairs_per_category:
                    break
                qa = _generate_qa_pair(chunk)
                if qa and qa["question"] and qa["ground_truth"]:
                    dataset.append(qa)
                    generated += 1

            print(f"  {category}: {generated} Q&A pairs generated")

    # Save dataset
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nGolden dataset: {len(dataset)} Q&A pairs → {output_file}")
    return dataset


def load_golden_dataset(path: Path = OUTPUT_FILE) -> list[dict]:
    """Load the golden dataset from JSONL file."""
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--pairs-per-category", type=int, default=5)
    args = parser.parse_args()

    generate_golden_dataset(
        tickers=args.tickers,
        pairs_per_category=args.pairs_per_category,
    )
