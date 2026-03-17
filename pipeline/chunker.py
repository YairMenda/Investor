"""
Semantic chunking with 10% overlap post-processing.

Uses LangChain SemanticChunker (percentile breakpoints) to respect
semantic boundaries in financial text. A post-processing step adds
trailing context from chunk[i] to the beginning of chunk[i+1].
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config.settings import settings
from pipeline.preprocessor import SectionDocument, FilingMetadata

CHUNKS_DIR = Path("data/chunks")


@dataclass
class Chunk:
    chunk_id: str             # SHA256[:32] of content — enables idempotent upserts
    text: str
    chunk_index: int          # position within the parent section
    total_chunks: int         # total chunks in parent section
    # --- filing metadata (duplicated for Pinecone storage) ---
    ticker: str
    company_name: str
    namespace: str
    filing_type: str
    period: str
    section_id: str
    section_title: str
    category: str
    source_file: str
    accession_number: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _make_chunk_id(text: str) -> str:
    """Deterministic SHA-256-based ID for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _get_splitter():
    """Lazily create the SemanticChunker (avoids import cost at module level)."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        openai_api_key=settings.openai_api_key,
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_breakpoint_threshold,
    )


def _add_overlap(chunks: list[str], overlap_pct: float = 0.10) -> list[str]:
    """
    Post-process chunk list to add trailing context from chunk[i] to chunk[i+1].

    Takes the last `overlap_pct` fraction of words from chunk[i] and prepends
    them to chunk[i+1], preserving semantic continuity across boundaries.
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        overlap_word_count = max(1, int(len(prev_words) * overlap_pct))
        overlap_text = " ".join(prev_words[-overlap_word_count:])
        result.append(overlap_text + " " + chunks[i])

    return result


def _split_large_chunk(text: str, max_words: int = 1200, overlap_words: int = 50) -> list[str]:
    """
    Split a large text into smaller pieces at paragraph boundaries.
    Falls back to word-count splitting if no paragraphs are found.
    """
    # Try to split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap
            current_words = current_words[-overlap_words:] + para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks if chunks else [text]


def chunk_section(
    doc: SectionDocument,
    splitter=None,
) -> list[Chunk]:
    """
    Chunk a single SectionDocument into Chunk objects.

    Args:
        doc: The SectionDocument to chunk.
        splitter: Optional pre-loaded SemanticChunker (avoids reloading).

    Returns:
        List of Chunk objects.
    """
    if splitter is None:
        splitter = _get_splitter()

    # Bail out on empty sections
    if not doc.text or not doc.text.strip():
        return []

    # SemanticChunker splits on semantic boundaries
    raw_chunks_initial = splitter.split_text(doc.text)

    # Secondary split: break any chunk > 1500 words using a simple paragraph splitter
    # (avoids importing langchain_text_splitters which has numpy/pandas compat issues)
    raw_chunks = []
    for rc in raw_chunks_initial:
        if len(rc.split()) > 1500:
            raw_chunks.extend(_split_large_chunk(rc, max_words=1200, overlap_words=50))
        else:
            raw_chunks.append(rc)

    # Post-process: add overlap between consecutive chunks
    overlapped = _add_overlap(raw_chunks, settings.chunk_overlap_pct)

    chunks: list[Chunk] = []
    total = len(overlapped)
    meta = doc.filing_meta

    for idx, text in enumerate(overlapped):
        text = text.strip()
        if not text:
            continue

        chunk = Chunk(
            chunk_id=_make_chunk_id(text),
            text=text,
            chunk_index=idx,
            total_chunks=total,
            ticker=meta.ticker,
            company_name=meta.company_name,
            namespace=meta.namespace,
            filing_type=meta.filing_type,
            period=meta.period,
            section_id=doc.section_id,
            section_title=doc.section_title,
            category=doc.category,
            source_file=meta.source_file,
            accession_number=meta.accession_number,
        )
        chunks.append(chunk)

    return chunks


def chunk_all(
    sections: dict[str, list[SectionDocument]],
    save: bool = True,
) -> dict[str, list[Chunk]]:
    """
    Chunk all sections for all tickers.

    Args:
        sections: Dict from preprocessor.parse_all().
        save: If True, serialize chunks to data/chunks/{ticker}.jsonl.

    Returns:
        Dict mapping ticker → list of Chunk objects.
    """
    splitter = _get_splitter()
    results: dict[str, list[Chunk]] = {}

    for ticker, docs in sections.items():
        if not docs:
            results[ticker] = []
            continue

        ticker_chunks: list[Chunk] = []
        for doc in tqdm(docs, desc=f"Chunking {ticker}"):
            chunks = chunk_section(doc, splitter=splitter)
            ticker_chunks.extend(chunks)

        results[ticker] = ticker_chunks
        print(f"  {ticker}: {len(ticker_chunks)} chunks from {len(docs)} sections")

        if save:
            _save_chunks(ticker, ticker_chunks)

    return results


def _save_chunks(ticker: str, chunks: list[Chunk]) -> Path:
    """Serialize chunks to JSONL for resumable pipeline runs."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{ticker}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")
    print(f"  Saved {len(chunks)} chunks to {out_path}")
    return out_path


def load_chunks(ticker: str) -> list[Chunk]:
    """Load serialized chunks for a ticker."""
    path = CHUNKS_DIR / f"{ticker}.jsonl"
    if not path.exists():
        return []
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def load_all_chunks(tickers: list[str] | None = None) -> dict[str, list[Chunk]]:
    """Load all serialized chunks for all tickers."""
    from config.companies import MAG7_COMPANIES
    tickers = tickers or list(MAG7_COMPANIES.keys())
    return {ticker: load_chunks(ticker) for ticker in tickers}
