"""
CLI orchestrator for the full data pipeline.

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --tickers AAPL MSFT --filing-types 10-K
    python -m pipeline.run_pipeline --tickers AAPL --filing-types 10-K --limit 2 --skip-download

Checkpointing:
    Each step is skipped if its output already exists (idempotent).
    Use --force to re-run a specific step even if output exists.
"""

import argparse
import json
import sys
from pathlib import Path

from config.companies import MAG7_COMPANIES, SUPPORTED_FILING_TYPES

# Checkpoint tracking
CHECKPOINT_FILE = Path("data/.pipeline_checkpoint.json")


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def _step_done(checkpoint: dict, step: str, ticker: str) -> bool:
    return checkpoint.get(ticker, {}).get(step, False)


def _mark_done(checkpoint: dict, step: str, ticker: str) -> None:
    checkpoint.setdefault(ticker, {})[step] = True


def run_pipeline(
    tickers: list[str],
    filing_types: list[str],
    limit: int = 5,
    after: str = "2023-12-31",
    skip_download: bool = False,
    skip_extract: bool = False,
    skip_preprocess: bool = False,
    skip_chunk: bool = False,
    skip_embed: bool = False,
    skip_index: bool = False,
    force: bool = False,
) -> None:
    checkpoint = _load_checkpoint()

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not skip_download:
        print("\n=== Step 1: Downloading SEC filings ===")
        from pipeline.downloader import download_filings
        download_filings(
            tickers=tickers,
            filing_types=filing_types,
            after=after,
            limit=limit,
        )
        for ticker in tickers:
            _mark_done(checkpoint, "download", ticker)
        _save_checkpoint(checkpoint)
    else:
        print("\n[Skipping download]")

    # ── Step 2: Extract PDFs → Markdown ──────────────────────────────────
    if not skip_extract:
        print("\n=== Step 2: Extracting PDFs → Markdown ===")
        from pipeline.extractor import extract_all
        extract_all(tickers=tickers, filing_types=filing_types, force=force)
        for ticker in tickers:
            _mark_done(checkpoint, "extract", ticker)
        _save_checkpoint(checkpoint)
    else:
        print("\n[Skipping extraction]")

    # ── Step 3: Preprocess → SectionDocuments ────────────────────────────
    print("\n=== Step 3: Preprocessing (section parsing) ===")
    from pipeline.preprocessor import parse_all
    sections_by_ticker = parse_all(tickers=tickers, filing_types=filing_types)

    total_sections = sum(len(v) for v in sections_by_ticker.values())
    print(f"  Total sections parsed: {total_sections}")

    # ── Step 4: Chunk sections ────────────────────────────────────────────
    if not skip_chunk:
        print("\n=== Step 4: Semantic chunking ===")
        from pipeline.chunker import chunk_all, load_all_chunks

        # Skip tickers where chunks already exist (unless force)
        needs_chunking = {}
        for ticker in tickers:
            from pipeline.chunker import load_chunks
            existing = load_chunks(ticker)
            if existing and not force:
                print(f"  {ticker}: {len(existing)} chunks already exist, skipping.")
                sections_by_ticker[ticker] = []  # don't rechunk
            else:
                needs_chunking[ticker] = sections_by_ticker.get(ticker, [])

        if needs_chunking:
            new_chunks = chunk_all(needs_chunking, save=True)
        else:
            new_chunks = {}

        # Reload all chunks (including previously chunked)
        all_chunks = {
            ticker: load_chunks(ticker) for ticker in tickers
        }
    else:
        print("\n[Skipping chunking — loading from disk]")
        from pipeline.chunker import load_all_chunks
        all_chunks = load_all_chunks(tickers)

    total_chunks = sum(len(v) for v in all_chunks.values())
    print(f"  Total chunks: {total_chunks}")

    # ── Step 5: Embed chunks ──────────────────────────────────────────────
    vectors_by_ticker: dict[str, list[list[float]]] = {}

    if not skip_embed:
        print("\n=== Step 5: Embedding chunks ===")
        from pipeline.embedder import embed_chunks

        for ticker in tickers:
            chunks = all_chunks.get(ticker, [])
            if not chunks:
                print(f"  {ticker}: no chunks to embed.")
                vectors_by_ticker[ticker] = []
                continue

            # Check if embedding file exists
            embed_cache = Path(f"data/chunks/{ticker}_embeddings.json")
            if embed_cache.exists() and not force:
                print(f"  {ticker}: embeddings cached, loading from disk.")
                import json as _json
                with open(embed_cache) as f:
                    vectors_by_ticker[ticker] = _json.load(f)
                continue

            print(f"  Embedding {len(chunks)} chunks for {ticker}...")
            vectors = embed_chunks(chunks)
            vectors_by_ticker[ticker] = vectors

            # Cache embeddings to avoid re-embedding on partial failure
            with open(embed_cache, "w") as f:
                import json as _json
                _json.dump(vectors, f)
            print(f"  Embeddings cached → {embed_cache}")
    else:
        print("\n[Skipping embedding — loading from cache]")
        for ticker in tickers:
            embed_cache = Path(f"data/chunks/{ticker}_embeddings.json")
            if embed_cache.exists():
                import json as _json
                with open(embed_cache) as f:
                    vectors_by_ticker[ticker] = _json.load(f)
            else:
                print(f"  WARNING: No embedding cache for {ticker}")
                vectors_by_ticker[ticker] = []

    # ── Step 6: Index into Pinecone ───────────────────────────────────────
    if not skip_index:
        print("\n=== Step 6: Indexing into Pinecone ===")
        from pipeline.indexer import index_chunks

        for ticker in tickers:
            chunks = all_chunks.get(ticker, [])
            vectors = vectors_by_ticker.get(ticker, [])

            if not chunks or not vectors:
                print(f"  {ticker}: nothing to index.")
                continue

            count = index_chunks(chunks, vectors)
            print(f"  {ticker}: {count} vectors indexed")
            _mark_done(checkpoint, "index", ticker)

        _save_checkpoint(checkpoint)
    else:
        print("\n[Skipping indexing]")

    print("\n=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Investor data pipeline for SEC filing ingestion."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(MAG7_COMPANIES.keys()),
        help="Ticker symbols to process (default: all Mag7)",
    )
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=SUPPORTED_FILING_TYPES,
        choices=SUPPORTED_FILING_TYPES,
        help="Filing types to process (default: 10-K 10-Q)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max filings to download per ticker per filing type (default: 5)",
    )
    parser.add_argument(
        "--after",
        default="2023-12-31",
        help="Only download filings after this date (default: 2023-12-31)",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if output already exists",
    )

    args = parser.parse_args()

    # Validate tickers
    for ticker in args.tickers:
        if ticker not in MAG7_COMPANIES:
            print(f"ERROR: Unknown ticker '{ticker}'. Must be one of: {list(MAG7_COMPANIES.keys())}")
            sys.exit(1)

    run_pipeline(
        tickers=args.tickers,
        filing_types=args.filing_types,
        limit=args.limit,
        after=args.after,
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
        skip_preprocess=args.skip_preprocess,
        skip_chunk=args.skip_chunk,
        skip_embed=args.skip_embed,
        skip_index=args.skip_index,
        force=args.force,
    )


if __name__ == "__main__":
    main()
