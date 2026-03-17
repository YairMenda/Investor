"""
CLI orchestrator for the full data pipeline.

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --tickers AAPL MSFT
    python -m pipeline.run_pipeline --tickers AAPL --filing-types 10-K
"""

import argparse
import sys

from config.companies import MAG7_COMPANIES, ALL_TICKERS


def run_ticker(ticker: str, filing_types: list[str]) -> None:
    """Run the full pipeline for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  Processing {ticker}")
    print(f"{'='*60}")

    # Step 1: Download
    print(f"\n[1/5] Downloading {ticker} filings...")
    from pipeline.downloader import download_filings
    downloaded = download_filings(tickers=[ticker], filing_types=filing_types)
    paths = downloaded.get(ticker, [])
    print(f"  Found {len(paths)} raw file(s) for {ticker}")
    if not paths:
        print(f"  WARNING: No filings found for {ticker}, skipping.")
        return

    # Step 2: Extract (HTML/PDF -> Markdown)
    print(f"\n[2/5] Extracting {ticker} filings to Markdown...")
    from pipeline.extractor import extract_all
    extracted = extract_all(tickers=[ticker], filing_types=filing_types)
    md_paths = extracted.get(ticker, [])
    print(f"  Extracted {len(md_paths)} markdown file(s)")
    if not md_paths:
        print(f"  WARNING: No markdown extracted for {ticker}, skipping.")
        return

    # Step 3: Preprocess (section splitting + metadata tagging)
    print(f"\n[3/5] Preprocessing {ticker} sections...")
    from pipeline.preprocessor import parse_all
    parsed = parse_all(tickers=[ticker], filing_types=filing_types)
    docs = parsed.get(ticker, [])
    print(f"  Parsed {len(docs)} section(s)")

    # Step 4: Chunk
    print(f"\n[4/5] Chunking {ticker} sections...")
    from pipeline.chunker import chunk_all
    chunks_by_ticker = chunk_all({ticker: docs}, save=True)
    ticker_chunks = chunks_by_ticker.get(ticker, [])
    print(f"  Created {len(ticker_chunks)} chunk(s)")
    if not ticker_chunks:
        print(f"  WARNING: No chunks created for {ticker}, skipping indexing.")
        return

    # Step 5: Embed + Index
    print(f"\n[5/5] Embedding and indexing {ticker}...")
    from pipeline.embedder import embed_chunks
    from pipeline.indexer import index_chunks
    vectors = embed_chunks(ticker_chunks)
    index_chunks(ticker_chunks, vectors)

    namespace = MAG7_COMPANIES[ticker]["namespace"]
    print(f"  Indexed {len(ticker_chunks)} vectors into namespace '{namespace}'")
    print(f"\n  {ticker} DONE.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Investor data pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=ALL_TICKERS,
        choices=ALL_TICKERS,
        help="Tickers to process (default: all Mag7)",
    )
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=["10-K", "10-Q"],
        choices=["10-K", "10-Q"],
        help="Filing types to process (default: 10-K and 10-Q)",
    )
    args = parser.parse_args()

    tickers = args.tickers
    filing_types = args.filing_types

    print(f"Pipeline starting for: {', '.join(tickers)}")
    print(f"Filing types: {', '.join(filing_types)}")

    for ticker in tickers:
        try:
            run_ticker(ticker, filing_types)
        except Exception as exc:
            print(f"\nERROR processing {ticker}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print(f"Continuing with next ticker...\n")

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
