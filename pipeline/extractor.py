"""
Extract text from SEC filing PDFs using marker-pdf.

Models are loaded once at module level (expensive on first import).
Extraction is idempotent: skips files where the .md output already exists.
"""

from pathlib import Path
from typing import Optional

from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Lazy-loaded marker models — initialized on first call to avoid import cost
_marker_models = None


def _get_marker_models():
    """Load marker-pdf models once and cache them."""
    global _marker_models
    if _marker_models is None:
        print("Loading marker-pdf models (first call — may be slow)...")
        from marker.models import create_model_dict
        _marker_models = create_model_dict()
        print("marker-pdf models loaded.")
    return _marker_models


def _pdf_to_output_path(pdf_path: Path) -> Path:
    """
    Compute the output markdown path for a given PDF.

    Maps:  data/raw/{TICKER}/{FILING_TYPE}/{accession}/...pdf
       →   data/processed/{TICKER}/{FILING_TYPE}/{accession}/{stem}.md
    """
    # Find the part of the path relative to RAW_DIR
    try:
        rel = pdf_path.relative_to(RAW_DIR)
    except ValueError:
        rel = Path(*pdf_path.parts[-4:])  # fallback: take last 4 parts

    out_dir = PROCESSED_DIR / rel.parent
    return out_dir / (pdf_path.stem + ".md")


def extract_pdf(pdf_path: Path, force: bool = False) -> Optional[Path]:
    """
    Convert a single PDF to Markdown using marker-pdf.

    Args:
        pdf_path: Path to the source PDF.
        force: Re-extract even if the .md file already exists.

    Returns:
        Path to the output .md file, or None on failure.
    """
    out_path = _pdf_to_output_path(pdf_path)

    if out_path.exists() and not force:
        return out_path  # idempotent: already extracted

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from marker.convert import convert_single_pdf

        models = _get_marker_models()
        full_text, images, metadata = convert_single_pdf(
            str(pdf_path),
            models,
            max_pages=None,
            langs=["English"],
            batch_multiplier=1,
        )

        if not full_text or not full_text.strip():
            print(f"  WARNING: Empty output for {pdf_path.name}")
            return None

        out_path.write_text(full_text, encoding="utf-8")
        return out_path

    except Exception as exc:
        print(f"  ERROR extracting {pdf_path}: {exc}")
        return None


def extract_all(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Extract all PDFs under data/raw/ for the given tickers/filing types.

    Returns:
        Dict mapping ticker → list of output .md paths.
    """
    from config.companies import MAG7_COMPANIES, SUPPORTED_FILING_TYPES

    tickers = tickers or list(MAG7_COMPANIES.keys())
    filing_types = filing_types or SUPPORTED_FILING_TYPES

    results: dict[str, list[Path]] = {}

    for ticker in tickers:
        ticker_dir = RAW_DIR / ticker
        if not ticker_dir.exists():
            print(f"  No raw data for {ticker}, skipping extraction.")
            results[ticker] = []
            continue

        pdf_files: list[Path] = []
        for ft in filing_types:
            pdf_files.extend(ticker_dir.rglob("*.pdf"))

        if not pdf_files:
            print(f"  No PDFs found for {ticker}.")
            results[ticker] = []
            continue

        md_paths: list[Path] = []
        for pdf in tqdm(pdf_files, desc=f"Extracting {ticker}"):
            result = extract_pdf(pdf, force=force)
            if result:
                md_paths.append(result)

        results[ticker] = md_paths
        print(f"  {ticker}: {len(md_paths)}/{len(pdf_files)} extracted")

    return results


def list_extracted_markdowns(tickers: list[str] | None = None) -> dict[str, list[Path]]:
    """Return dict of ticker → [.md paths] for already-extracted filings."""
    from config.companies import MAG7_COMPANIES
    tickers = tickers or list(MAG7_COMPANIES.keys())
    result: dict[str, list[Path]] = {}
    for ticker in tickers:
        ticker_dir = PROCESSED_DIR / ticker
        if ticker_dir.exists():
            result[ticker] = sorted(ticker_dir.rglob("*.md"))
        else:
            result[ticker] = []
    return result
