"""
Extract text from SEC filing documents (PDF or HTML) to Markdown.

- HTML files (from EDGAR): converted via markdownify (no native deps needed)
- PDF files: converted via marker-pdf (loaded lazily, expensive first load)

Extraction is idempotent: skips files where the .md output already exists.
"""

from pathlib import Path
from typing import Optional

from tqdm import tqdm

RAW_DIR = Path("data/raw")
EDGAR_DIR = RAW_DIR / "sec-edgar-filings"
PROCESSED_DIR = Path("data/processed")

# Lazy-loaded marker models
_marker_models = None


def _get_marker_models():
    """Load marker-pdf models once and cache them."""
    global _marker_models
    if _marker_models is None:
        print("Loading marker-pdf models (first call -- may be slow)...")
        from marker.models import create_model_dict
        _marker_models = create_model_dict()
        print("marker-pdf models loaded.")
    return _marker_models


def _source_to_output_path(source_path: Path) -> Path:
    """
    Compute the output .md path for a given source file.

    Maps:  data/raw/sec-edgar-filings/{TICKER}/{FILING_TYPE}/{accession}/file.html
       ->  data/processed/{TICKER}/{FILING_TYPE}/{accession}/{stem}.md
    """
    try:
        rel = source_path.relative_to(EDGAR_DIR)
    except ValueError:
        try:
            rel = source_path.relative_to(RAW_DIR)
        except ValueError:
            rel = Path(*source_path.parts[-4:])

    out_dir = PROCESSED_DIR / rel.parent
    return out_dir / (source_path.stem + ".md")


def _html_to_markdown(html_path: Path) -> Optional[str]:
    """Convert an EDGAR HTML filing to Markdown using BeautifulSoup + markdownify."""
    try:
        from bs4 import BeautifulSoup, Tag
        from markdownify import markdownify as md
        import re

        html_content = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove XBRL / inline metadata elements
        # iXBRL filings embed machine-readable data in hidden elements
        # Guard against non-Tag nodes (e.g. ProcessingInstruction) that have attrs=None
        for tag in soup.find_all(style=True):
            if not isinstance(tag, Tag) or not tag.attrs:
                continue
            style = tag.get("style", "") or ""
            if "display:none" in style.replace(" ", "") or "display: none" in style:
                tag.decompose()

        # Remove ix: namespace tags (inline XBRL) — keep their children
        for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
            tag.unwrap()

        # Remove script, style, head, and other non-content tags
        for tag_name in ["script", "style", "head", "meta", "link", "noscript"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Convert to markdown
        clean_html = str(soup)
        markdown = md(
            clean_html,
            heading_style="ATX",
            bullets="-",
            convert_tables=True,
        )

        # Post-process: collapse excessive blank lines, strip leading garbage
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        # Find first real content — skip until we hit a proper section or heading
        # (common EDGAR structure: cover page starts with SEC header)
        start_markers = [
            "UNITED STATES", "SECURITIES AND EXCHANGE", "FORM 10-K", "FORM 10-Q",
            "## ", "# ", "Item 1", "ITEM 1",
        ]
        lines = markdown.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if any(marker in line for marker in start_markers):
                # Back up a bit to catch any heading just before
                start_idx = max(0, i - 2)
                break

        markdown = "\n".join(lines[start_idx:]).strip()
        return markdown if markdown else None

    except Exception as exc:
        print(f"  ERROR converting HTML {html_path.name}: {exc}")
        return None


def _pdf_to_markdown(pdf_path: Path) -> Optional[str]:
    """Convert a PDF filing to Markdown using marker-pdf."""
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
        return full_text.strip() if full_text else None
    except Exception as exc:
        print(f"  ERROR extracting PDF {pdf_path.name}: {exc}")
        return None


def extract_file(source_path: Path, force: bool = False) -> Optional[Path]:
    """
    Convert a single source file (HTML or PDF) to Markdown.

    Args:
        source_path: Path to the source file (.html, .htm, or .pdf).
        force: Re-extract even if the .md file already exists.

    Returns:
        Path to the output .md file, or None on failure.
    """
    out_path = _source_to_output_path(source_path)

    if out_path.exists() and not force:
        return out_path  # idempotent

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower()
    if suffix in (".html", ".htm"):
        text = _html_to_markdown(source_path)
    elif suffix == ".pdf":
        text = _pdf_to_markdown(source_path)
    else:
        print(f"  Unsupported format: {source_path.suffix}")
        return None

    if not text:
        print(f"  WARNING: Empty output for {source_path.name}")
        return None

    out_path.write_text(text, encoding="utf-8")
    return out_path


# Keep backward-compatible alias
extract_pdf = extract_file


def extract_all(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Extract all source files under EDGAR_DIR for the given tickers.

    Returns:
        Dict mapping ticker -> list of output .md paths.
    """
    from config.companies import MAG7_COMPANIES

    tickers = tickers or list(MAG7_COMPANIES.keys())

    results: dict[str, list[Path]] = {}

    for ticker in tickers:
        ticker_dir = EDGAR_DIR / ticker
        if not ticker_dir.exists():
            print(f"  No raw data for {ticker}, skipping extraction.")
            results[ticker] = []
            continue

        # Collect all extractable files
        source_files: list[Path] = []
        for pattern in ("*.html", "*.htm", "*.pdf"):
            source_files.extend(ticker_dir.rglob(pattern))

        # Filter by filing type if requested
        if filing_types:
            source_files = [
                f for f in source_files
                if any(ft.replace("-", "") in str(f).upper().replace("-", "") for ft in filing_types)
            ]

        # Skip full-submission.txt (not extractable)
        source_files = [f for f in source_files if f.name != "full-submission.txt"]

        if not source_files:
            print(f"  No extractable files found for {ticker}.")
            results[ticker] = []
            continue

        md_paths: list[Path] = []
        for src in tqdm(source_files, desc=f"Extracting {ticker}"):
            result = extract_file(src, force=force)
            if result:
                md_paths.append(result)

        results[ticker] = md_paths
        print(f"  {ticker}: {len(md_paths)}/{len(source_files)} extracted")

    return results


def list_extracted_markdowns(tickers: list[str] | None = None) -> dict[str, list[Path]]:
    """Return dict of ticker -> [.md paths] for already-extracted filings."""
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
