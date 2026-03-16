"""
Download SEC EDGAR filings (10-K and 10-Q) for Mag7 companies.

EDGAR sometimes returns .htm files instead of PDFs. After downloading,
any HTML files are converted to PDF using weasyprint before further processing.
"""

import os
import subprocess
from pathlib import Path

from sec_edgar_downloader import Downloader
from tqdm import tqdm

from config.companies import MAG7_COMPANIES, SUPPORTED_FILING_TYPES
from config.settings import settings

# Base directory for raw downloads
RAW_DIR = Path("data/raw")


def _get_downloader() -> Downloader:
    """Create an SEC EDGAR downloader with proper user agent."""
    return Downloader(
        company_name=settings.sec_user_agent_name,
        email_address=settings.sec_user_agent_email,
        save_location=str(RAW_DIR),
    )


def _html_to_pdf(html_path: Path) -> Path:
    """Convert an HTML file to PDF using weasyprint. Returns the PDF path."""
    pdf_path = html_path.with_suffix(".pdf")
    try:
        import weasyprint
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        html_path.unlink()  # remove source HTML after successful conversion
        print(f"  Converted HTML → PDF: {pdf_path.name}")
    except Exception as exc:
        print(f"  WARNING: weasyprint conversion failed for {html_path.name}: {exc}")
        return html_path  # return original if conversion fails
    return pdf_path


def _normalize_downloads(ticker_dir: Path) -> None:
    """Walk the ticker directory and convert any .htm/.html files to PDF."""
    for htm_file in list(ticker_dir.rglob("*.htm")) + list(ticker_dir.rglob("*.html")):
        _html_to_pdf(htm_file)


def download_filings(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    after: str = "2023-12-31",
    limit: int = 5,
) -> dict[str, list[Path]]:
    """
    Download SEC filings for the given tickers and filing types.

    Args:
        tickers: List of ticker symbols. Defaults to all Mag7.
        filing_types: List of filing types ('10-K', '10-Q'). Defaults to both.
        after: Only download filings after this date (YYYY-MM-DD).
        limit: Maximum number of filings to download per ticker per filing type.

    Returns:
        Dict mapping ticker → list of downloaded PDF paths.
    """
    tickers = tickers or list(MAG7_COMPANIES.keys())
    filing_types = filing_types or SUPPORTED_FILING_TYPES

    dl = _get_downloader()
    results: dict[str, list[Path]] = {}

    for ticker in tqdm(tickers, desc="Downloading filings"):
        if ticker not in MAG7_COMPANIES:
            print(f"  WARNING: Unknown ticker {ticker}, skipping.")
            continue

        cik = MAG7_COMPANIES[ticker]["cik"]
        ticker_paths: list[Path] = []

        for filing_type in filing_types:
            print(f"  Downloading {filing_type} for {ticker} (CIK {cik})...")
            try:
                dl.get(
                    form=filing_type,
                    ticker_or_cik=ticker,
                    limit=limit,
                    after=after,
                    download_details=True,
                )
            except Exception as exc:
                print(f"  ERROR downloading {filing_type} for {ticker}: {exc}")
                continue

        # Normalize: convert any HTML files to PDF
        ticker_dir = RAW_DIR / ticker
        if ticker_dir.exists():
            _normalize_downloads(ticker_dir)
            # Collect all PDFs
            ticker_paths = list(ticker_dir.rglob("*.pdf"))

        results[ticker] = ticker_paths
        print(f"  {ticker}: {len(ticker_paths)} PDF(s) ready")

    return results


def list_downloaded_pdfs(tickers: list[str] | None = None) -> dict[str, list[Path]]:
    """Return a dict of ticker → [PDF paths] for already-downloaded filings."""
    tickers = tickers or list(MAG7_COMPANIES.keys())
    result: dict[str, list[Path]] = {}
    for ticker in tickers:
        ticker_dir = RAW_DIR / ticker
        if ticker_dir.exists():
            result[ticker] = sorted(ticker_dir.rglob("*.pdf"))
        else:
            result[ticker] = []
    return result


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    download_filings(tickers=tickers)
