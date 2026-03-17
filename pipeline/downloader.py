"""
Download SEC EDGAR filings (10-K and 10-Q) for Mag7 companies.

sec-edgar-downloader v5 saves files to:
  {download_folder}/sec-edgar-filings/{TICKER}/{FILING_TYPE}/{accession}/

EDGAR typically returns HTML filings. The extractor handles HTML directly
via markdownify -- no PDF conversion step needed.
"""

from pathlib import Path

from sec_edgar_downloader import Downloader
from tqdm import tqdm

from config.companies import MAG7_COMPANIES, SUPPORTED_FILING_TYPES
from config.settings import settings

# sec-edgar-downloader v5 always appends sec-edgar-filings/ to the download folder
RAW_DIR = Path("data/raw")
EDGAR_DIR = RAW_DIR / "sec-edgar-filings"


def _get_downloader() -> Downloader:
    return Downloader(
        company_name=settings.sec_user_agent_name,
        email_address=settings.sec_user_agent_email,
        download_folder=str(RAW_DIR),
    )


def download_filings(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    after: str = "2023-12-31",
    limit: int = 5,
) -> dict[str, list[Path]]:
    """
    Download SEC filings for the given tickers and filing types.

    Returns:
        Dict mapping ticker -> list of downloaded file paths.
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

        # Collect all downloaded files
        ticker_dir = EDGAR_DIR / ticker
        if ticker_dir.exists():
            files = [
                f for f in ticker_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in (".html", ".htm", ".pdf", ".txt")
            ]
            results[ticker] = files
            print(f"  {ticker}: {len(files)} file(s) downloaded")
        else:
            results[ticker] = []

    return results


def list_downloaded_files(tickers: list[str] | None = None) -> dict[str, list[Path]]:
    """Return a dict of ticker -> [file paths] for already-downloaded filings."""
    tickers = tickers or list(MAG7_COMPANIES.keys())
    result: dict[str, list[Path]] = {}
    for ticker in tickers:
        ticker_dir = EDGAR_DIR / ticker
        if ticker_dir.exists():
            result[ticker] = sorted(ticker_dir.rglob("*.*"))
        else:
            result[ticker] = []
    return result


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    download_filings(tickers=tickers)
