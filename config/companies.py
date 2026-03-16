from typing import TypedDict


class CompanyInfo(TypedDict):
    name: str
    cik: str
    namespace: str


# Magnificent 7 — ticker → company info
MAG7_COMPANIES: dict[str, CompanyInfo] = {
    "AAPL": {"name": "Apple Inc.", "cik": "0000320193", "namespace": "apple"},
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019", "namespace": "microsoft"},
    "GOOGL": {"name": "Alphabet Inc.", "cik": "0001652044", "namespace": "alphabet"},
    "AMZN": {"name": "Amazon.com Inc.", "cik": "0001018724", "namespace": "amazon"},
    "META": {"name": "Meta Platforms Inc.", "cik": "0001326801", "namespace": "meta"},
    "NVDA": {"name": "NVIDIA Corporation", "cik": "0001045810", "namespace": "nvidia"},
    "TSLA": {"name": "Tesla Inc.", "cik": "0001318605", "namespace": "tesla"},
}

# Section heading → metadata category
# Keys are lowercase normalized section titles (partial matches)
SECTION_CATEGORY_MAP: dict[str, str] = {
    # Business / Technology
    "item 1 ": "tech",          # Item 1: Business
    "item 1\n": "tech",
    "item 1.": "tech",
    "item 2 ": "tech",          # Item 2: Properties (also tech/ops)
    "item 2\n": "tech",
    "item 2.": "tech",

    # Legal / Risk
    "item 1a": "legal",         # Item 1A: Risk Factors
    "item 3 ": "legal",         # Item 3: Legal Proceedings
    "item 3\n": "legal",
    "item 3.": "legal",
    "item 9a": "legal",         # Item 9A: Controls and Procedures
    "item 4 ": "legal",         # Item 4: Mine Safety (placeholder)

    # Finance
    "item 7 ": "finance",       # Item 7: MD&A
    "item 7\n": "finance",
    "item 7.": "finance",
    "item 7a": "finance",       # Item 7A: Quantitative Disclosures
    "item 8 ": "finance",       # Item 8: Financial Statements
    "item 8\n": "finance",
    "item 8.": "finance",
    "item 6 ": "finance",       # Item 6: Selected Financial Data
    "item 5 ": "finance",       # Item 5: Market for Registrant

    # Personas (governance, execs, compensation)
    "item 10": "personas",      # Item 10: Directors, Executive Officers
    "item 11": "personas",      # Item 11: Executive Compensation
    "item 12": "personas",      # Item 12: Security Ownership
    "item 13": "personas",      # Item 13: Certain Relationships
    "item 14": "personas",      # Item 14: Principal Accountant Fees
}

# Ordered list of filing types supported
SUPPORTED_FILING_TYPES = ["10-K", "10-Q"]

# All tickers as list for convenience
ALL_TICKERS = list(MAG7_COMPANIES.keys())
