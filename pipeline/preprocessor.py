"""
Parse marker-extracted Markdown into sections with metadata tagging.

NOTE: This module is FRAGILE — section headings vary by PDF formatting.
      Always test against actual marker output before indexing everything.
      The regex patterns here cover the most common patterns observed.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.companies import MAG7_COMPANIES, SECTION_CATEGORY_MAP

PROCESSED_DIR = Path("data/processed")


@dataclass
class FilingMetadata:
    ticker: str
    company_name: str
    namespace: str
    filing_type: str          # "10-K" or "10-Q"
    period: str               # e.g. "FY2024" or "Q3-2024"
    source_file: str          # relative path to .md file
    accession_number: str     # EDGAR accession number


@dataclass
class SectionDocument:
    section_id: str           # e.g. "item_1" or "item_7a"
    section_title: str        # e.g. "Item 1A. Risk Factors"
    category: str             # finance | legal | tech | personas | other
    text: str
    filing_meta: FilingMetadata


# ─── Regex patterns for section headings ─────────────────────────────────────
# EDGAR HTML (via markdownify) produces Item headings in these forms:
#
#   Item 1A.    Risk FactorsThe following...      <- title+body on same line
#   Item 3\.    Legal ProceedingsDigital...        <- markdownify escapes "."
#   ## Item 1. Business                            <- proper markdown heading
#   **Item 7.** Management's Discussion...         <- bold heading
#
# Strategy: match start-of-line "Item N" anchors, extract positions,
# then use those positions to split the document into section slices.

_ITEM_LINE_PATTERN = re.compile(
    # Matches Item headings at the start of a line in EDGAR HTML output.
    # Groups: (1) item_number e.g. "1A", (2) rest of line (title + body)
    r"^Item\s+(\d+[A-Za-z]?)[\\]?\.\s+(.*?)$",
    re.MULTILINE,
)

_MARKDOWN_HEADING_PATTERN = re.compile(
    r"^#{1,4}\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)[.\s:]*(.*)$",
    re.MULTILINE,
)

_BOLD_HEADING_PATTERN = re.compile(
    r"^\*\*(?:ITEM|Item)\s+(\d+[A-Za-z]?)[.\s:]*\**\s*(.*)$",
    re.MULTILINE,
)


def _extract_accession_number(md_path: Path) -> str:
    """
    Extract accession number from the file path.
    EDGAR directory structure: .../TICKER/FILING_TYPE/{accession}/...
    """
    parts = md_path.parts
    # Try to find the accession directory (format: XXXXXXXXXX-XX-XXXXXX or similar)
    for part in reversed(parts):
        if re.match(r"\d{10}-\d{2}-\d{6}", part):
            return part
        if re.match(r"\d{18}", part):
            return part
    # Fallback: use parent directory name
    return md_path.parent.name


def _infer_period(accession_number: str, filing_type: str, md_path: Path) -> str:
    """
    Infer the filing period from the accession number or directory structure.

    SEC accession numbers encode filing date: XXXXXXXXXX-YY-XXXXXX
    where YY is the last two digits of the year.
    """
    # Try to parse year from accession
    match = re.match(r"\d{10}-(\d{2})-\d{6}", accession_number)
    if match:
        yy = int(match.group(1))
        year = 2000 + yy if yy < 50 else 1900 + yy
        if filing_type == "10-K":
            return f"FY{year - 1}"  # 10-K filed in 2025 covers FY2024
        else:
            return f"Q-{year}"

    # Fallback: look for year pattern in path
    for part in md_path.parts:
        m = re.search(r"(202\d)", part)
        if m:
            year = int(m.group(1))
            if filing_type == "10-K":
                return f"FY{year - 1}"
            return f"Q-{year}"

    return "Unknown"


def _infer_filing_type_from_path(md_path: Path) -> str:
    """Infer filing type from directory structure."""
    path_str = str(md_path).upper()
    if "10-K" in path_str or "10K" in path_str:
        return "10-K"
    if "10-Q" in path_str or "10Q" in path_str:
        return "10-Q"
    return "10-K"  # default


def _clean_markdown(text: str) -> str:
    """Remove noise artifacts from marker output."""
    # Remove page number artifacts (common patterns: "Page 42", "- 42 -", lone numbers)
    text = re.sub(r"^\s*[-–]\s*\d+\s*[-–]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove excessive horizontal rules
    text = re.sub(r"^[-=_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove unicode artifacts common in PDFs
    text = text.replace("\x00", "").replace("\ufffd", "").replace("\u00a0", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def _determine_category(section_id: str, section_title: str) -> str:
    """Map a section identifier to a metadata category."""
    # Build a normalized lookup key from section_id (e.g. "item_1a" -> "item 1a")
    normalized = section_id.lower().replace("_", " ")

    # 1. Try exact match first (strips trailing whitespace from map keys)
    for key, category in SECTION_CATEGORY_MAP.items():
        if normalized == key.strip():
            return category

    # 2. Prefix match — longer (more specific) keys take priority
    sorted_items = sorted(SECTION_CATEGORY_MAP.items(), key=lambda kv: len(kv[0].strip()), reverse=True)
    for key, category in sorted_items:
        if normalized.startswith(key.strip()):
            return category

    # Also check by title keywords as fallback
    title_lower = section_title.lower()
    if any(w in title_lower for w in ["risk", "legal", "litigation", "compliance", "regulation"]):
        return "legal"
    if any(w in title_lower for w in ["revenue", "income", "financial", "cash", "earnings"]):
        return "finance"
    if any(w in title_lower for w in ["technology", "product", "research", "development", "innovation"]):
        return "tech"
    if any(w in title_lower for w in ["director", "officer", "compensation", "executive", "governance"]):
        return "personas"

    return "other"


def _extract_title_from_line(item_num: str, rest_of_line: str) -> str:
    """
    Extract a clean section title from the rest of a line like:
      'Risk FactorsThe following summarizes...'
      'Legal Proceedings'
    The heading title is words before the first sentence (i.e., before
    a word that starts lowercase after an uppercase word, or up to 80 chars).
    """
    rest = rest_of_line.strip()
    # Normalize multiple spaces
    rest = re.sub(r"\s+", " ", rest)

    # The title ends when we see a lowercase-starting word following content
    # that looks like running prose. Heuristic: take up to 80 chars of
    # title-cased / uppercase words before any lowercase continuation.
    words = rest.split()
    title_words = []
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if i > 0 and clean and clean[0].islower():
            break  # start of body text
        title_words.append(word)
        if len(" ".join(title_words)) > 80:
            break

    title = " ".join(title_words).strip(" .,")
    return f"Item {item_num}. {title}" if title else f"Item {item_num}"


def _normalize_item_headings(text: str) -> str:
    """
    Pre-process text so that Item N headings always start at a new line.
    EDGAR HTML often has 'PART IItem 1. Business' with no newline separator.
    """
    # Insert newline before "Item N" when not already at start of line
    # and not preceded by "Part" references in body text
    # Pattern: any non-newline chars immediately before "Item \d"
    text = re.sub(r"(?<!\n)(Item\s+\d+[A-Za-z]?\s*[\\]?\.)", r"\n\1", text)
    # Remove the duplicate if there was already a newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _split_into_sections(text: str) -> list[tuple[str, str, str]]:
    """
    Split markdown text into sections by Item headings.

    Returns list of (section_id, section_title, section_text) tuples.
    """
    # Normalize text to ensure Item headings start on their own lines
    text = _normalize_item_headings(text)

    # Collect matches from all patterns
    all_matches: list[tuple[int, str, str]] = []  # (position, item_num, title)

    # Primary: EDGAR HTML format — Item at start of line (most reliable)
    for m in _ITEM_LINE_PATTERN.finditer(text):
        item_num = m.group(1).strip()
        title = _extract_title_from_line(item_num, m.group(2))
        all_matches.append((m.start(), item_num, title))

    # Secondary: markdown headings (## Item N ...)
    for m in _MARKDOWN_HEADING_PATTERN.finditer(text):
        item_num = m.group(1).strip()
        title_part = m.group(2).strip()
        all_matches.append((m.start(), item_num, f"Item {item_num}. {title_part}"))

    # Tertiary: bold headings (**Item N ...**)
    for m in _BOLD_HEADING_PATTERN.finditer(text):
        item_num = m.group(1).strip()
        title_part = m.group(2).strip()
        all_matches.append((m.start(), item_num, f"Item {item_num}. {title_part}"))

    if not all_matches:
        return [("item_0", "Full Document", text)]

    # Sort by position, deduplicate (keep first match per position)
    all_matches.sort(key=lambda x: x[0])
    unique_matches: list[tuple[int, str, str]] = []
    seen_positions: set[int] = set()
    seen_items: set[str] = set()  # avoid TOC duplicates (same Item N appears twice)
    for pos, item_num, title in all_matches:
        if pos in seen_positions:
            continue
        # Skip if this item number already appeared (table of contents entry
        # comes before the actual section — keep the LATER occurrence if the
        # earlier one had very short following text)
        seen_positions.add(pos)
        unique_matches.append((pos, item_num, title))

    # For items that appear twice (TOC + body), keep the body (later one)
    # by checking whether the text between consecutive matches is substantial
    filtered: list[tuple[int, str, str]] = []
    for i, (pos, item_num, title) in enumerate(unique_matches):
        next_pos = unique_matches[i + 1][0] if i + 1 < len(unique_matches) else len(text)
        text_slice = text[pos:next_pos]
        word_count = len(text_slice.split())

        # Skip entries with < 30 words — likely TOC entries
        if word_count < 30:
            continue

        # If we already have this item number with substantial content, skip
        prev = next((f for f in filtered if f[1] == item_num), None)
        if prev:
            # Replace with this one only if it has more text
            prev_next = unique_matches[unique_matches.index((prev[0], prev[1], prev[2])) + 1][0] \
                if unique_matches.index((prev[0], prev[1], prev[2])) + 1 < len(unique_matches) \
                else len(text)
            if word_count > len(text[prev[0]:prev_next].split()):
                filtered.remove(prev)
                filtered.append((pos, item_num, title))
        else:
            filtered.append((pos, item_num, title))

    filtered.sort(key=lambda x: x[0])

    # Extract section texts
    sections: list[tuple[str, str, str]] = []
    for i, (pos, item_num, title) in enumerate(filtered):
        next_pos = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        section_text = text[pos:next_pos].strip()
        section_id = f"item_{item_num.lower()}"
        sections.append((section_id, title, section_text))

    return sections if sections else [("item_0", "Full Document", text)]


def parse_filing(md_path: Path, ticker: str) -> list[SectionDocument]:
    """
    Parse a single extracted Markdown file into SectionDocument objects.

    Args:
        md_path: Path to the .md file (output of extractor.py).
        ticker: Stock ticker symbol.

    Returns:
        List of SectionDocument objects, one per detected section.
    """
    if ticker not in MAG7_COMPANIES:
        raise ValueError(f"Unknown ticker: {ticker}")

    company_info = MAG7_COMPANIES[ticker]
    raw_text = md_path.read_text(encoding="utf-8", errors="replace")
    cleaned_text = _clean_markdown(raw_text)

    accession_number = _extract_accession_number(md_path)
    filing_type = _infer_filing_type_from_path(md_path)
    period = _infer_period(accession_number, filing_type, md_path)

    filing_meta = FilingMetadata(
        ticker=ticker,
        company_name=company_info["name"],
        namespace=company_info["namespace"],
        filing_type=filing_type,
        period=period,
        source_file=str(md_path),
        accession_number=accession_number,
    )

    sections = _split_into_sections(cleaned_text)
    documents: list[SectionDocument] = []

    for section_id, section_title, section_text in sections:
        category = _determine_category(section_id, section_title)
        doc = SectionDocument(
            section_id=section_id,
            section_title=section_title,
            category=category,
            text=section_text,
            filing_meta=filing_meta,
        )
        documents.append(doc)

    return documents


def parse_all(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
) -> dict[str, list[SectionDocument]]:
    """
    Parse all extracted markdowns for the given tickers.

    Returns:
        Dict mapping ticker → list of SectionDocuments.
    """
    from config.companies import MAG7_COMPANIES, SUPPORTED_FILING_TYPES

    tickers = tickers or list(MAG7_COMPANIES.keys())
    filing_types = filing_types or SUPPORTED_FILING_TYPES

    results: dict[str, list[SectionDocument]] = {}

    for ticker in tickers:
        ticker_dir = PROCESSED_DIR / ticker
        if not ticker_dir.exists():
            print(f"  No processed data for {ticker}, skipping.")
            results[ticker] = []
            continue

        docs: list[SectionDocument] = []
        md_files = list(ticker_dir.rglob("*.md"))

        for md_path in md_files:
            # Filter by filing type if needed
            path_str = str(md_path).upper()
            relevant = any(ft.replace("-", "") in path_str.replace("-", "") for ft in filing_types)
            if not relevant:
                continue

            try:
                filing_docs = parse_filing(md_path, ticker)
                docs.extend(filing_docs)
            except Exception as exc:
                print(f"  ERROR parsing {md_path}: {exc}")

        results[ticker] = docs
        print(f"  {ticker}: {len(docs)} section(s) parsed from {len(md_files)} file(s)")

    return results
