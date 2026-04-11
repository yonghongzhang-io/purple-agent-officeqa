from __future__ import annotations

import logging
import os
import re


logger = logging.getLogger(__name__)


_treasury_source_files: list[str] = []
_treasury_content_cache: dict[str, str] = {}
_treasury_preview_cache: dict[str, str] = {}
_treasury_norm_cache: dict[str, str] = {}
_treasury_table_cache: dict[str, list[dict[str, object]]] = {}
_treasury_heading_cache: dict[str, str] = {}
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "from", "on", "by",
    "what", "which", "when", "where", "how", "much", "many", "was", "were", "is",
    "are", "did", "does", "do", "with", "at", "as", "that", "this", "these",
    "those", "using", "based", "according", "reported", "specifically", "only",
    "calendar", "monthly", "individual", "values", "between", "fiscal", "year",
    "years", "month", "months", "treasury", "bulletin", "table", "amount",
    "value", "number", "nominal", "dollars", "millions", "billions",
    "trillions", "report", "question", "given", "reporting", "date",
    "provided", "all", "associated", "activities", "activity", "corresponding",
    "rounded", "nearest", "hundredths", "place", "just", "each", "inclusive",
    "including", "not",
}

_DOMAIN_PHRASE_EXPANSIONS = {
    r"\btreasury balance\b": [
        "treasury balance",
        "operating cash balance",
        "status of treasury",
        "cash balance",
    ],
    r"\bpublic debt\b": [
        "public debt",
        "debt outstanding",
        "interest-bearing debt",
        "public debt outstanding",
    ],
    r"\b(receipts?|federal receipts|budget receipts)\b": [
        "federal receipts",
        "budget receipts",
        "receipts by source",
        "total receipts",
    ],
    r"\b(outlays?|budget outlays)\b": [
        "budget outlays",
        "federal outlays",
        "outlays by function",
        "total outlays",
    ],
    r"\bexchange stabilization fund\b|\besf\b": [
        "exchange stabilization fund",
        "esf",
        "foreign currency holdings",
    ],
    r"\boptions? positions?\b": [
        "options positions",
        "net options positions",
    ],
    r"\bsecurities\b": [
        "securities outstanding",
        "marketable securities",
        "treasury securities",
    ],
}

_TABLE_FAMILY_MATCHERS: dict[str, dict[str, list[str]]] = {
    "fd": {
        "codes": ["fd"],
        "keywords": [
            "federal debt",
            "gross federal debt",
            "public debt",
            "debt outstanding",
            "debt subject to statutory limitation",
            "interest-bearing marketable public debt securities",
            "gross debt",
            "summary of federal debt",
            "summary of public debt and guaranteed agency securities",
        ],
    },
    "ffo": {
        "codes": ["ffo"],
        "keywords": [
            "receipts by source",
            "budget outlays by function",
            "federal receipts",
            "budget receipts",
            "outlays by function",
            "interest cost",
            "net of refunds",
            "fiscal operations",
        ],
    },
    "cm": {
        "codes": ["cm"],
        "keywords": [
            "claims on foreigners",
            "claims owed by a country",
            "country breakdown",
            "capital movements",
            "reported by banks",
            "liabilities to foreigners",
        ],
    },
    "esf": {
        "codes": ["esf"],
        "keywords": [
            "exchange stabilization fund",
            "esf",
            "foreign currency holdings",
            "special drawing rights",
        ],
    },
    "fcp": {
        "codes": ["fcp"],
        "keywords": [
            "foreign exchange and securities investments",
            "net options positions",
            "japanese yen",
            "british pound",
            "euro",
            "foreign currency",
            "options positions",
        ],
    },
    "auction": {
        "codes": ["pdo", "tso"],
        "keywords": [
            "13-week",
            "26-week",
            "treasury-bill rates",
            "treasury bill rates",
            "auction",
            "accepted tenders",
            "issue date",
            "maturing",
            "2-year treasury notes",
            "offerings of bills",
            "offerings of notes",
            "public debt operations",
        ],
    },
    "bill_rates": {
        "codes": ["pdo"],
        "keywords": [
            "13-week",
            "26-week",
            "91-day",
            "182-day",
            "treasury-bill rates",
            "average rate",
            "average issuing rate",
            "investment rate",
            "discount rate",
            "equivalent coupon issue yield",
        ],
    },
    "auction_results": {
        "codes": ["pdo"],
        "keywords": [
            "accepted tenders",
            "amount of bids accepted",
            "bids submitted",
            "competitive",
            "noncompetitive",
            "rollover tenders",
            "offerings of notes",
            "offerings of bills",
            "public offerings of marketable securities other than regular weekly treasury bills",
        ],
    },
    "maturity_schedule": {
        "codes": ["tso"],
        "keywords": [
            "maturity schedule",
            "maturing",
            "description of securities",
            "amount outstanding",
            "held by u.s. govt. accounts",
            "held by federal reserve banks",
            "all other investors",
            "interest-bearing marketable public debt securities",
        ],
    },
    "agency_expense": {
        "codes": [],
        "keywords": [
            "veterans administration",
            "national defense",
            "public works",
            "expenditures",
            "federal government expenditures",
            "budget expenditures",
            "analysis of national defense expenditures",
            "budget receipts and expenditures",
            "general government expenditures",
        ],
    },
    "calendar_defense": {
        "codes": [],
        "keywords": [
            "analysis of national defense expenditures",
            "expenditures for national defense and related activities",
            "cash income and outgo of the treasury, by major classifications",
            "budget expenditures classified as general, by major functions",
            "calendar yr",
            "calendar year",
            "individual calendar months",
        ],
    },
    "general_expenditures": {
        "codes": [],
        "keywords": [
            "analysis of general expenditures",
            "departments and agencies",
            "budget expenditures",
            "general government expenditures",
            "public works",
        ],
    },
    "national_defense": {
        "codes": [],
        "keywords": [
            "analysis of national defense expenditures",
            "national defense expenditures",
            "national defense",
        ],
    },
    "veterans_public_works": {
        "codes": [],
        "keywords": [
            "veterans administration",
            "public works",
            "analysis of general expenditures",
            "general expenditures",
            "departments and agencies",
            "budget receipts and expenditures",
        ],
    },
    "tax_receipts": {
        "codes": [],
        "keywords": [
            "individual income tax receipts",
            "net of refunds",
            "internal revenue collections",
            "income tax receipts",
            "tax receipts",
        ],
    },
    "individual_income_tax": {
        "codes": [],
        "keywords": [
            "individual income tax receipts",
            "internal revenue collections",
            "summary by principal sources",
            "net of refunds",
        ],
    },
}


def _normalize_text(text: str) -> str:
    """Normalize text for lexical matching."""
    return re.sub(r"['`’]", "", text.lower())


def _extract_table_codes(question: str) -> list[str]:
    """Extract Treasury-style table codes such as FFO-3 or CM-II-2."""
    codes: list[str] = []
    for code in re.findall(r"[A-Z]{1,6}-[A-Z]?-?\d+[A-Z]?", question):
        lowered = code.lower()
        if lowered not in codes:
            codes.append(lowered)
    return codes


def _extract_named_entities(question: str) -> list[str]:
    """Extract multi-word title-cased entities from a question."""
    entities: list[str] = []
    patterns = [
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
        r"[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|of|the|for|in|to))+\s+[A-Z][a-z]+",
    ]
    for pattern in patterns:
        for entity in re.findall(pattern, question):
            lowered = entity.lower()
            if lowered not in entities and len(lowered) >= 6:
                entities.append(lowered)
    return entities


def _extract_quoted_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    matches = re.findall(r"['\u2018\u2019]([^'\u2018\u2019]{3,})['\u2018\u2019]|\"([^\"]{3,})\"", question)
    for p1, p2 in matches:
        phrase = (p1 or p2).lower().strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _extract_focus_phrases(question: str) -> list[str]:
    """Extract compact finance phrases that identify the target table family."""
    patterns = [
        r"\bnational defense expenditures\b",
        r"\bnational defense and related activities\b",
        r"\bnational defense\b",
        r"\bveterans administration\b",
        r"\bpublic works\b",
        r"\bgross federal debt\b",
        r"\bpublic debt\b",
        r"\bdebt outstanding\b",
        r"\bsummary of public debt\b",
        r"\bindividual income tax receipts\b",
        r"\binternal revenue collections\b",
        r"\bfederal receipts\b",
        r"\bbudget outlays\b",
        r"\bnet options positions\b",
        r"\bforeign exchange and securities investments\b",
        r"\bjapanese yen\b",
        r"\bbritish pound\b",
        r"\bexchange stabilization fund\b",
        r"\b13-week\b",
        r"\b26-week\b",
        r"\b2-year treasury notes?\b",
        r"\btreasury[- ]bill rates?\b",
        r"\baccepted tenders\b",
        r"\bamount of bids accepted\b",
        r"\bnoncash rollover tenders\b",
        r"\bcompetitive\b",
        r"\bnoncompetitive\b",
        r"\baverage issuing rate\b",
        r"\binvestment rate\b",
        r"\bdiscount rate\b",
        r"\bequivalent coupon issue yield\b",
        r"\bofferings of bills\b",
        r"\bdescription of securities\b",
        r"\ball other investors\b",
        r"\bamount outstanding\b",
        r"\bmarketable securities\b",
        r"\binterest[- ]bearing marketable public debt securities\b",
        r"\bclaims owed\b",
        r"\bclaims on foreigners\b",
        r"\btotal claims by type and country\b",
        r"\btotal liabilities by type and country\b",
        r"\bpublic offerings of marketable securities other than regular weekly treasury bills\b",
    ]
    lowered = question.lower()
    phrases: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            phrase = match.group(0).strip()
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _looks_like_explanatory_prose(line: str) -> bool:
    """Heuristic to avoid treating nearby prose as a table title."""
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped:
        return False
    if _is_exact_table_title(stripped):
        return False
    if re.match(r"^section\s+[ivxlcdm0-9]+", lower):
        return False
    if len(stripped) > 140:
        return True
    if re.search(r"\b(shows?|lists?|reports?|contains?|presents?|summarizes?|describes?)\b", lower):
        return True
    if lower.count(".") >= 2:
        return True
    if len(lower.split()) > 18 and not re.match(r"^(?:table|chart|statement)\b", lower):
        return True
    return False


def _is_exact_table_title(line: str) -> bool:
    """Detect clean Treasury table title lines such as 'Table FD-1. - ...'."""
    stripped = line.strip()
    return bool(
        re.match(
            r"^(?:TABLE|Table)\s+[A-Z0-9]{1,6}(?:[-.][A-Z0-9]+)*(?:\s*[—-]|\.\s*-|\.)",
            stripped,
        )
    )


def _looks_like_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _looks_like_explanatory_prose(stripped):
        return False
    if re.match(r"^section\s+[ivxlcdm0-9]+", stripped, re.IGNORECASE):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    upper_ratio = (
        sum(1 for ch in letters if ch.isupper()) / len(letters)
        if letters
        else 0.0
    )
    return (
        (upper_ratio >= 0.7 and len(stripped) <= 100)
        or (_heading_signal(stripped) >= 2 and len(stripped) <= 120)
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _extract_clean_table_title(prefix_lines: list[str], start_line: int) -> tuple[str, str]:
    """Extract a clean table title and unit hint from lines above a pipe-table block."""
    cleaned = [line.strip() for line in prefix_lines if line.strip()]
    unit_hint = ""
    for line in reversed(cleaned):
        lower = line.lower()
        if re.search(r"\b(in|amounts in)\s+(millions?|billions?|trillions?|thousands?)\b", lower):
            unit_hint = line
            break

    exact_idx = None
    for idx in range(len(cleaned) - 1, -1, -1):
        if _is_exact_table_title(cleaned[idx]) and not _looks_like_explanatory_prose(cleaned[idx]):
            exact_idx = idx
            break

    title_parts: list[str] = []
    if exact_idx is not None:
        if exact_idx > 0 and _looks_like_section_header(cleaned[exact_idx - 1]):
            title_parts.append(cleaned[exact_idx - 1])
        title_parts.append(cleaned[exact_idx])
    else:
        for line in reversed(cleaned):
            if _looks_like_section_header(line):
                title_parts.append(line)
                if len(title_parts) >= 2:
                    break
        title_parts.reverse()

    title = " | ".join(_dedupe_preserve_order(title_parts))
    if title and _normalize_text(title) == _normalize_text(unit_hint):
        title = ""
    if not title:
        title = f"Table block near line {start_line}"
    return title, unit_hint


def _detect_table_families(question: str, profile: dict[str, object] | None = None) -> list[str]:
    """Infer likely Treasury table families from the question semantics."""
    q_lower = question.lower()
    profile = profile or {}
    families: list[str] = []

    def add(name: str) -> None:
        if name not in families:
            families.append(name)

    if re.search(r"\b(public debt|gross federal debt|debt outstanding|debt subject to statutory limitation|fha)\b", q_lower):
        add("fd")
    if re.search(r"\b(receipts?|budget outlays|federal interest cost|net of refunds|budget receipts)\b", q_lower):
        add("ffo")
    if re.search(r"\b(individual income tax receipts|income tax receipts|net of refunds|internal revenue)\b", q_lower):
        add("tax_receipts")
    if re.search(r"\bindividual income tax receipts\b", q_lower):
        add("individual_income_tax")
    if re.search(r"\b(claims owed|claims on foreigners|country-level|capital movements|foreigners reported by banks)\b", q_lower):
        add("cm")
    if re.search(r"\b(exchange stabilization fund|esf)\b", q_lower):
        add("esf")
    if re.search(r"\b(japanese yen|british pound|foreign exchange and securities investments|options positions?|foreign currency)\b", q_lower):
        add("fcp")
    if re.search(r"\b(13-week|26-week|treasury-bill|auction|accepted tenders|issue date|maturing)\b", q_lower):
        add("auction")
    if profile.get("wants_bill_rate_data"):
        add("bill_rates")
    if profile.get("wants_auction_bid_data"):
        add("auction_results")
    if profile.get("wants_maturity_schedule"):
        add("maturity_schedule")
    if re.search(r"\b(veterans administration|national defense|public works|expenditures?)\b", q_lower):
        add("agency_expense")
    if (profile.get("wants_calendar_year") or profile.get("wants_monthly_series")) and re.search(
        r"\bnational defense\b", q_lower
    ):
        add("calendar_defense")
    if re.search(r"\b(expenditures?|departments and agencies|general expenditures)\b", q_lower):
        add("general_expenditures")
    if re.search(r"\bnational defense\b", q_lower):
        add("national_defense")
    if re.search(r"\b(veterans administration|public works)\b", q_lower):
        add("veterans_public_works")

    if profile.get("expects_regression") and "receipts" in q_lower:
        add("ffo")
        add("tax_receipts")
    if profile.get("expects_list") and "debt" in q_lower:
        add("fd")

    return families


def _resolve_treasury_data_dir() -> str:
    """Resolve the first available Treasury data directory across common layouts."""
    candidates = [
        os.environ.get("TREASURY_DATA_DIR"),
        "/data/treasury",
        os.path.join(os.getcwd(), "treasury_data"),
        os.path.join(os.getcwd(), "..", "treasury_data"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.environ.get("TREASURY_DATA_DIR", "/data/treasury")


_treasury_data_dir = _resolve_treasury_data_dir()


def _list_source_files() -> list[str]:
    """List all Treasury Bulletin .txt files in the data directory (cached)."""
    global _treasury_source_files
    if _treasury_source_files:
        return _treasury_source_files
    try:
        _treasury_source_files = sorted(
            f for f in os.listdir(_treasury_data_dir)
            if f.endswith(".txt") and f.startswith("treasury_bulletin_")
        )
        logger.info(f"Found {len(_treasury_source_files)} Treasury source files")
    except Exception as e:
        logger.warning(f"Failed to list Treasury source files: {e}")
    return _treasury_source_files


def _load_source_preview(filename: str, preview_chars: int | None = None) -> str:
    """Load and cache a sampled preview for lexical source ranking.

    We include head/middle/tail slices so later tables can influence file ranking
    without sending the full bulletin into the ranker.
    """
    cached = _treasury_preview_cache.get(filename)
    if cached is not None:
        return cached

    fpath = os.path.join(_treasury_data_dir, filename)
    try:
        if preview_chars is None:
            preview_chars = int(os.environ.get("SOURCE_PREVIEW_CHARS", "8000"))
        else:
            preview_chars = int(os.environ.get("SOURCE_PREVIEW_CHARS", str(preview_chars)))
        content = _load_source_content(filename)
        if len(content) <= preview_chars:
            preview = content
        else:
            chunk = max(2000, preview_chars // 3)
            head = content[:chunk]
            mid_start = max(0, (len(content) // 2) - (chunk // 2))
            middle = content[mid_start:mid_start + chunk]
            tail = content[-chunk:]
            preview = "\n".join([head, middle, tail])
    except Exception as e:
        logger.warning(f"Failed to read preview for {fpath}: {e}")
        preview = ""
    _treasury_preview_cache[filename] = preview
    return _treasury_preview_cache[filename]


def _load_source_content(filename: str) -> str:
    """Load and cache the full content of a Treasury bulletin."""
    cached = _treasury_content_cache.get(filename)
    if cached is not None:
        return cached

    fpath = os.path.join(_treasury_data_dir, filename)
    try:
        with open(fpath) as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read content for {fpath}: {e}")
        content = ""
    _treasury_content_cache[filename] = content
    return content


def _load_normalized_content(filename: str) -> str:
    """Load and cache full normalized content for lexical ranking."""
    cached = _treasury_norm_cache.get(filename)
    if cached is not None:
        return cached

    try:
        content = _load_source_content(filename)
        normalized = _normalize_text(content)
    except Exception as e:
        logger.warning(f"Failed to normalize content for {filename}: {e}")
        normalized = ""
    _treasury_norm_cache[filename] = normalized
    return normalized


def _extract_table_blocks(filename: str) -> list[dict[str, object]]:
    """Extract pipe-table blocks with nearby headings and unit hints."""
    cached = _treasury_table_cache.get(filename)
    if cached is not None:
        return cached

    content = _load_source_content(filename)
    if not content:
        _treasury_table_cache[filename] = []
        _treasury_heading_cache[filename] = ""
        return []

    lines = [line.rstrip("\n") for line in content.splitlines()]
    tables: list[dict[str, object]] = []
    headings: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if "|" not in stripped or stripped.count("|") < 2:
            idx += 1
            continue

        start = idx
        block_lines: list[str] = []
        while idx < len(lines):
            current = lines[idx].rstrip()
            current_stripped = current.strip()
            if not current_stripped:
                if block_lines:
                    break
                idx += 1
                start = idx
                continue
            if "|" not in current_stripped or current_stripped.count("|") < 2:
                break
            block_lines.append(current)
            idx += 1

        if len(block_lines) < 2:
            if idx == start:
                idx += 1
            continue

        context_start = max(0, start - 16)
        prefix_lines = [lines[p].strip() for p in range(context_start, start) if lines[p].strip()]
        title, unit_hint = _extract_clean_table_title(prefix_lines, start + 1)
        norm_title = _normalize_text(title)
        table_text = "\n".join(block_lines).strip()
        norm_text = _normalize_text(table_text)
        table_code_match = re.search(r"\b([A-Z]{1,6}-[A-Z]?-?\d+[A-Z]?)\b", title)
        is_contents = "table of contents" in norm_title or "contents" in norm_title

        if title and title not in headings:
            headings.append(title)

        tables.append(
            {
                "title": title,
                "unit_hint": unit_hint,
                "text": table_text,
                "norm_title": norm_title,
                "norm_text": norm_text,
                "table_code": table_code_match.group(1).lower() if table_code_match else "",
                "start": start,
                "end": idx,
                "pipe_lines": len(block_lines),
                "is_contents": is_contents,
            }
        )

    _treasury_table_cache[filename] = tables
    _treasury_heading_cache[filename] = "\n".join(headings)
    return tables


def _load_heading_summary(filename: str) -> str:
    """Load cached table headings for a Treasury bulletin."""
    cached = _treasury_heading_cache.get(filename)
    if cached is not None:
        return cached
    _extract_table_blocks(filename)
    return _treasury_heading_cache.get(filename, "")


def _diversify_source_files(files: list[str], max_files: int) -> list[str]:
    """Pick roughly evenly spaced files to avoid month-prefix bias."""
    if len(files) <= max_files:
        return files

    if max_files <= 1:
        return [files[len(files) // 2]]

    chosen_indices = {
        round(i * (len(files) - 1) / (max_files - 1))
        for i in range(max_files)
    }
    return [files[idx] for idx in sorted(chosen_indices)]


def _question_keywords(question: str) -> list[str]:
    """Extract lightweight lexical hints for snippet retrieval."""
    keywords = []
    lower_question = question.lower()
    negative_terms: set[str] = set()
    if "revolving fund" in lower_question:
        negative_terms.update({"revolving", "funds", "fund"})
    if any(phrase in lower_question for phrase in ["trust fund accounts", "trust accounts", "transfers to trust"]):
        negative_terms.update({"trust", "accounts", "account", "transfers"})
    if "territories" in lower_question:
        negative_terms.add("territories")
    if "regional aggregate" in lower_question or "regional aggregates" in lower_question:
        negative_terms.update({"regional", "aggregate", "aggregates"})
    for phrase in _extract_focus_phrases(question):
        if phrase not in keywords:
            keywords.append(phrase)
    for ent in _extract_named_entities(question):
        if ent not in keywords:
            keywords.append(ent)
    for phrase in _extract_quoted_phrases(question):
        if phrase not in keywords:
            keywords.append(phrase)
    for code in _extract_table_codes(question):
        if code not in keywords:
            keywords.append(code)

    words = re.findall(r"[A-Za-z][A-Za-z0-9$%.-]{2,}", question.lower())
    for word in words:
        if word not in _STOPWORDS and word not in keywords:
            keywords.append(word)

    for pattern, expansions in _DOMAIN_PHRASE_EXPANSIONS.items():
        if re.search(pattern, lower_question):
            for phrase in expansions:
                if phrase not in keywords:
                    keywords.append(phrase)
    filtered = [kw for kw in keywords if kw not in negative_terms]
    return filtered[:28]


def _extract_years(question: str) -> list[str]:
    """Extract all explicit years from the question."""
    years = set(re.findall(r"\b(19[2-9]\d|20[0-2]\d)\b", question))
    range_patterns = [
        r"\bfrom\s+(19[2-9]\d|20[0-2]\d)\s+to\s+(19[2-9]\d|20[0-2]\d)\b",
        r"\bbetween\s+(19[2-9]\d|20[0-2]\d)\s+and\s+(19[2-9]\d|20[0-2]\d)\b",
        r"\b(19[2-9]\d|20[0-2]\d)\s*[-–—]\s*(19[2-9]\d|20[0-2]\d)\b",
    ]
    for pattern in range_patterns:
        for start_text, end_text in re.findall(pattern, question, re.IGNORECASE):
            start_year = int(start_text)
            end_year = int(end_text)
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            if 0 < end_year - start_year <= 20:
                years.update(str(year) for year in range(start_year, end_year + 1))
    years = list(years)
    years.sort()
    return years


def _extract_dimension_terms(question: str, profile: dict[str, object] | None = None) -> list[str]:
    """Extract row/column-style cues that should appear near the target table cells."""
    q_lower = question.lower()
    profile = profile or {}
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip().lower()
        if term and term not in terms:
            terms.append(term)

    direct_terms = [
        "competitive",
        "noncompetitive",
        "accepted tenders",
        "amount of bids accepted",
        "bids submitted",
        "rollover tenders",
        "noncash rollover tenders",
        "all other investors",
        "held by u.s. govt. accounts",
        "held by federal reserve banks",
        "amount outstanding",
        "description of securities",
        "outlays",
        "receipts",
        "actual",
        "calendar year",
        "calendar yr",
        "fiscal year",
        "grand total",
        "total outstanding",
        "gross federal debt",
        "summary of public debt",
        "debt outstanding",
    ]
    for term in direct_terms:
        if term in q_lower:
            add(term)

    if profile.get("wants_calendar_year") or profile.get("wants_monthly_series"):
        add("calendar year")
        add("calendar yr")
    if profile.get("wants_bill_rate_data"):
        for term in [
            "average rate",
            "average issuing rate",
            "investment rate",
            "discount rate",
            "equivalent coupon issue yield",
            "91-day",
            "182-day",
        ]:
            add(term)
    if profile.get("wants_auction_bid_data"):
        for term in [
            "accepted tenders",
            "amount of bids accepted",
            "competitive",
            "noncompetitive",
            "rollover tenders",
        ]:
            add(term)
    if profile.get("wants_maturity_schedule"):
        for term in [
            "description of securities",
            "amount outstanding",
            "all other investors",
            "held by u.s. govt. accounts",
        ]:
            add(term)
    return terms


def _build_question_profile(question: str) -> dict[str, object]:
    """Classify the retrieval shape of an OfficeQA question.

    These questions are mostly tabular. We need to know whether the user is
    looking for a scalar cell, a list/series, a date chosen by a comparison, or
    a derived calculation that requires nearby monthly/annual rows.
    """
    q_lower = question.lower()
    month_names = [name for name in _MONTH_MAP if name in q_lower]
    year_count = len(_extract_years(question))
    mentions_veterans_admin = "veterans administration" in q_lower or "veterans' administration" in q_lower
    include_public_works = "public works" in q_lower
    exclude_revolving = "revolving fund" in q_lower
    exclude_trust_accounts = any(
        phrase in q_lower
        for phrase in [
            "trust fund accounts",
            "trust accounts",
            "transfers to trust",
        ]
    )
    prefers_country_claims = (
        ("claims" in q_lower and "country" in q_lower)
        and ("highest amount" in q_lower or "highest" in q_lower or "owed by a country" in q_lower)
    )
    exclude_territories = "territories" in q_lower
    exclude_regional_aggregates = "regional aggregate" in q_lower or "regional aggregates" in q_lower
    note_term_match = re.search(r"\b(\d+)-year\s+u\.?s\.?\s+treasury\s+notes?\b", q_lower)
    note_term_years = int(note_term_match.group(1)) if note_term_match else 0
    maturity_match = re.search(
        r"\bmaturing(?:\s+at\s+the\s+end\s+of|\s+end\s+of)?\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+"
        r"(19[3-9]\d|20[0-2]\d)\b",
        q_lower,
    )
    maturity_month = maturity_match.group(1) if maturity_match else ""
    maturity_year = maturity_match.group(2) if maturity_match else ""
    wants_tax_regression = (
        "individual income tax receipts" in q_lower
        and any(phrase in q_lower for phrase in ["ordinary least squares", "linear regression", "slope and intercept"])
    )
    expects_series_math = any(
        phrase in q_lower
        for phrase in [
            "geometric mean",
            "arithmetic mean",
            "weighted average",
            "weighted mean",
            "average",
            "mean",
            "median",
            "variance",
            "standard deviation",
            "std. dev",
            "std dev",
            "correlation",
            "volatility",
        ]
    )
    wants_january_debt_series = (
        "january" in q_lower
        and "debt" in q_lower
        and any(phrase in q_lower for phrase in ["list the", "comma separated list", "inclusive"])
    )
    wants_calendar_year = "calendar year" in q_lower or "calendar yr" in q_lower
    wants_monthly_series = any(
        phrase in q_lower
        for phrase in [
            "all individual calendar months",
            "monthly values",
            "each year from",
            "comma separated list",
            "each month",
            "for each month",
            "month from",
        ]
    ) or (expects_series_math and bool(month_names) and year_count >= 2)
    wants_auction_bid_data = any(
        phrase in q_lower
        for phrase in [
            "accepted tenders",
            "amount of bids accepted",
            "bids submitted",
            "rollover tenders",
            "noncash rollover tenders",
            "competitive",
            "noncompetitive",
            "investors submitted",
        ]
    )
    wants_bill_rate_data = (
        any(phrase in q_lower for phrase in ["13-week", "26-week", "91-day", "182-day"])
        and any(phrase in q_lower for phrase in ["rate", "rates", "yield", "yields", "discount"])
    ) or any(
        phrase in q_lower
        for phrase in [
            "treasury-bill rates",
            "treasury bill rates",
            "average issuing rate",
            "investment rate",
            "discount rate",
            "equivalent coupon issue yield",
        ]
    )
    wants_maturity_schedule = (
        "maturing" in q_lower
        and not wants_bill_rate_data
        and not wants_auction_bid_data
    ) or any(
        phrase in q_lower
        for phrase in [
            "held by all other investors",
            "held by federal reserve banks",
            "amount outstanding",
            "description of securities",
            "maturity schedule",
        ]
    )
    return {
        "expects_list": any(
            phrase in q_lower
            for phrase in [
                "list the",
                "provide the list",
                "inside square brackets",
                "enclosed brackets",
                "containing 2 numbers",
                "containing 12 numbers",
            ]
        ),
        "expects_regression": any(
            phrase in q_lower
            for phrase in [
                "ordinary least squares",
                "linear regression",
                "fit a regression",
                "slope and intercept",
            ]
        ),
        "expects_series_math": expects_series_math,
        "expects_date": any(
            phrase in q_lower
            for phrase in [
                "which date",
                "what date",
                "issue date",
                "report your answer in the u.s. long date format",
                "which march",
                "which month",
                "what month",
                "which year",
                "what year",
            ]
        ) or expects_series_math,
        "expects_percent": any(token in q_lower for token in ["percent", "percentage", "%"]),
        "expects_difference": any(
            phrase in q_lower for phrase in ["difference", "gap", "spread", "minus", "absolute change"]
        ),
        "expects_sum": any(
            phrase in q_lower
            for phrase in [
                "sum of",
                "sum the",
                "total sum",
                "using all monthly values",
                "all individual calendar months",
                "add up",
                "combined total from all",
            ]
        ),
        "expects_scalar_lookup": not any(
            phrase in q_lower
            for phrase in [
                "sum of",
                "difference",
                "gap",
                "regression",
                "geometric mean",
                "average",
                "mean",
                "weighted average",
                "median",
                "correlation",
                "standard deviation",
                "variance",
                "list the",
                "inside square brackets",
                "containing",
                "which date",
            ]
        ),
        "months": month_names,
        "year_count": year_count,
        "wants_full_number": "full number" in q_lower,
        "mentions_veterans_admin": mentions_veterans_admin,
        "include_public_works": include_public_works,
        "exclude_revolving": exclude_revolving,
        "exclude_trust_accounts": exclude_trust_accounts,
        "prefers_country_claims": prefers_country_claims,
        "exclude_territories": exclude_territories,
        "exclude_regional_aggregates": exclude_regional_aggregates,
        "note_term_years": note_term_years,
        "maturity_month": maturity_month,
        "maturity_year": maturity_year,
        "wants_tax_regression": wants_tax_regression,
        "wants_january_debt_series": wants_january_debt_series,
        "wants_calendar_year": wants_calendar_year,
        "wants_monthly_series": wants_monthly_series,
        "wants_auction_bid_data": wants_auction_bid_data,
        "wants_bill_rate_data": wants_bill_rate_data,
        "wants_maturity_schedule": wants_maturity_schedule,
    }


def build_officeqa_strategy(question: str) -> str:
    """Return question-shape-aware solving guidance for OfficeQA prompts."""
    profile = _build_question_profile(question)
    hints = [
        "Treat the reference data as Treasury tables, not free-form prose.",
        "Identify the exact table title, target row, target column, and unit before choosing an answer.",
    ]

    if profile.get("expects_regression"):
        hints.extend(
            [
                "This is a series/regression task.",
                "Extract the full ordered series first, then compute the final slope/intercept.",
                "Do not answer with a single year, label, or intermediate value.",
            ]
        )
    elif profile.get("expects_series_math"):
        hints.extend(
            [
                "This is a series/statistics task.",
                "Extract the full ordered monthly or yearly series first, then compute the requested mean, average, median, variance, standard deviation, correlation, or other statistic.",
                "Do not answer with a single year, label, or partial subtotal.",
            ]
        )
    elif profile.get("expects_list"):
        hints.extend(
            [
                "This question expects a multi-value list.",
                "Collect every requested value in order and preserve the full list in the final answer.",
                "Do not collapse the list into a single scalar.",
            ]
        )
    elif profile.get("expects_sum"):
        hints.extend(
            [
                "This is an aggregation task.",
                "Locate the full monthly/annual series and combine all requested values, not just one row.",
            ]
        )
    elif profile.get("expects_difference"):
        hints.extend(
            [
                "This is a comparison task.",
                "Find both operands from the same table context before computing the difference or spread.",
            ]
        )
    elif profile.get("expects_percent"):
        hints.extend(
            [
                "This is a percentage task.",
                "Use the exact numerator and denominator from the reference data and output one final percentage only.",
            ]
        )
    elif profile.get("expects_date"):
        hints.extend(
            [
                "This is a date-selection task.",
                "Choose the exact reporting or issue date from the table and return only the final date.",
            ]
        )
    else:
        hints.extend(
            [
                "This is an exact cell-lookup task.",
                "Prefer the row/column intersection that best matches the requested entity, year, month, and unit.",
            ]
        )

    if profile.get("wants_bill_rate_data"):
        hints.append("Prefer Treasury-bill rate or yield tables over accepted-tenders or amount-submitted tables.")
    if profile.get("wants_auction_bid_data"):
        hints.append("Prefer auction-result tables with accepted tenders, competitive, noncompetitive, or rollover columns.")
    if profile.get("wants_maturity_schedule"):
        hints.append("Prefer maturity-schedule tables with amount outstanding, description of securities, and investor-holding columns.")
    if profile.get("wants_calendar_year"):
        hints.append("Prefer calendar-year tables or rows over fiscal-year summary tables unless the question explicitly asks for fiscal year.")
    if profile.get("wants_monthly_series"):
        hints.append("Prefer month-by-month tables over annual summary rows when the question asks for monthly values or a statistic computed from months.")
    if re.search(r"\bnational defense\b", question.lower()) and (
        profile.get("wants_calendar_year") or profile.get("wants_monthly_series")
    ):
        hints.append(
            "For national-defense calendar questions, prefer the month-by-month national-defense table over generic fiscal summary tables. If the bulletin only gives monthly values, compute the requested total or statistic from those monthly values."
        )

    if profile.get("months"):
        hints.append("Month cues are important here; prefer rows and columns that explicitly mention the requested month names.")
    if profile.get("year_count", 0) >= 2:
        hints.append("Multiple years are involved; ensure you use the exact requested year range, not a nearby year.")
    if profile.get("wants_full_number"):
        hints.append("Return the fully expanded number, not a shortened million/billion shorthand.")

    return "\n".join(f"- {hint}" for hint in hints)


def _file_year_month(filename: str) -> tuple[int, int]:
    """Extract publication year/month from a Treasury filename."""
    match = re.match(r"treasury_bulletin_(\d{4})_(\d{2})\.txt", filename)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _shift_year_month(year: int, month: int, delta_months: int) -> tuple[int, int]:
    """Shift a year/month pair by delta months."""
    month_index = (year * 12 + (month - 1)) + delta_months
    shifted_year, shifted_month_index = divmod(month_index, 12)
    return shifted_year, shifted_month_index + 1


def _table_signal(line: str) -> int:
    """Score how table-like a line looks."""
    score = 0
    if re.search(r"\d", line):
        score += 1
    if len(re.findall(r"-?\d[\d,]*\.?\d*", line)) >= 2:
        score += 2
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", line, re.IGNORECASE):
        score += 1
    if "%" in line or "$" in line or "¥" in line or "£" in line:
        score += 1
    if re.search(r"\s{2,}", line) or "\t" in line:
        score += 1
    return score


def _heading_signal(line: str) -> int:
    """Score how much a line resembles a section/table heading."""
    lower = line.lower().strip()
    if not lower:
        return 0
    score = 0
    if re.match(
        r"^(table|chart|statement|summary|account|securities|receipts|outlays|cash|income|outgo|debt|balance|assets|liabilities|net|gross|holdings|position|exchange stabilization|federal|budget)\b",
        lower,
    ):
        score += 2
    if re.search(
        r"\b(total|gross|net|fiscal year|calendar year|outstanding|millions?|billions?|trillions?|public debt|balances?|receipts|outlays|cash|income|outgo|expenditures?|major classifications|holdings|positions?)\b",
        lower,
    ):
        score += 1
    if lower.endswith(":"):
        score += 1
    return score


def _split_into_sections(content: str) -> list[str]:
    """Split a bulletin into section/table-aware chunks."""
    lines = [line.rstrip() for line in content.splitlines()]
    sections: list[str] = []
    current: list[str] = []
    line_count = 0

    for line in lines:
        stripped = line.strip()
        starts_new_section = False
        if stripped:
            is_pipe_table_header = (
                "|" in stripped
                and stripped.count("|") >= 3
                and current
                and "|" not in current[-1]
            )
            if is_pipe_table_header:
                starts_new_section = True
            elif _heading_signal(stripped) >= 2 and current:
                starts_new_section = True
            elif _table_signal(stripped) >= 3 and current and _table_signal(current[-1]) == 0:
                starts_new_section = True
            elif line_count >= 120:
                starts_new_section = True
            elif not stripped and line_count >= 25:
                starts_new_section = True

        if starts_new_section:
            block = "\n".join(current).strip()
            if block:
                sections.append(block)
            current = [line] if stripped else []
            line_count = 1 if stripped else 0
            continue

        if stripped or current:
            current.append(line)
            line_count += 1

    block = "\n".join(current).strip()
    if block:
        sections.append(block)
    return sections


def _score_section(section: str, keywords: list[str], years: list[str], profile: dict[str, object] | None = None) -> int:
    """Score a section by title/header relevance, table density, and year evidence."""
    lines = [line for line in section.splitlines() if line.strip()]
    if not lines:
        return 0

    title_lines = lines[:6]
    title_text = _normalize_text(" ".join(title_lines))
    body_text = _normalize_text(section)
    score = 0

    for keyword in keywords:
        keyword_norm = _normalize_text(keyword)
        title_hits = title_text.count(keyword_norm)
        if title_hits:
            score += min(title_hits, 2) * (36 if " " in keyword_norm else 10)
        body_hits = body_text.count(keyword_norm)
        if body_hits:
            score += min(body_hits, 4) * (8 if " " in keyword_norm else 2)

    table_lines = [line for line in lines if _table_signal(line) >= 2]
    score += min(len(table_lines), 12)
    score += sum(_heading_signal(line) for line in title_lines[:4])

    profile = profile or {}
    expects_list = bool(profile.get("expects_list"))
    expects_regression = bool(profile.get("expects_regression"))
    expects_series_math = bool(profile.get("expects_series_math"))
    expects_date = bool(profile.get("expects_date"))
    expects_percent = bool(profile.get("expects_percent"))
    expects_sum = bool(profile.get("expects_sum"))
    expects_difference = bool(profile.get("expects_difference"))
    months = profile.get("months", [])

    if expects_regression:
        year_dense_lines = sum(1 for line in lines if len(re.findall(r"\b(?:19|20)\d{2}\b", line)) >= 2)
        score += min(year_dense_lines, 8) * 4
    if expects_series_math:
        year_dense_lines = sum(1 for line in lines if len(re.findall(r"\b(?:19|20)\d{2}\b", line)) >= 2)
        month_dense_lines = sum(
            1
            for line in lines
            if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\b", line, re.IGNORECASE)
        )
        score += min(year_dense_lines, 6) * 3
        score += min(month_dense_lines, 8) * 2
    if expects_list:
        repetitive_numeric_lines = sum(1 for line in lines if len(re.findall(r"-?\d[\d,]*\.?\d*", line)) >= 4)
        score += min(repetitive_numeric_lines, 8) * 3
    if expects_date:
        date_lines = sum(
            1
            for line in lines
            if re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\bdate\b|\bissue\b",
                line,
                re.IGNORECASE,
            )
        )
        score += min(date_lines, 8) * 3
    if expects_percent:
        percent_lines = sum(1 for line in lines if "%" in line or "percent" in line.lower())
        score += min(percent_lines, 8) * 2
    if expects_sum or expects_difference:
        aggregate_lines = sum(
            1
            for line in lines
            if re.search(r"\b(total|sum|gross|net|difference|balance|outstanding|cost|receipts|outlays)\b", line, re.IGNORECASE)
        )
        score += min(aggregate_lines, 10) * 2
    if months:
        month_lines = sum(
            1 for line in lines if any(month in line.lower() for month in months)
        )
        score += min(month_lines, 6) * 2

    for year in years:
        year_re = re.compile(rf"(?<!\d){year}(?!\d)")
        year_hits = sum(1 for line in lines if year_re.search(line))
        if year_hits:
            score += min(year_hits, 4) * 6
            if any(_table_signal(line) >= 2 and year_re.search(line) for line in lines):
                score += 8

    if len(table_lines) >= 3 and any(_heading_signal(line) for line in title_lines):
        score += 10

    return score


def _score_table_block(
    table: dict[str, object],
    question: str,
    keywords: list[str],
    years: list[str],
    profile: dict[str, object] | None = None,
) -> int:
    """Score a parsed table block against an OfficeQA question."""
    if table.get("is_contents"):
        return -50

    profile = profile or {}
    families = _detect_table_families(question, profile)
    score = 0
    title = str(table.get("norm_title", ""))
    text = str(table.get("norm_text", ""))
    unit_hint = _normalize_text(str(table.get("unit_hint", "")))
    table_code = str(table.get("table_code", ""))
    family_text = "\n".join([title, text, unit_hint])
    dimension_terms = _extract_dimension_terms(question, profile)

    for keyword in keywords:
        keyword_norm = _normalize_text(keyword)
        title_hits = title.count(keyword_norm)
        text_hits = text.count(keyword_norm)
        if title_hits:
            score += min(title_hits, 2) * (42 if " " in keyword_norm else 12)
        if text_hits:
            score += min(text_hits, 6) * (10 if " " in keyword_norm else 3)

    if table_code:
        score += 22
        for question_code in _extract_table_codes(question):
            if question_code == table_code:
                score += 55

    pipe_lines = int(table.get("pipe_lines", 0) or 0)
    score += min(pipe_lines, 14)

    for year in years:
        if re.search(rf"(?<!\d){year}(?!\d)", title):
            score += 15
        year_hits = len(re.findall(rf"(?<!\d){year}(?!\d)", text))
        if year_hits:
            score += min(year_hits, 6) * 4

    lower_question = question.lower()
    months = profile.get("months", [])
    if months:
        month_hits = sum(1 for month in months if month in title or month in text)
        score += month_hits * 8
    if dimension_terms:
        dimension_hits = sum(1 for term in dimension_terms if term in family_text)
        if dimension_hits:
            score += min(dimension_hits, 5) * 9

    if profile.get("expects_sum") and re.search(r"\b(total|sum|gross|net|outstanding|balance|receipts|outlays|expenditures?)\b", title + "\n" + text):
        score += 20
    if profile.get("expects_difference") and re.search(r"\b(net|gross|difference|spread|less|minus)\b", title + "\n" + text):
        score += 16
    if profile.get("expects_percent") and ("percent" in title or "%" in text or "ratio" in title):
        score += 14
    if profile.get("expects_regression"):
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        if year_hits >= 6:
            score += 22
    if profile.get("expects_series_math"):
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        month_hits = len(re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\b", text))
        if year_hits >= 4:
            score += 18
        if month_hits >= 6:
            score += 18
    if profile.get("expects_list"):
        numeric_hits = len(re.findall(r"-?\d[\d,]*\.?\d*", text))
        if numeric_hits >= 8:
            score += 16
    if profile.get("expects_date") and re.search(r"\bdate\b|\bissue\b|\bmaturity\b", title + "\n" + text):
        score += 18
    if profile.get("wants_calendar_year") and re.search(r"\bcalendar yr\b|\bcalendar year\b", family_text):
        score += 26
    if profile.get("wants_calendar_year") and re.search(r"\bcomplete fiscal years?\b|\bfiscal year\b", family_text) and not re.search(r"\bcalendar yr\b|\bcalendar year\b", family_text):
        score -= 18
    if profile.get("wants_monthly_series") and len(re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\b", family_text)) >= 6:
        score += 20

    if "million" in lower_question and "million" in unit_hint:
        score += 12
    if "billion" in lower_question and "billion" in unit_hint:
        score += 12
    if "trillion" in lower_question and "trillion" in unit_hint:
        score += 12
    if "thousand" in lower_question and "thousand" in unit_hint:
        score += 12

    matched_family = False
    for family in families:
        family_rules = _TABLE_FAMILY_MATCHERS.get(family, {})
        if table_code and any(table_code.startswith(prefix) for prefix in family_rules.get("codes", [])):
            score += 40
            matched_family = True
        keyword_hits = sum(1 for keyword in family_rules.get("keywords", []) if keyword in family_text)
        if keyword_hits:
            score += min(keyword_hits, 4) * 12
            matched_family = True

    if "fd" in families and re.search(r"\bownership of federal securities\b", family_text):
        score -= 32
        if profile.get("expects_list"):
            score -= 18
    if "fd" in families and re.search(r"\b(summary of public debt|summary of federal debt|debt outstanding)\b", family_text):
        score += 26
    if "auction" in families and "notes" in lower_question and re.search(r"\bofferings of bills\b", family_text):
        score -= 34
    if "auction" in families and "notes" in lower_question and re.search(r"\bnotes?\b", family_text):
        score += 10
    if "auction" in families and "notes" in lower_question and re.search(
        r"\b(public offerings of marketable securities other than regular weekly treasury bills|marketable securities other than regular weekly treasury bills)\b",
        family_text,
    ):
        score += 42
    if "cm" in families and re.search(r"\bnet foreign transactions\b", family_text):
        score -= 24
    if "cm" in families and re.search(r"\b(total claims by type and country|total liabilities by type and country|claims on foreigners|liabilities to foreigners)\b", family_text):
        score += 28
    if "tax_receipts" in families and re.search(r"\bcorporation income tax returns\b", family_text):
        score -= 10
    if "tax_receipts" in families and re.search(r"\b(individual income tax receipts|internal revenue collections|net of refunds)\b", family_text):
        score += 18
    if profile.get("wants_tax_regression"):
        if re.search(r"\b(individual income tax receipts|summary by principal sources|analysis of receipts from internal revenue)\b", family_text):
            score += 44
        if re.search(r"\bcorporation income tax returns\b", family_text):
            score -= 68
    if "agency_expense" in families and re.search(r"\banalysis of national defense expenditures\b", family_text):
        score += 24
    if "agency_expense" in families and re.search(r"\b(veterans administration|public works|departments and agencies)\b", family_text):
        score += 20
    if "agency_expense" in families and re.search(
        r"\b(assets and liabilities of governmental corporations and credit agencies|receipts and expenditures for trust accounts)\b",
        family_text,
    ):
        score -= 18
    if "agency_expense" in families and "exclud" in lower_question and re.search(
        r"\b(revolving funds|trust accounts|transfers to trust accounts)\b",
        family_text,
    ):
        score -= 40
    if profile.get("mentions_veterans_admin") and profile.get("include_public_works"):
        if re.search(r"\banalysis of general expenditures\b", family_text):
            score += 34
        if re.search(r"\bveterans[' ]administration\b", family_text):
            score += 28
        if re.search(r"\bpublic works\b", family_text):
            score += 24
        if re.search(r"\btable\s+3\.- analysis of general expenditures\b", family_text):
            score += 24
        if re.search(
            r"\bincludes public works undertaken by the veterans[' ]administration\b",
            family_text,
        ):
            score += 70
        if re.search(
            r"\bdoes not include expenditures for \(1\) revolving funds or \(2\) transfers to trust accounts\b",
            family_text,
        ):
            score += 80
    if profile.get("exclude_revolving") and re.search(
        r"\btable\s+5\b|\banalysis of expenditures for \(1\) revolving funds\b|\brevolving funds\b",
        title,
    ):
        score -= 95
    if profile.get("exclude_trust_accounts") and re.search(
        r"\btransfers to trust accounts\b|\btrust accounts\b",
        title,
    ):
        score -= 75
    if "national_defense" in families and re.search(r"\banalysis of national defense expenditures\b", family_text):
        score += 36
    if "national_defense" in families and re.search(r"\b(summary of fiscal statistics|analysis of general expenditures|budget receipts and expenditures)\b", family_text):
        score -= 18
    if "veterans_public_works" in families and re.search(r"\b(veterans administration|public works|analysis of general expenditures|departments and agencies)\b", family_text):
        score += 34
    if "veterans_public_works" in families and re.search(
        r"\b(revolving funds|trust accounts|transfers to trust accounts|assets and liabilities of governmental corporations and credit agencies)\b",
        family_text,
    ):
        score -= 52
    if profile.get("wants_january_debt_series"):
        if re.search(r"\bsummary of federal debt\b|\bsummary of public debt\b|\bgross federal debt\b", family_text):
            score += 40
        if re.search(r"\bownership of federal securities\b|\bmaturity distribution\b", family_text):
            score -= 48
    if "individual_income_tax" in families and re.search(r"\b(individual income tax receipts|internal revenue collections|summary by principal sources|net of refunds)\b", family_text):
        score += 36
    if "individual_income_tax" in families and re.search(r"\bcorporation income tax returns\b", family_text):
        score -= 42
    if profile.get("prefers_country_claims"):
        if re.search(r"\btotal claims by country\b", family_text):
            score += 68
        if re.search(r"\bclaims on foreigners by type and country\b", family_text):
            score -= 18
        if re.search(r"\btotal claims by type and country\b", family_text):
            score -= 24
        if re.search(r"\bliabilities to foreigners\b|\btotal liabilities\b", family_text):
            score -= 62
        if re.search(r"\bcountry\b", family_text):
            score += 10
    if profile.get("exclude_territories") or profile.get("exclude_regional_aggregates"):
        if re.search(r"\btype and country\b", family_text):
            score -= 10
    if "fcp" in families and re.search(r"\b(japanese yen positions|british pound positions|foreign currency positions)\b", family_text):
        score += 20
    if "fcp" in families and re.search(r"\b(statement of net cost|assets, liabilities, and net position of the fund)\b", family_text):
        score -= 16
    if "bill_rates" in families:
        if re.search(r"\b(average rate|average issuing rate|investment rate|discount rate|equivalent coupon issue yield|91-day|182-day)\b", family_text):
            score += 42
        if re.search(r"\b(accepted tenders|amount of bids accepted|competitive|noncompetitive|rollover tenders)\b", family_text):
            score -= 38
    if "auction_results" in families:
        if re.search(r"\b(accepted tenders|amount of bids accepted|competitive|noncompetitive|rollover tenders|bids submitted)\b", family_text):
            score += 44
        if re.search(r"\b(maturity schedule|description of securities|amount outstanding|all other investors)\b", family_text):
            score -= 42
    if "maturity_schedule" in families:
        if re.search(r"\b(maturity schedule|description of securities|amount outstanding|all other investors|held by federal reserve banks)\b", family_text):
            score += 42
        if re.search(r"\b(accepted tenders|amount of bids accepted|average issuing rate|investment rate|discount rate)\b", family_text):
            score -= 42
    if "calendar_defense" in families:
        if re.search(r"\b(analysis of national defense expenditures|expenditures for national defense and related activities)\b", family_text):
            score += 54
        if re.search(
            r"\b(cash income and outgo of the treasury, by major classifications|budget expenditures classified as general, by major functions)\b",
            family_text,
        ):
            score += 40
        if re.search(r"\b(summary table on budget receipts and expenditures|budget receipts and expenditures|summary of fiscal statistics)\b", family_text):
            score -= 30
    if ("calendar_defense" in families or "national_defense" in families) and profile.get("wants_monthly_series"):
        if re.search(r"\bsummary of budget results by months and years\b", family_text):
            score -= 28
        if re.search(r"\bbusiness-type activities\b", family_text):
            score -= 70
    if "general_expenditures" in families:
        if re.search(r"\b(analysis of general expenditures|departments and agencies)\b", family_text):
            score += 26
        if re.search(r"\bsummary table on budget receipts and expenditures\b", family_text):
            score -= 14
    if (title.startswith("[") or title.lower().startswith("in ")) and not table_code:
        score -= 12

    # Penalise unnamed table blocks — they likely failed title extraction and
    # will confuse downstream ranking / prompt assembly.
    if title.startswith("Table block near line"):
        score -= 25

    # --- Stronger auction sub-family cross-pollution penalties ---
    # When the question asks about bill *rates*, penalise "offerings of bills"
    # and "accepted tenders" tables that talk about bid amounts, not rates.
    if "bill_rates" in families and re.search(
        r"\b(offerings of bills|public debt operations.*offerings)\b", family_text
    ):
        score -= 45
    # When the question asks about notes, heavily penalise bills tables
    if "auction" in families and "notes" in lower_question and re.search(
        r"\bofferings of bills\b", family_text
    ) and not re.search(r"\bnotes?\b", family_text):
        score -= 50
    # When the question asks about bills auction *results* (bids/tenders),
    # penalise rate tables
    if "auction_results" in families and re.search(
        r"\b(average rate|average issuing rate|investment rate|discount rate|equivalent coupon issue yield)\b",
        family_text,
    ):
        score -= 40

    if families and not matched_family and table_code:
        known_prefixes = {
            prefix
            for rules in _TABLE_FAMILY_MATCHERS.values()
            for prefix in rules.get("codes", [])
        }
        if any(table_code.startswith(prefix) for prefix in known_prefixes):
            score -= 10

    return score


def _score_source_file_cheap(
    filename: str,
    question: str,
    keywords: list[str],
    years: list[str],
    profile: dict[str, object],
    families: list[str],
    dimension_terms: list[str],
) -> int:
    """Fast pre-filter score using only heading summary + filename time proximity.

    Does NOT call _extract_table_blocks or _score_table_block, making it ~50x
    cheaper than _score_source_file.  Used to narrow 697 files to ~30 candidates
    before running the expensive scorer.
    """
    heading_summary = _normalize_text(_load_heading_summary(filename))
    score = 0

    # Keyword hits in headings (cheap — headings are cached short strings)
    for kw in keywords:
        kw_norm = _normalize_text(kw)
        if not kw_norm:
            continue
        if kw_norm in heading_summary:
            hits = heading_summary.count(kw_norm)
            score += min(hits, 3) * (8 if " " in kw_norm else 3)

    # Dimension term hits in headings
    if dimension_terms and heading_summary:
        dim_hits = sum(1 for term in dimension_terms if term in heading_summary)
        if dim_hits:
            score += min(dim_hits, 5) * 8

    # Time proximity (same logic as full scorer, but standalone)
    if years:
        target_year = int(years[0])
        file_year, file_month = _file_year_month(filename)
        if file_year:
            is_fiscal = "fiscal" in question.lower() or "fy" in question.lower()
            if is_fiscal:
                earliest_year, earliest_month = target_year - 1, 10
            else:
                earliest_year, earliest_month = target_year, 1
            months_after = (file_year - earliest_year) * 12 + (file_month - earliest_month)
            if months_after >= 0:
                if months_after <= 12:
                    score += 24 - months_after
                elif months_after <= 48:
                    score += max(4, 14 - (months_after - 12) // 3)
            elif months_after >= -12:
                score += 3

    # Family code match in headings
    for family in families:
        family_rules = _TABLE_FAMILY_MATCHERS.get(family, {})
        kw_hits = sum(1 for kw in family_rules.get("keywords", []) if kw in heading_summary)
        if kw_hits:
            score += min(kw_hits, 4) * 6

    return score


def _score_source_file(filename: str, question: str) -> int:
    """Score a candidate Treasury file using time proximity and section-aware lexical overlap."""
    keywords = _question_keywords(question)
    years = _extract_years(question)
    profile = _build_question_profile(question)
    families = _detect_table_families(question, profile)
    preview = _load_source_preview(filename)
    if not preview:
        return 0
    heading_summary = _normalize_text(_load_heading_summary(filename))
    dimension_terms = _extract_dimension_terms(question, profile)

    preview_sections = _split_into_sections(preview)
    if not preview_sections:
        preview_sections = [preview]

    best_section = max((_score_section(section, keywords, years, profile) for section in preview_sections), default=0)
    score = best_section
    norm_content = _load_normalized_content(filename)
    tables = _extract_table_blocks(filename)
    best_table = max((_score_table_block(table, question, keywords, years, profile) for table in tables), default=0)
    score += max(0, min(best_table, 260))

    lower_filename = filename.lower()
    for kw in keywords:
        kw_norm = _normalize_text(kw)
        if kw_norm in lower_filename:
            score += 2 if " " in kw_norm else 1
        if heading_summary and kw_norm and kw_norm in heading_summary:
            heading_hits = heading_summary.count(kw_norm)
            score += min(heading_hits, 3) * (8 if " " in kw_norm else 3)
        if not norm_content or not kw_norm:
            continue
        if " " in kw_norm:
            phrase_hits = norm_content.count(kw_norm)
            if phrase_hits:
                score += min(phrase_hits, 3) * 6
        elif len(kw_norm) >= 6 and kw_norm in norm_content:
            score += 2
    if dimension_terms and heading_summary:
        dimension_hits = sum(1 for term in dimension_terms if term in heading_summary)
        if dimension_hits:
            score += min(dimension_hits, 5) * 8

    if years:
        target_year = int(years[0])
        file_year, file_month = _file_year_month(filename)
        if file_year:
            is_fiscal = "fiscal" in question.lower() or "fy" in question.lower()
            if is_fiscal:
                earliest_year, earliest_month = target_year - 1, 10
            else:
                earliest_year, earliest_month = target_year, 1
            months_after = (file_year - earliest_year) * 12 + (file_month - earliest_month)
            if months_after >= 0:
                if months_after <= 12:
                    score += 24 - months_after
                elif months_after <= 48:
                    score += max(4, 14 - (months_after - 12) // 3)
            elif months_after >= -12:
                score += 3

            if profile.get("months"):
                expected_months = {int(_MONTH_MAP[m]) for m in profile["months"] if m in _MONTH_MAP}
                if file_month in expected_months:
                    if "fd" in families and profile.get("expects_list"):
                        score += 2
                    else:
                        score += 10
            if profile.get("wants_january_debt_series"):
                start_year = min(int(y) for y in years) if years else 0
                if file_month == 10:
                    score += 44
                    if file_year == start_year:
                        score += 72
                    elif file_year == start_year + 1:
                        score += 6
                elif file_month == 11:
                    score -= 12
                    if file_year == start_year:
                        score -= 18
                else:
                    score -= 12
            if ("national_defense" in families or "agency_expense" in families) and not is_fiscal:
                if file_year == target_year + 1:
                    score += 14
                    if 1 <= file_month <= 4:
                        score += 8
            if "calendar_defense" in families and not is_fiscal:
                if target_year <= file_year <= target_year + 2:
                    score += 22
                if file_year >= target_year and 1 <= file_month <= 4:
                    score += 10
            if "veterans_public_works" in families and profile.get("mentions_veterans_admin"):
                if (file_year, file_month) == (target_year + 10, 1):
                    score += 120
                elif (file_year, file_month) == (target_year + 10, 2):
                    score += 88
                elif file_year == target_year + 10 and 3 <= file_month <= 6:
                    score += 20
                elif file_year > target_year + 10:
                    score -= min(40, (file_year - (target_year + 10)) * 12 + file_month)
            if profile.get("prefers_country_claims") and file_month == 12:
                score += 8
                if file_year >= target_year + 2:
                    score += 10
                if file_year == target_year + 3:
                    score += 18

            if "auction" in families and profile.get("expects_date") and profile.get("months"):
                requested_month = _MONTH_MAP.get(profile["months"][0], "")
                if requested_month and file_year == target_year:
                    next_year, next_month = _shift_year_month(target_year, int(requested_month), 1)
                    if (file_year, file_month) == (next_year, next_month):
                        score += 28
                    elif file_month == int(requested_month):
                        score += 10

            if (
                "auction" in families
                and profile.get("note_term_years")
                and profile.get("maturity_month")
                and profile.get("maturity_year")
            ):
                maturity_month = _MONTH_MAP.get(str(profile["maturity_month"]), "")
                if maturity_month:
                    issue_year, issue_month = _shift_year_month(
                        int(profile["maturity_year"]),
                        int(maturity_month),
                        -12 * int(profile["note_term_years"]) + 1,
                    )
                    if (file_year, file_month) == (issue_year, issue_month):
                        score += 90
                    elif abs((file_year - issue_year) * 12 + (file_month - issue_month)) <= 1:
                        score += 32

    if profile.get("expects_regression") and len(years) >= 4:
        file_year, _ = _file_year_month(filename)
        if file_year and years and int(years[0]) <= file_year <= int(years[-1]) + 2:
            score += 8
    if profile.get("expects_regression") and ("tax_receipts" in families or "individual_income_tax" in families):
        file_year, file_month = _file_year_month(filename)
        max_year = max(int(year) for year in years) if years else 0
        if file_year in {max_year, max_year + 1}:
            score += 18
            if 6 <= file_month <= 9:
                score += 6
        if profile.get("wants_tax_regression"):
            if file_month == 7:
                score += 48
            elif file_month == 6:
                score += 12
            elif file_month == 8:
                score += 18
            elif file_month not in {6, 7, 8}:
                score -= 10

    if families:
        family_hits = 0
        for family in families:
            rules = _TABLE_FAMILY_MATCHERS.get(family, {})
            family_hits += sum(1 for keyword in rules.get("keywords", []) if keyword in heading_summary)
        if family_hits:
            score += min(family_hits, 4) * 10
        if "fd" in families:
            if "summary of public debt" in heading_summary or "summary of federal debt" in heading_summary:
                score += 24
            if "ownership of federal securities" in heading_summary:
                score -= 16
                if profile.get("expects_list"):
                    score -= 14
            if profile.get("wants_january_debt_series"):
                if "summary of federal debt" in heading_summary or "summary of public debt" in heading_summary:
                    score += 40
                if "ownership of federal securities" in heading_summary or "maturity distribution" in heading_summary:
                    score -= 54
        if "auction" in families and "notes" in question.lower():
            if "public offerings of marketable securities other than regular weekly treasury bills" in heading_summary:
                score += 36
            if "offerings of bills" in heading_summary and "notes" not in heading_summary:
                score -= 24
        if "bill_rates" in families:
            if re.search(r"\b(average rate|average issuing rate|investment rate|discount rate|equivalent coupon issue yield)\b", heading_summary):
                score += 30
            if re.search(r"\b(accepted tenders|amount of bids accepted|competitive|noncompetitive)\b", heading_summary):
                score -= 34
        if "auction_results" in families:
            if re.search(r"\b(accepted tenders|amount of bids accepted|competitive|noncompetitive|rollover tenders)\b", heading_summary):
                score += 34
            if re.search(r"\b(maturity schedule|description of securities|amount outstanding|all other investors)\b", heading_summary):
                score -= 38
        if "maturity_schedule" in families:
            if re.search(r"\b(maturity schedule|description of securities|amount outstanding|all other investors)\b", heading_summary):
                score += 34
            if re.search(r"\b(accepted tenders|amount of bids accepted|average rate|investment rate|discount rate)\b", heading_summary):
                score -= 38
        if "cm" in families:
            if "total claims by type and country" in heading_summary or "total liabilities by type and country" in heading_summary:
                score += 26
            if "net foreign transactions" in heading_summary:
                score -= 18
        if profile.get("prefers_country_claims"):
            if "total claims by country" in heading_summary:
                score += 56
            if "total claims by type and country" in heading_summary:
                score -= 20
            if "liabilities to foreigners" in heading_summary or "total liabilities by type and country" in heading_summary:
                score -= 48
        if "agency_expense" in families and "analysis of national defense expenditures" in heading_summary:
            score += 18
        if "agency_expense" in families and (
            "veterans administration" in heading_summary
            or "public works" in heading_summary
            or "departments and agencies" in heading_summary
        ):
            score += 18
        if "agency_expense" in families and "exclud" in question.lower():
            if "revolving funds" in heading_summary or "trust accounts" in heading_summary:
                score -= 26
        if profile.get("mentions_veterans_admin") and profile.get("include_public_works"):
            if (
                "analysis of general expenditures" in heading_summary
                or "veterans administration" in heading_summary
                or "public works" in heading_summary
            ):
                score += 26
            if (
                "includes public works undertaken by the veterans' administration" in norm_content
                or "includes public works undertaken by the veterans administration" in norm_content
            ):
                score += 110
            if "does not include expenditures for (1) revolving funds or (2) transfers to trust accounts" in norm_content:
                score += 130
        if profile.get("exclude_revolving") and (
            "analysis of expenditures for (1) revolving funds" in heading_summary
            or re.search(r"\btable\s+5\b", heading_summary)
        ):
            score -= 54
        if profile.get("exclude_trust_accounts") and (
            "transfers to trust accounts" in heading_summary
            or "trust accounts" in heading_summary
        ):
            score -= 44
        if "national_defense" in families:
            if "analysis of national defense expenditures" in heading_summary:
                score += 30
            if "summary of fiscal statistics" in heading_summary or "analysis of general expenditures" in heading_summary:
                score -= 16
        if "calendar_defense" in families:
            if "analysis of national defense expenditures" in heading_summary or "expenditures for national defense and related activities" in heading_summary:
                score += 40
            if "summary table on budget receipts and expenditures" in heading_summary or "budget receipts and expenditures" in heading_summary:
                score -= 24
        if "general_expenditures" in families:
            if "analysis of general expenditures" in heading_summary or "departments and agencies" in heading_summary:
                score += 22
            if "summary table on budget receipts and expenditures" in heading_summary:
                score -= 12
        if "veterans_public_works" in families:
            if (
                "veterans administration" in heading_summary
                or "public works" in heading_summary
                or "departments and agencies" in heading_summary
                or "analysis of general expenditures" in heading_summary
            ):
                score += 24
            if (
                "revolving funds" in heading_summary
                or "trust accounts" in heading_summary
                or "assets and liabilities of governmental corporations and credit agencies" in heading_summary
            ):
                score -= 36
        if "fcp" in families:
            if "foreign currency positions" in heading_summary or "japanese yen positions" in heading_summary:
                score += 16
            if "statement of net cost" in heading_summary:
                score -= 12
        if "tax_receipts" in families and heading_summary:
            if "individual income tax receipts" in heading_summary or "internal revenue collections" in heading_summary:
                score += 18
            if profile.get("wants_tax_regression") and "corporation income tax returns" in heading_summary:
                score -= 52
        if "individual_income_tax" in families and heading_summary:
            if (
                "individual income tax receipts" in heading_summary
                or "internal revenue collections" in heading_summary
                or "summary by principal sources" in heading_summary
            ):
                score += 30
            if "corporation income tax returns" in heading_summary:
                score -= 34
            if profile.get("wants_tax_regression") and "totals by months, beginning with 1933" in heading_summary:
                score += 26

    if (profile.get("expects_regression") or profile.get("expects_list")) and len(years) >= 2 and norm_content:
        distinct_year_hits = sum(
            1 for year in years if re.search(rf"(?<!\\d){year}(?!\\d)", norm_content)
        )
        if distinct_year_hits:
            score += min(distinct_year_hits, len(years)) * 8
            if distinct_year_hits >= max(4, len(years) // 2):
                score += 26

    for code in _extract_table_codes(question):
        if code in preview.lower() or (norm_content and code in norm_content):
            score += 30

    return score


def _detect_bulletin_reference(question: str, available: list[str]) -> list[str]:
    """Detect direct bulletin month/year references from the question."""
    results: list[str] = []
    patterns = [
        r"(?:treasury\s+bulletin\s+(?:from|of|in)\s+)(\w+)\s+(\d{4})",
        r"(\w+)\s+(\d{4})\s+treasury\s+bulletin",
        r"(?:the\s+)(\w+)\s+(\d{4})\s+bulletin",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            month_name = match.group(1).lower()
            year = match.group(2)
            if month_name in _MONTH_MAP:
                fname = f"treasury_bulletin_{year}_{_MONTH_MAP[month_name]}.txt"
                if fname in available and fname not in results:
                    results.append(fname)

    return results


def _select_best_source_files(
    candidates: list[str],
    question: str,
    max_files: int,
    pinned: list[str] | None = None,
) -> list[str]:
    """Rank candidate files lexically; fall back to diversified chronology."""
    pinned = [fname for fname in (pinned or []) if fname in candidates]
    if len(candidates) <= max_files:
        return pinned + [fname for fname in candidates if fname not in pinned]

    # Score ALL candidates (including pinned) so we can rank them fairly
    ranked = [
        (_score_source_file(fname, question), idx, fname)
        for idx, fname in enumerate(candidates)
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))

    if ranked and ranked[0][0] > 0:
        # Build result: pinned files get a score bonus to stay near the top,
        # but they participate in the same ranking to avoid crowding out
        # high-scoring non-pinned files.
        pin_set = set(pinned)
        chosen: list[str] = []
        seen: set[str] = set()
        # First pass: interleave top-scored pinned files
        for _, _, fname in ranked:
            if fname in pin_set and fname not in seen:
                chosen.append(fname)
                seen.add(fname)
            if len(chosen) >= max_files:
                break
        # Second pass: fill remaining slots with highest-scored files
        for _, _, fname in ranked:
            if fname in seen:
                continue
            chosen.append(fname)
            seen.add(fname)
            if len(chosen) >= max_files:
                break
        return chosen[:max_files]

    fallback = _diversify_source_files(candidates, max_files)
    chosen = list(pinned[:max_files])
    seen = set(chosen)
    for fname in fallback:
        if fname not in seen:
            chosen.append(fname)
        if len(chosen) >= max_files:
            break
    return chosen[:max_files]


def _needs_global_file_search(profile: dict[str, object], families: list[str]) -> bool:
    """Whether local time-window candidate generation is likely too weak."""
    return bool(
        profile.get("mentions_veterans_admin")
        or profile.get("prefers_country_claims")
        or profile.get("expects_regression")
        or profile.get("expects_series_math")
        or profile.get("expects_list")
        or profile.get("note_term_years")
        or (profile.get("wants_monthly_series") and profile.get("year_count", 0) >= 3)
        or ("auction" in families and profile.get("expects_date"))
    )


def _source_context_limits(question: str) -> tuple[int, int]:
    """Choose retrieval breadth based on the question's structure."""
    base_files = int(os.environ.get("MAX_SOURCE_FILES", "2"))
    base_chars = int(os.environ.get("SOURCE_MAX_CHARS", "12000"))
    profile = _build_question_profile(question)
    families = _detect_table_families(question, profile)

    max_files = base_files
    max_chars = base_chars

    if profile.get("expects_regression") or profile.get("expects_list") or profile.get("expects_series_math"):
        max_files = max(max_files, 4)
        max_chars = max(max_chars, 22000)
    if profile.get("wants_monthly_series") or profile.get("expects_sum") or profile.get("expects_percent"):
        max_files = max(max_files, 3)
        max_chars = max(max_chars, 18000)
    if profile.get("prefers_country_claims") or profile.get("year_count", 0) >= 5:
        max_files = max(max_files, 4)
    if "calendar_defense" in families:
        max_files = max(max_files, 3)
        max_chars = max(max_chars, 18000)

    return max_files, max_chars


def _seed_anchor_files(
    years_found: set[str],
    profile: dict[str, object],
    families: list[str],
    available: list[str],
) -> list[str]:
    """Add a few high-value anchor bulletins inferred from domain timing."""
    anchors: list[str] = []

    def add(year: int, month: int) -> None:
        fname = f"treasury_bulletin_{year}_{month:02d}.txt"
        if fname in available and fname not in anchors:
            anchors.append(fname)

    if years_found:
        for year in sorted(int(y) for y in years_found):
            if "national_defense" in families:
                add(year + 1, 1)
                add(year + 1, 2)
            if profile.get("mentions_veterans_admin"):
                add(year + 10, 1)
                add(year + 10, 2)
                for anchor_year in range(year + 8, year + 12):
                    add(anchor_year, 1)
                    add(anchor_year, 2)
            if profile.get("prefers_country_claims"):
                for follow_year in range(year + 1, year + 4):
                    add(follow_year, 12)
            if profile.get("wants_tax_regression") and "individual_income_tax" in families:
                add(year, 7)
                add(year + 1, 7)
                add(year, 6)
                add(year + 1, 6)
            elif profile.get("expects_regression") and "individual_income_tax" in families:
                add(year, 7)
                add(year + 1, 7)
            if profile.get("wants_january_debt_series") and "fd" in families:
                add(year, 10)
                add(year, 11)
                add(year + 1, 10)
            elif profile.get("expects_list") and "fd" in families:
                add(year, 10)
                add(year, 11)

    if (
        "auction" in families
        and profile.get("note_term_years")
        and profile.get("maturity_month")
        and profile.get("maturity_year")
    ):
        maturity_month = _MONTH_MAP.get(str(profile["maturity_month"]), "")
        if maturity_month:
            issue_year, issue_month = _shift_year_month(
                int(profile["maturity_year"]),
                int(maturity_month),
                -12 * int(profile["note_term_years"]) + 1,
            )
            add(issue_year, issue_month)
            next_year, next_month = _shift_year_month(issue_year, issue_month, 1)
            add(next_year, next_month)

    if "auction" in families and profile.get("expects_date") and profile.get("months") and years_found:
        month_num = _MONTH_MAP.get(profile["months"][0], "")
        if month_num:
            for year in sorted(int(y) for y in years_found):
                issue_year, issue_month = _shift_year_month(year, int(month_num), 1)
                add(issue_year, issue_month)

    return anchors


def _find_source_files(question: str, max_files: int | None = None) -> list[str]:
    """Find relevant Treasury source files by extracting dates from the question."""
    available = _list_source_files()
    if not available:
        return []

    lower = question.lower()
    profile = _build_question_profile(question)
    if max_files is None:
        max_files, _ = _source_context_limits(question)
    matched = []
    pinned = []
    direct_refs = _detect_bulletin_reference(question, available)
    for fname in direct_refs:
        if fname not in matched:
            matched.append(fname)
        if fname not in pinned:
            pinned.append(fname)

    for month_name, month_num in _MONTH_MAP.items():
        pattern = rf"{month_name}\s+(\d{{4}})"
        for match in re.finditer(pattern, lower):
            year = match.group(1)
            fname = f"treasury_bulletin_{year}_{month_num}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)
            if fname in available and fname not in pinned:
                pinned.append(fname)
            # Publication lag: Treasury data for month X appears in bulletins
            # published 1-3 months later.  Pin the next 3 monthly bulletins.
            y, m = int(year), int(month_num)
            for offset in range(1, 4):
                nm = m + offset
                ny = y + (nm - 1) // 12
                nm = ((nm - 1) % 12) + 1
                lag_fname = f"treasury_bulletin_{ny}_{nm:02d}.txt"
                if lag_fname in available and lag_fname not in matched:
                    matched.append(lag_fname)
                if lag_fname in available and lag_fname not in pinned:
                    pinned.append(lag_fname)

    years_found = set()
    for match in re.finditer(r"\b(19[3-9]\d|20[0-2]\d)\b", lower):
        years_found.add(match.group(1))

    families = _detect_table_families(question, profile)
    anchors = _seed_anchor_files(years_found, profile, families, available)
    for fname in anchors:
        if fname not in matched:
            matched.append(fname)
        if fname not in pinned:
            pinned.append(fname)
    followup_years = int(os.environ.get("SOURCE_FOLLOWUP_YEARS", "6"))
    if "auction" in families:
        followup_years = min(followup_years, 2)
    if "fd" in families or "fcp" in families or "cm" in families:
        followup_years = min(followup_years, 3)
    if "agency_expense" in families or "tax_receipts" in families:
        followup_years = max(followup_years, 12)
    if years_found and min(int(year) for year in years_found) < 1950:
        followup_years = max(followup_years, 10)

    for match in re.finditer(r"fiscal\s+year\s+(\d{4})", lower):
        fy = match.group(1)
        years_found.add(fy)
        prev_year = str(int(fy) - 1)
        for month in ["10", "11", "12"]:
            fname = f"treasury_bulletin_{prev_year}_{month}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            fname = f"treasury_bulletin_{fy}_{month}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)

    if years_found:
        for year in sorted(years_found):
            start_year = int(year)
            end_year = min(start_year + followup_years + 1, 2026)
            if (
                profile.get("expects_regression")
                or profile.get("expects_list")
                or profile.get("expects_series_math")
            ):
                end_year = min(start_year + max(followup_years, 10) + 1, 2026)
            for bullet_year in range(start_year, end_year):
                for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
                    fname = f"treasury_bulletin_{bullet_year}_{month}.txt"
                    if fname in available and fname not in matched:
                        matched.append(fname)

    if _needs_global_file_search(profile, families):
        # Two-stage ranking: cheap prefilter on all 697 files, then expensive
        # full scoring on only the top 30 candidates.  This cuts retrieval time
        # from ~60s to ~5s for global-search questions.
        _gkeywords = _question_keywords(question)
        _gyears = _extract_years(question)
        _gdimension = _extract_dimension_terms(question, profile)
        cheap_ranked = sorted(
            available,
            key=lambda fname: (
                -_score_source_file_cheap(
                    fname, question, _gkeywords, _gyears, profile, families, _gdimension
                ),
                fname,
            ),
        )[:30]
        # Now run the expensive scorer on only 30 candidates
        global_ranked = sorted(
            cheap_ranked,
            key=lambda fname: (-_score_source_file(fname, question), fname),
        )[: max(18, max_files * 4)]
        for fname in global_ranked:
            if fname not in matched:
                matched.append(fname)

    if matched:
        matched = _select_best_source_files(matched, question, max_files, pinned=pinned)
    else:
        matched = _select_best_source_files(available, question, max_files, pinned=pinned)
        if matched:
            logger.info(
                f"Lexical source fallback selected {len(matched)} files for undated question"
            )

    if direct_refs:
        logger.info(f"Direct bulletin references matched {len(direct_refs)} files")
    if matched:
        logger.info(f"Source retrieval: {len(matched)} files for years {sorted(years_found)}")
    else:
        logger.warning(f"No source files matched for question: {question[:100]}...")

    return matched


def _score_line_window(lines: list[str], start: int, end: int, keywords: list[str], years: list[str], profile: dict[str, object]) -> int:
    window = [line for line in lines[start:end] if line.strip()]
    if not window:
        return 0

    text = _normalize_text("\n".join(window))
    score = 0
    for keyword in keywords:
        keyword_norm = _normalize_text(keyword)
        hits = text.count(keyword_norm)
        if hits:
            score += min(hits, 4) * (10 if " " in keyword_norm else 3)

    for year in years:
        if re.search(rf"(?<!\d){year}(?!\d)", text):
            score += 10

    table_lines = sum(1 for line in window if _table_signal(line) >= 2)
    heading_hits = sum(_heading_signal(line) for line in window[:4])
    score += min(table_lines, 12) + heading_hits

    lower_text = text.lower()
    if profile.get("wants_bill_rate_data") and re.search(r"\b(average rate|average issuing rate|investment rate|discount rate|equivalent coupon issue yield)\b", lower_text):
        score += 12
    if profile.get("wants_auction_bid_data") and re.search(r"\b(accepted tenders|amount of bids accepted|competitive|noncompetitive|rollover tenders)\b", lower_text):
        score += 12
    if profile.get("wants_maturity_schedule") and re.search(r"\b(description of securities|amount outstanding|all other investors)\b", lower_text):
        score += 12
    if profile.get("wants_calendar_year") and re.search(r"\bcalendar yr\b|\bcalendar year\b", lower_text):
        score += 10
    if profile.get("expects_sum") or profile.get("expects_difference"):
        if re.search(r"\b(total|sum|gross|net|difference|balance|outstanding|cost|receipts|outlays)\b", lower_text):
            score += 12
    if profile.get("expects_regression"):
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", lower_text))
        if year_hits >= 4:
            score += 14
    if profile.get("expects_list"):
        number_hits = len(re.findall(r"-?\d[\d,]*\.?\d*", lower_text))
        if number_hits >= 6:
            score += 10
    if profile.get("expects_date"):
        if re.search(r"\bdate\b|\bissue\b", lower_text):
            score += 12
        if any(month in lower_text for month in _MONTH_MAP):
            score += 6
    if profile.get("expects_percent"):
        if "%" in lower_text or "percent" in lower_text:
            score += 8
    return score


def _extract_windowed_table_snippets(section: str, question: str, budget_chars: int, profile: dict[str, object]) -> str:
    lines = [line.rstrip() for line in section.splitlines()]
    if not lines:
        return ""

    keywords = _question_keywords(question)
    years = _extract_years(question)
    window_radius = 10
    if profile.get("expects_list") or profile.get("expects_regression") or profile.get("expects_series_math"):
        window_radius = 18
    elif profile.get("expects_sum"):
        window_radius = 14

    candidates: list[tuple[int, int, int]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if _table_signal(line) == 0 and _heading_signal(line) == 0:
            continue
        start = max(0, idx - window_radius)
        end = min(len(lines), idx + window_radius + 1)
        score = _score_line_window(lines, start, end, keywords, years, profile)
        if score > 0:
            candidates.append((score, start, end))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    merged: list[tuple[int, int]] = []
    for _, start, end in candidates:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        if len(merged) >= 4:
            break

    parts: list[str] = []
    used = 0
    for start, end in merged:
        block = "\n".join(lines[start:end]).strip()
        if not block:
            continue
        if used and used + len(block) > budget_chars:
            break
        parts.append(block[: max(0, budget_chars - used)])
        used += len(parts[-1]) + 2
        if used >= budget_chars:
            break

    return "\n\n".join(parts).strip()


def _narrow_table_rows(table_text: str, question: str, max_rows: int = 20) -> str:
    """Narrow a pipe-delimited table to the header + rows most relevant to the question.

    Keeps the first row (header) and up to *max_rows* rows that mention years,
    months, or key entities from the question.  This dramatically reduces token
    waste and prevents the LLM from reading the wrong row.
    """
    lines = [line for line in table_text.splitlines() if line.strip()]
    if len(lines) <= max_rows + 2:
        # Table is already small enough — return as-is
        return table_text

    q_lower = question.lower()
    years = set(re.findall(r"\b(19[2-9]\d|20[0-2]\d)\b", question))
    months_in_q = set()
    for m in _MONTH_MAP:
        if m in q_lower:
            months_in_q.add(m)

    # Collect row-level entities from the question for matching
    row_terms: list[str] = []
    for phrase in _extract_focus_phrases(question):
        row_terms.append(phrase.lower())
    for ent in _extract_named_entities(question):
        if len(ent) > 3:
            row_terms.append(ent.lower())

    # Always keep header rows (first 1-2 pipe rows) and separator rows
    header_lines: list[str] = []
    data_lines: list[tuple[int, str]] = []  # (score, line)

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Keep separator rows (e.g., |---|---|)
        if re.match(r"^[\s|:-]+$", stripped.replace("-", "")):
            header_lines.append(line)
            continue
        # First 2 lines are likely headers
        if i < 2:
            header_lines.append(line)
            continue

        # Score each data row
        row_score = 0
        line_lower = stripped.lower()
        for year in years:
            if year in stripped:
                row_score += 10
        for month in months_in_q:
            if month in line_lower:
                row_score += 5
        for term in row_terms:
            if term in line_lower:
                row_score += 8
        # Rows with "total" are often useful
        if re.search(r"\btotal\b", line_lower):
            row_score += 3

        data_lines.append((row_score, line))

    # Sort data rows by score, keep top max_rows
    data_lines.sort(key=lambda x: -x[0])
    # Keep rows that score > 0, up to max_rows
    relevant = [(s, l) for s, l in data_lines if s > 0][:max_rows]
    if not relevant:
        # No rows matched — keep the first max_rows data rows as fallback
        relevant = data_lines[:max_rows]

    # Re-order by original position to preserve table structure
    original_order = {line: i for i, line in enumerate(lines)}
    relevant_lines = sorted(
        [l for _, l in relevant],
        key=lambda l: original_order.get(l, 9999),
    )

    result_lines = header_lines + relevant_lines
    return "\n".join(result_lines)


def _extract_relevant_snippets(filename: str, content: str, question: str, budget_chars: int) -> str:
    """Extract table-aware snippets from the most relevant parts of a source file."""
    keywords = _question_keywords(question)
    years = _extract_years(question)
    profile = _build_question_profile(question)
    tables = _extract_table_blocks(filename)
    if tables:
        scored_tables = [
            (_score_table_block(table, question, keywords, years, profile), idx, table)
            for idx, table in enumerate(tables)
        ]
        scored_tables = [item for item in scored_tables if item[0] > 0]
        scored_tables.sort(key=lambda item: (-item[0], item[1]))
        selected_tables: list[str] = []
        used = 0
        for _, _, table in scored_tables:
            title = str(table.get("title", "")).strip()
            unit_hint = str(table.get("unit_hint", "")).strip()
            table_text = str(table.get("text", "")).strip()
            if not table_text:
                continue
            # Row/column narrowing: trim large tables to relevant rows only
            table_text = _narrow_table_rows(table_text, question, max_rows=20)
            block_parts = []
            if title:
                block_parts.append(f"TABLE TITLE: {title}")
            if unit_hint and unit_hint not in title:
                block_parts.append(f"UNIT: {unit_hint}")
            block_parts.append(table_text)
            block = "\n".join(block_parts).strip()
            if used and used + len(block) > budget_chars:
                break
            selected_tables.append(block[: max(0, budget_chars - used)])
            used += len(selected_tables[-1]) + 2
            if used >= budget_chars or len(selected_tables) >= 3:
                break
        if selected_tables:
            return "\n\n".join(selected_tables).strip()

    sections = _split_into_sections(content)
    if not sections:
        return ""

    scored_sections = [
        (_score_section(section, keywords, years, profile), idx, section)
        for idx, section in enumerate(sections)
    ]
    scored_sections = [item for item in scored_sections if item[0] > 0]
    if not scored_sections:
        scored_sections = []
        for idx, section in enumerate(sections):
            lines = [line.rstrip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            table_density = sum(_table_signal(line) for line in lines)
            heading_hits = sum(_heading_signal(line) for line in lines[:6])
            score = table_density + heading_hits
            if score > 0:
                scored_sections.append((score, idx, section))

    scored_sections.sort(key=lambda item: (-item[0], item[1]))
    if not scored_sections:
        return sections[0][:budget_chars].strip()

    selected: list[str] = []
    used = 0
    seen_indices = set()
    section_budget = max(2500, budget_chars // max(1, min(3, len(scored_sections))))
    for _, idx, section in scored_sections:
        neighbor_indices = [idx]
        if idx > 0:
            neighbor_indices.append(idx - 1)
        if idx + 1 < len(sections):
            neighbor_indices.append(idx + 1)
        for section_idx in neighbor_indices:
            if section_idx in seen_indices:
                continue
            block = _extract_windowed_table_snippets(sections[section_idx], question, section_budget, profile)
            if not block:
                block = sections[section_idx].strip()
            if not block:
                continue
            block_len = len(block)
            if used + block_len > budget_chars and selected:
                break
            selected.append(block)
            seen_indices.add(section_idx)
            used += block_len + 2
            if used >= budget_chars:
                break
        if used >= budget_chars:
            break
    return "\n\n".join(selected).strip()


def _load_source_context(
    question: str,
    max_chars: int | None = None,
    max_files: int | None = None,
) -> str:
    """Load relevant Treasury Bulletin source text for a question."""
    default_files, default_chars = _source_context_limits(question)
    if max_chars is None:
        max_chars = default_chars
    if max_files is None:
        max_files = default_files
    source_files = _find_source_files(question, max_files=max_files)
    if not source_files:
        return ""

    context_parts = []
    total_chars = 0
    chars_per_file = max_chars // max(len(source_files), 1)

    for sf in source_files:
        fpath = os.path.join(_treasury_data_dir, sf)
        if not os.path.exists(fpath):
            logger.warning(f"Source file not found: {fpath}")
            continue
        try:
            with open(fpath) as f:
                content = f.read()
            snippet = _extract_relevant_snippets(sf, content, question, chars_per_file)
            if not snippet:
                continue
            if len(snippet) > chars_per_file:
                snippet = snippet[:chars_per_file].rstrip() + "\n[...truncated...]"
            context_parts.append(f"=== SOURCE: {sf} ===\n{snippet}")
            total_chars += len(snippet)
            if total_chars >= max_chars:
                break
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")

    if context_parts:
        return "\n\n".join(context_parts)
    return ""
