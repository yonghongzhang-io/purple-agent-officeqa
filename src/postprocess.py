from __future__ import annotations

import logging
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Callable


logger = logging.getLogger(__name__)


_SCALAR_PATTERN = re.compile(
    r"[-+]?\$?\d[\d,]*\.?\d*(?:\s*(?:%|percent|billion|million|trillion|thousand|dollars))?",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

_PLACEHOLDER_PATTERNS = [
    r"\[?\s*unable to determine\s*\]?",
    r"\[?\s*not found(?: in the provided reference data)?\s*\]?",
    r"\[?\s*not available\s*\]?",
    r"\[?\s*cannot be determined(?: from provided data)?\s*\]?",
    r"\[?\s*not possible to calculate\s*\]?",
    r"\[?\s*no answer found\s*\]?",
    r"\bno\s+.*?data\b",
]


_FORMAT_RULES: list[tuple[str, str, str]] = [
    (
        r"Treasury Bulletin|FINAL_ANSWER|treasury.*expenditure|"
        r"fiscal year.*(?:million|billion|thousand)|public debt.*outstanding",
        "xml_final_answer",
        (
            "OUTPUT FORMAT — end your response with this EXACT tag:\n"
            "<FINAL_ANSWER>\n[value only]\n</FINAL_ANSWER>\n"
            "The value inside the tag must be the bare answer. "
            "Show reasoning first, then the tag at the end."
        ),
    ),
    (
        r"Return JSON|return.*json.*format|answer.*json|provide.*json",
        "json_generic",
        (
            "OUTPUT FORMAT — return your answer as valid JSON.\n"
            "Match the exact structure requested in the task.\n"
            "No markdown fences, no prose outside the JSON object."
        ),
    ),
    (
        r"risk[_\s-]*classification|classif(?:y|ication).*(?:risk|risks)",
        "json_risk_classification",
        (
            "OUTPUT FORMAT — return valid JSON with a top-level "
            "\"risk_classification\" field.\n"
            "Do not wrap the JSON in markdown fences or extra prose."
        ),
    ),
    (
        r"trading decision|stop[_\s-]*loss|take[_\s-]*profit|position size|confidence score",
        "json_trading_decision",
        (
            "OUTPUT FORMAT — return valid JSON with the required trading fields "
            "(such as action, size, stop_loss, take_profit, reasoning, confidence) "
            "using the exact keys requested by the task."
        ),
    ),
    (
        r"\bcot\b|chain of thought|reasoning and answer|\"cot\"",
        "json_cot_answer",
        (
            "OUTPUT FORMAT — return valid JSON with explicit reasoning and answer "
            "fields, such as {\"cot\": \"...\", \"answer\": \"...\"}, matching the task."
        ),
    ),
    (
        r"options?|greeks|delta|gamma|theta|vega|implied volatility",
        "json_options",
        (
            "OUTPUT FORMAT — return valid JSON for the options task. "
            "If the prompt specifies a nested result object, preserve that exact structure."
        ),
    ),
    (
        r"Return.*CSV|output.*CSV|CSV.*format",
        "csv_output",
        (
            "OUTPUT FORMAT — return the result as CSV.\n"
            "First line: header row with exact column names.\n"
            "Subsequent lines: data rows. Comma delimiter."
        ),
    ),
    (
        r"portfolio allocation|allocate.*portfolio|weights? across|target weights?",
        "portfolio_allocation",
        (
            "OUTPUT FORMAT — return each allocation on its own line in the exact "
            "portfolio format requested by the task. Keep ticker symbols, weights, "
            "and any required rationale concise and consistent."
        ),
    ),
    (
        r"business summary|industry.*products.*geography|business overview",
        "json_business_summary",
        (
            "OUTPUT FORMAT — return valid JSON for the business summary with the exact "
            "fields requested, such as industry, products, and geography."
        ),
    ),
    (
        r"consistency check|inconsisten|contradiction",
        "json_consistency_check",
        (
            "OUTPUT FORMAT — return valid JSON for the consistency check using the exact "
            "schema requested by the task."
        ),
    ),
]


def _detect_output_format(task_text: str, task_type: str | None = None) -> tuple[str | None, str]:
    if task_type == "officeqa":
        pattern, fmt_key, directive = _FORMAT_RULES[0]
        if re.search(pattern, task_text, re.IGNORECASE | re.DOTALL):
            return fmt_key, directive
        return None, ""

    for pattern, fmt_key, directive in _FORMAT_RULES:
        if re.search(pattern, task_text, re.IGNORECASE | re.DOTALL):
            return fmt_key, directive
    return None, ""


def _verify_computation(question: str, answer: str) -> str:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer)
    if not nums:
        return answer

    q_lower = question.lower()
    if any(word in q_lower for word in ["percent", "percentage", "%"]):
        if "%" not in answer and "percent" not in answer.lower():
            clean = answer.strip().rstrip(".")
            if re.fullmatch(r"-?\d[\d,]*\.?\d*", clean):
                answer = clean + "%"
                logger.info(f"COMPUTE VERIFY: Added missing % sign -> {answer}")

    if any(word in q_lower for word in ["in dollars", "dollar amount", "how much money"]):
        if "$" not in answer and "dollar" not in answer.lower():
            clean = answer.strip()
            if re.fullmatch(r"-?\d[\d,]*\.?\d*", clean):
                answer = "$" + clean
                logger.info(f"COMPUTE VERIFY: Added missing $ sign -> {answer}")

    if "in millions" in q_lower and "billion" in answer.lower():
        logger.warning("COMPUTE VERIFY: Unit mismatch — question says millions, answer says billion")
    if "in billions" in q_lower and "million" in answer.lower():
        logger.warning("COMPUTE VERIFY: Unit mismatch — question says billions, answer says million")

    return answer


_HEDGE_PATTERNS = [
    r"\beither\b.*\bor\b",
    r"\bcould be\b.*\bor\b",
    r"\bpossibly\b.*\bor\b",
    r"\bmaybe\b.*\bor\b",
    r"\bOne possibility is\b",
    r"\balternatively\b",
]


def _contains_hedge(answer: str) -> bool:
    lower = answer.lower()
    for pattern in _HEDGE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _contains_multiple_numbers(answer: str) -> bool:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", answer)
    if len(numbers) <= 1:
        return False
    unique_values = set()
    for number in numbers:
        cleaned = number.replace(",", "")
        try:
            value = float(cleaned)
            if 1900 <= value <= 2100 and "." not in cleaned:
                continue
            unique_values.add(value)
        except ValueError:
            continue
    return len(unique_values) > 1


def _extract_bare_answer(answer: str) -> str:
    preambles = [
        r"^(?:the\s+)?answer\s+is\s*:?\s*",
        r"^(?:the\s+)?(?:final\s+)?(?:value|result)\s+is\s*:?\s*",
        r"^based\s+on\s+.*?,\s*",
        r"^according\s+to\s+.*?,\s*",
        r"^the\s+total\s+.*?\s+(?:was|is|were)\s+",
        r"^i\s+need\s+to\s+(?:verify|audit|double-check|identify).*?(?:\n\s*\n|\n)",
        r"^the\s+(?:provided|reference)\s+data\s+does\s+not\s+contain.*?(?:\n\s*\n|\n)",
        r"^no\s+.*?data.*?(?:\n\s*\n|\n)",
    ]
    cleaned = answer
    for pattern in preambles:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else answer


def _looks_like_bare_value(text: str) -> bool:
    candidate = text.strip()
    if not candidate or "\n" in candidate:
        return False
    if len(candidate) > 120:
        return False
    if re.search(r"\b(the|answer|based on|according|because|from|table|therefore)\b", candidate, re.IGNORECASE):
        return False
    return bool(
        re.fullmatch(
            r"[-+]?\$?\d[\d,]*\.?\d*(?:\s*(?:%|percent|billion|million|trillion|thousand|dollars))?",
            candidate,
            re.IGNORECASE,
        )
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9 ,.&()/-]{0,80}", candidate)
    )


def _extract_compact_final_line(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) < 2:
        return answer
    last_line = lines[-1]
    return last_line if _looks_like_bare_value(last_line) else answer


def _extract_best_compact_line(answer: str) -> str:
    """Prefer the strongest standalone answer-like line from a verbose response."""
    lines = [line.strip(" -*:\t") for line in answer.splitlines() if line.strip()]
    for line in reversed(lines):
        if _looks_like_bare_value(line):
            return line
    return answer


def _normalize_final_text(answer: str) -> str:
    cleaned = answer.strip().strip("`").strip()
    cleaned = cleaned.strip("\"' ")
    if cleaned.endswith(".") and len(cleaned.split()) <= 6:
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _format_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _is_placeholder_answer(answer: str) -> bool:
    lower = answer.strip().lower()
    if lower in {"", "nan", "+nan", "-nan", "none", "null", "n/a", "na"}:
        return True
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _looks_like_list_answer(answer: str) -> bool:
    candidate = answer.strip()
    if re.fullmatch(r"\[[^\]]+\]", candidate):
        return True
    parts = [part.strip() for part in candidate.split(",")]
    if len(parts) < 2:
        return False
    numericish = 0
    for part in parts:
        part = part.strip("[]() ")
        if re.fullmatch(r"[-+]?\$?\d[\d,]*\.?\d*%?", part):
            numericish += 1
    return numericish >= 2


def _question_asks_percent(question: str) -> bool:
    q_lower = question.lower()
    return any(word in q_lower for word in ["percent", "percentage", "%"])


def _question_wants_percent_symbol(question: str) -> bool:
    q_lower = question.lower()
    return (
        "%" in question
        or "percent value" in q_lower
        or "percentage value" in q_lower
        or "reported as a percent value" in q_lower
        or "report your answer as a percent" in q_lower
    )


def _question_expects_date_or_year(question: str) -> bool:
    q_lower = question.lower()
    return any(
        phrase in q_lower
        for phrase in [
            "what year",
            "which year",
            "what date",
            "which date",
            "what month",
            "which month",
            "what day",
            "which day",
            "reporting date",
            "month and year",
            "year and month",
        ]
    )


def _question_expected_list_len(question: str) -> int | None:
    q_lower = question.lower()
    explicit = re.search(r"\bcontaining\s+(\d+)\s+numbers?\b", q_lower)
    if explicit:
        return int(explicit.group(1))

    ordinal_positions = {
        "first value": 1,
        "second value": 2,
        "third value": 3,
        "fourth value": 4,
        "first number": 1,
        "second number": 2,
        "third number": 3,
        "fourth number": 4,
    }
    best = 0
    for phrase, idx in ordinal_positions.items():
        if phrase in q_lower:
            best = max(best, idx)
    if best:
        return best

    if "slope and intercept" in q_lower:
        return 2

    if "inside square brackets" in q_lower or "enclosed brackets" in q_lower:
        return 2

    return None


def _question_unit(question: str) -> str | None:
    q_lower = question.lower()
    for unit in ["trillion", "billion", "million", "thousand"]:
        if f"in {unit}s" in q_lower or f"in {unit}" in q_lower:
            return unit
    return None


def _parse_scalar_token(token: str) -> tuple[Decimal | None, str | None]:
    cleaned = token.strip().strip("[]() ")
    unit = None
    lower = cleaned.lower()
    for candidate in ["trillion", "billion", "million", "thousand"]:
        if candidate in lower:
            unit = candidate
            cleaned = re.sub(candidate, "", cleaned, flags=re.IGNORECASE).strip()
            break
    cleaned = cleaned.replace("$", "").replace(",", "").replace("percent", "").replace("%", "").replace("dollars", "").strip()
    if not cleaned:
        return None, unit
    try:
        return Decimal(cleaned), unit
    except InvalidOperation:
        return None, unit


def _is_year_like_token(token: str) -> bool:
    cleaned = token.strip().strip("[]() ").replace("$", "").replace("%", "").replace("percent", "").replace(",", "").strip()
    return bool(re.fullmatch(r"(?:19|20)\d{2}", cleaned))


def _is_table_label(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,3}-\d{1,3}[A-Za-z]?", token.strip()))


def _convert_unit_for_question(question: str, answer: str) -> str:
    target_unit = _question_unit(question)
    if not target_unit:
        return answer.replace("$", "").strip()

    value, source_unit = _parse_scalar_token(answer)
    if value is None:
        return answer.replace("$", "").strip()

    # Do not invent a scale for bare numerals. Only convert when the candidate
    # explicitly carries a unit (e.g. "129,124 million").
    if not source_unit:
        return _format_decimal(value)

    scale = {
        "thousand": Decimal("1e3"),
        "million": Decimal("1e6"),
        "billion": Decimal("1e9"),
        "trillion": Decimal("1e12"),
    }
    if source_unit:
        value *= scale[source_unit]
    converted = value / scale[target_unit]
    return _format_decimal(converted)


def _clean_scalar_candidate(token: str) -> str:
    return token.replace("$", "").replace("percent", "%").replace(" ", "").strip()


def _candidate_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return list(reversed(lines))


def _line_is_negative_context(line: str) -> bool:
    lower = line.lower()
    return any(
        phrase in lower
        for phrase in [
            "does not contain",
            "not possible",
            "unable to determine",
            "cannot be determined",
            "not available",
            "not found",
            "impossible to calculate",
            "cannot provide",
        ]
    )


def _extract_percent_answer(question: str, text: str) -> str | None:
    wants_symbol = _question_wants_percent_symbol(question)
    explicit_matches: list[tuple[int, Decimal, str]] = []
    scalar_matches: list[tuple[int, Decimal, str]] = []
    for line in _candidate_lines(text):
        if _line_is_negative_context(line):
            continue
        lower = line.lower()
        line_bonus = 0
        if any(word in lower for word in ["percent", "percentage", "change", "increase", "decrease", "growth", "difference"]):
            line_bonus += 2
        for token in _SCALAR_PATTERN.findall(line):
            if _is_year_like_token(token) or _is_table_label(token):
                continue
            value, _ = _parse_scalar_token(token)
            if value is None:
                continue
            score = line_bonus
            cleaned = _clean_scalar_candidate(token)
            if "%" in cleaned:
                score += 6
                explicit_matches.append((score, value, cleaned))
                continue
            if Decimal("0") < abs(value) < Decimal("1"):
                score += 4
            elif abs(value) <= Decimal("1000"):
                score += 2
            else:
                score -= 10
            scalar_matches.append((score, value, cleaned))

    if explicit_matches:
        explicit_matches.sort(key=lambda item: item[0], reverse=True)
        chosen = explicit_matches[0][2]
        return chosen if wants_symbol else chosen.replace("%", "")

    if scalar_matches:
        scalar_matches.sort(key=lambda item: item[0], reverse=True)
        _, value, _ = scalar_matches[0]
        if Decimal("0") < abs(value) < Decimal("1"):
            value *= Decimal("100")
        rendered = _format_decimal(value)
        return rendered + "%" if wants_symbol else rendered
    return None


def _normalize_list_answer(answer: str) -> str:
    inner = answer.strip()[1:-1]
    parts = [part.strip() for part in inner.split(",") if part.strip()]
    return "[" + ", ".join(_clean_scalar_candidate(part) for part in parts) + "]"


def _extract_bracketed_numeric_list(text: str, expected_len: int | None) -> str | None:
    matches = re.findall(r"\[([^\]]+)\]", text)
    for match in reversed(matches):
        parts = [part.strip() for part in match.split(",") if part.strip()]
        if expected_len is not None and len(parts) != expected_len:
            continue
        if not parts:
            continue
        normalized = []
        valid = True
        for part in parts:
            cleaned = part.strip("[]() ")
            if not re.fullmatch(
                r"[-+]?\$?\d[\d,]*\.?\d*(?:\s*(?:%|percent|billion|million|trillion|thousand|dollars))?",
                cleaned,
                re.IGNORECASE,
            ):
                valid = False
                break
            normalized.append(_clean_scalar_candidate(cleaned))
        if valid and (expected_len is None or len(normalized) == expected_len):
            return "[" + ", ".join(normalized) + "]"
    return None


def _select_scalar_candidate(question: str, text: str) -> str | None:
    target_unit = _question_unit(question)
    q_lower = question.lower()
    expects_year_or_date = _question_expects_date_or_year(question)
    best: tuple[int, str] | None = None

    for line in _candidate_lines(text):
        lower = line.lower()
        if _line_is_negative_context(line):
            continue
        line_score = 0
        if any(
            cue in lower
            for cue in [
                "total",
                "amount",
                "value",
                "sum",
                "highest",
                "lowest",
                "outstanding",
                "balance",
                "cost",
                "receipts",
                "outlays",
                "expenditures",
                "bids",
                "accepted",
                "weighted average",
                "listed as",
                "totaled",
                "at ",
            ]
        ):
            line_score += 3
        if any(month in lower for month in _MONTH_NAMES):
            line_score -= 1

        for token in _SCALAR_PATTERN.findall(line):
            raw = token.strip()
            if _is_table_label(raw):
                continue
            if _is_year_like_token(raw) and not expects_year_or_date:
                continue
            value, unit = _parse_scalar_token(raw)
            if value is None:
                continue
            score = line_score
            if target_unit and unit == target_unit:
                score += 8
            elif unit:
                score += 3
            if not unit and not _question_asks_percent(question) and Decimal("0") < abs(value) < Decimal("1"):
                score -= 6
            if "full number" in q_lower and unit:
                score += 2
            candidate = _convert_unit_for_question(question, raw) if target_unit else _clean_scalar_candidate(raw)
            if best is None or score > best[0]:
                best = (score, candidate)

    return best[1] if best else None


def _is_suspicious_officeqa_answer(question: str, answer: str) -> bool:
    candidate = answer.strip()
    if not candidate:
        return True

    expected_list_len = _question_expected_list_len(question)
    if expected_list_len and not _looks_like_list_answer(candidate):
        return True

    if _is_table_label(candidate):
        return True

    if _is_year_like_token(candidate) and not _question_expects_date_or_year(question):
        return True

    value, unit = _parse_scalar_token(candidate)
    if value is None:
        return False

    if _question_asks_percent(question):
        if _is_year_like_token(candidate):
            return True
        if abs(value) > Decimal("1000"):
            return True
        return False

    target_unit = _question_unit(question)
    if target_unit and unit is None and Decimal("0") < abs(value) < Decimal("1"):
        return True

    q_lower = question.lower()
    if "full number" in q_lower and unit is None and Decimal("0") < abs(value) < Decimal("1"):
        return True

    return False


def _salvage_officeqa_answer(question: str, answer: str, response: str) -> str:
    combined = "\n".join(part for part in [answer, response] if part)
    expected_list_len = _question_expected_list_len(question)

    if expected_list_len and _looks_like_list_answer(answer):
        return _normalize_list_answer(answer)

    if expected_list_len:
        bracketed = _extract_bracketed_numeric_list(combined, expected_list_len)
        if bracketed:
            return bracketed
        return answer

    if _question_asks_percent(question):
        percent = _extract_percent_answer(question, combined)
        if percent and (_is_placeholder_answer(answer) or _is_suspicious_officeqa_answer(question, answer)):
            return percent

    if _is_placeholder_answer(answer) or _looks_like_list_answer(answer) or _is_suspicious_officeqa_answer(question, answer):
        candidate = _select_scalar_candidate(question, combined)
        if candidate:
            return candidate

    if _question_unit(question):
        return _convert_unit_for_question(question, answer)

    q_lower = question.lower()
    if "$" in answer and "dollar" not in q_lower and "money" not in q_lower:
        return answer.replace("$", "").strip()

    return answer


def _clean_response(
    response: str,
    original_prompt: str = "",
    provider: str = "",
    task_type: str = "",
    verify_with_llm: Callable[[str, str, str], str] | None = None,
    should_run_numeric_audit: Callable[[str, str], bool] | None = None,
) -> str:
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    while "<think>" in response:
        idx = response.index("<think>")
        after = response[idx:]
        close = re.search(r"</(?:FINAL_ANSWER|REASONING)>", after, re.IGNORECASE)
        if close:
            response = response[:idx] + after[close.start():]
        else:
            response = response[:idx].strip()
        break

    if "<FINAL_ANSWER>" not in response and "<final_answer>" not in response.lower():
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        answer = lines[-1] if lines else response
        response = f"<REASONING>\n{response}\n</REASONING>\n<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>"

    match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        if "<think>" in answer:
            answer = answer[:answer.index("<think>")].strip()
        answer = answer.strip()

        if not answer:
            reasoning_match = re.search(r"<REASONING>(.*?)</REASONING>", response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
                if lines:
                    answer = lines[-1]
                    answer = _extract_bare_answer(answer)
                    logger.info(f"FINAL_ANSWER was empty — extracted from REASONING: {answer[:100]}")

        if "no answer found" in answer.lower():
            answer = "Unable to determine from available data"

        answer = _extract_bare_answer(answer)
        answer = _extract_compact_final_line(answer)
        answer = _extract_best_compact_line(answer)
        answer = _normalize_final_text(answer)
        if task_type == "officeqa":
            if _is_placeholder_answer(answer) or not answer.strip():
                answer = _salvage_officeqa_answer(original_prompt, answer, response)
            answer = _normalize_final_text(answer)

        if original_prompt and task_type != "officeqa":
            answer = _verify_computation(original_prompt, answer)
            enable_numeric_audit = os.environ.get("ENABLE_NUMERIC_AUDIT", "false").lower() == "true"
            if (
                task_type != "officeqa"
                and enable_numeric_audit
                and verify_with_llm
                and should_run_numeric_audit
                and should_run_numeric_audit(original_prompt, answer)
            ):
                answer = verify_with_llm(original_prompt, answer, provider)

        if len(answer) > 500:
            answer = answer[:500].rsplit(" ", 1)[0] if " " in answer[:500] else answer[:500]

        if _contains_multiple_numbers(answer):
            logger.warning(f"MULTIPLE NUMBERS in FINAL_ANSWER: {answer[:100]}...")
        if _contains_hedge(answer):
            logger.warning(f"HEDGE DETECTED in FINAL_ANSWER: {answer[:100]}...")

        response = re.sub(
            r"(<FINAL_ANSWER>).*?(</FINAL_ANSWER>)",
            rf"\g<1>\n{answer}\n\g<2>",
            response,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )

    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    if "<think>" in response:
        response = response[:response.index("<think>")].strip()
    if "<FINAL_ANSWER>" not in response and "<final_answer>" not in response.lower():
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        answer = lines[-1] if lines else "Unable to determine"
        response = f"<REASONING>\n{response}\n</REASONING>\n<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>"

    return response


def _extract_final_answer_text(response: str) -> str:
    match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return _extract_compact_final_line(_extract_bare_answer(answer))
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _parse_vote_number(text: str) -> float | None:
    s = text.strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    match = re.match(r"^\((.+)\)$", s)
    if match:
        s = "-" + match.group(1)
    number = re.search(r"[-+]?\d+\.?\d*", s)
    if not number:
        return None
    try:
        value = float(number.group())
    except ValueError:
        return None
    lower = text.lower()
    if "trillion" in lower:
        value *= 1e12
    elif "billion" in lower:
        value *= 1e9
    elif "million" in lower:
        value *= 1e6
    elif "thousand" in lower:
        value *= 1e3
    return value


def _vote_answers(answers: list[str]) -> str:
    answers = [answer.strip() for answer in answers if answer and answer.strip()]
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]

    parsed = []
    for answer in answers:
        value = _parse_vote_number(answer)
        if value is not None:
            parsed.append((value, answer))

    if len(parsed) >= len(answers) * 0.5:
        parsed.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, str]]] = []
        for value, answer in parsed:
            placed = False
            for cluster in clusters:
                center = cluster[0][0]
                tolerance = 0.5 if center == 0 else abs(center) * 0.02
                if abs(value - center) <= tolerance:
                    cluster.append((value, answer))
                    placed = True
                    break
            if not placed:
                clusters.append([(value, answer)])
        best_cluster = max(clusters, key=len)
        return best_cluster[len(best_cluster) // 2][1]

    counts: dict[str, int] = {}
    for answer in answers:
        counts[answer.lower()] = counts.get(answer.lower(), 0) + 1
    winner = max(counts, key=counts.get)
    for answer in answers:
        if answer.lower() == winner:
            return answer
    return answers[0]
