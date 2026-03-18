import asyncio
import json
import logging
import os
import re
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    DataPart,
    Message,
    Part,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Provider defaults for OpenAI-compatible services
PROVIDER_DEFAULTS = {
    "openai":    (None,                                 "gpt-4o"),
    "groq":      ("https://api.groq.com/openai/v1",    "llama-3.3-70b-versatile"),
    "nebius":    ("https://api.studio.nebius.ai/v1",    "meta-llama/Llama-3.3-70B-Instruct"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai","meta-llama/Llama-3.3-70B-Instruct-Turbo"),
}


# ---------------------------------------------------------------------------
# System prompts — specialized per task type
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BWIM = """You are an expert AI agent competing in the AgentX-AgentBeats Sprint 1 competition.
Your current task is a BWIM (Build What I Mean) instruction-following task.

RULES:
1. Follow the instruction EXACTLY. Do precisely what is asked — nothing more, nothing less.
2. Pay close attention to format requirements (JSON, CSV, plain text, code, etc.).
3. If asked to produce structured data, ensure it is valid and properly formatted.
4. If asked to transform text, preserve all content not explicitly changed.
5. Be precise with numbers, names, capitalization, and punctuation.
6. NEVER hedge or give multiple possible answers. Give ONE definitive answer.
7. Keep your FINAL_ANSWER under 500 characters.

RESPONSE FORMAT:
<REASONING>
[Brief step-by-step analysis of what the instruction requires]
</REASONING>
<FINAL_ANSWER>
[Your complete answer — the exact output requested, nothing else]
</FINAL_ANSWER>

CRITICAL:
- The <FINAL_ANSWER> tag MUST contain ONLY the requested output. No explanations, no caveats, no "Here is..." preamble.
- NEVER say "either X or Y" or "could be X or Y" — pick ONE answer.
- Keep FINAL_ANSWER under 500 characters."""

SYSTEM_PROMPT_CRM = """You are an expert AI agent competing in the AgentX-AgentBeats Sprint 1 competition.
Your current task is a CRMArena task involving CRM data operations.

RULES:
1. When given CRM data (contacts, accounts, opportunities, leads), analyze it precisely.
2. For queries: extract the exact information requested with full precision.
3. For calculations: show your work, use exact numbers, handle currency/percentages correctly.
4. For data manipulation: output valid JSON matching the schema exactly.
5. Handle edge cases: null values, empty fields, date formats, currency symbols.
6. When filtering: apply ALL conditions, don't miss any.
7. When aggregating: include ALL matching records, don't skip any.

RESPONSE FORMAT:
<REASONING>
[Step-by-step analysis showing your work. For calculations, show each step.]
</REASONING>
<FINAL_ANSWER>
[Your precise answer. For structured data, output valid JSON. For numbers, give exact values.]
</FINAL_ANSWER>

CRITICAL:
- The <FINAL_ANSWER> tag MUST contain ONLY the requested output.
- NEVER hedge or give multiple possible answers. Give ONE definitive answer.
- Keep FINAL_ANSWER under 500 characters."""

SYSTEM_PROMPT_FINANCIAL = """You are an expert AI agent competing in the AgentX-AgentBeats Sprint 1 competition.
Your current task involves financial document analysis and quantitative reasoning.

RULES:
1. Ensure NUMERICAL ACCURACY — use full precision, do not round unless explicitly asked.
2. For extraction tasks: find the exact value from the source data.
3. For calculation tasks: show every step, handle unit conversions (million, billion, trillion).
4. For statistical tasks: use correct formulas (std dev, correlation, regression, VaR).
5. For time series: handle date ranges correctly, include all data points.
6. When units are involved, state them clearly in your answer.
7. If a percentage is asked, give the percentage. If a dollar amount, give dollars.
8. NEVER hedge or give multiple possible answers. Give ONE definitive answer.
9. Keep your FINAL_ANSWER under 500 characters.
10. Use commas in large numbers only if the source does. The scoring is exact.

SCORING INFO (how your answer is judged):
- Numerical answers: exact match by default (0% tolerance). "2602" must match "2602".
- Commas are stripped: "2,602" = "2602".
- Units like "billion", "million" are detected from context words.
- Text answers: case-insensitive substring match against ground truth.
- If you give multiple candidate answers, you AUTOMATICALLY FAIL (hedge detection).

RESPONSE FORMAT:
<REASONING>
[Detailed step-by-step calculations. Show ALL intermediate values.]
</REASONING>
<FINAL_ANSWER>
[The precise value. Include units if the question implies them. ONE answer only.]
</FINAL_ANSWER>

CRITICAL:
- The <FINAL_ANSWER> tag MUST contain ONLY the value requested.
- NEVER say "either X or Y" or "approximately" — give the EXACT value.
- Keep FINAL_ANSWER under 500 characters."""

SYSTEM_PROMPT_OFFICEQA = """You are an expert AI agent answering questions about the U.S. Treasury Bulletin.
The dataset spans January 1939 to September 2025 and includes financial data from U.S. Treasury Bulletins.

IMPORTANT: You will be provided with REFERENCE DATA from the actual Treasury Bulletin documents.
Use ONLY the provided reference data to find exact numbers. Do NOT guess or make up values.
Search the reference data carefully for tables, figures, and statistics that answer the question.

QUESTION TYPES:
- Simple data extraction (e.g., "What was the total receipts in fiscal year 2020?")
- Multi-year calculations with inflation adjustments
- Statistical analysis (regression, correlation, standard deviation)
- Time series forecasting
- Complex financial metrics (VaR, weighted averages)

RULES:
1. ALWAYS search for the exact data first. Do NOT rely on memory for numbers.
2. Ensure FULL NUMERICAL PRECISION — never round unless explicitly asked.
3. Match the exact value from the source. "2602" must be "2602", not "approximately 2600".
4. Handle unit conversions correctly: million, billion, trillion.
5. If the answer is a dollar amount, include the dollar sign or "dollars".
6. If the answer is a percentage, include the percent sign.
7. For dates, use the exact format from the source data.
8. NEVER hedge. Give ONE definitive answer, never "either X or Y".
9. Keep FINAL_ANSWER under 500 characters.

SCORING (exact rules used by the judge):
- Numbers: exact match (0% tolerance). Commas are stripped ("2,602" = "2602").
- Units: "billion", "million", "thousand" detected from context words.
- Text: case-insensitive substring match.
- Hedging: if you give multiple candidate answers = AUTOMATIC FAIL.
- "No answer found" = AUTOMATIC FAIL.

CRITICAL — FINAL_ANSWER FORMAT:
- Put ONLY the bare value inside <FINAL_ANSWER> tags. NO sentences, NO explanations.
- Good: <FINAL_ANSWER>2602</FINAL_ANSWER>
- Good: <FINAL_ANSWER>$3.2 billion</FINAL_ANSWER>
- BAD: <FINAL_ANSWER>The total receipts were 2602 million dollars</FINAL_ANSWER>
- BAD: <FINAL_ANSWER>Based on my analysis, the answer is 2602</FINAL_ANSWER>
- Put ONLY ONE number in FINAL_ANSWER. Multiple numbers = AUTOMATIC FAIL.

RESPONSE FORMAT:
<REASONING>
[Show your work. For calculations, show every step with full precision.]
</REASONING>
<FINAL_ANSWER>
[ONLY the bare value. ONE answer. No words except units.]
</FINAL_ANSWER>"""

SYSTEM_PROMPT_GENERAL = """You are an expert AI agent competing in the AgentX-AgentBeats Sprint 1 competition.
You handle diverse tasks with precision and accuracy.

RULES:
1. Follow instructions exactly as given.
2. Be precise: numbers, names, and formats must match exactly what is asked.
3. For structured data output, use valid JSON.
4. For text output, match the requested format precisely.
5. When uncertain, pick the most reasonable interpretation.
6. NEVER hedge or give multiple possible answers. Give ONE definitive answer.
7. Keep FINAL_ANSWER under 500 characters.

RESPONSE FORMAT:
<REASONING>
[Brief step-by-step analysis]
</REASONING>
<FINAL_ANSWER>
[Your complete, precise answer]
</FINAL_ANSWER>

CRITICAL:
- Always produce a <FINAL_ANSWER> tag. Only put the requested output inside it.
- NEVER say "either X or Y" — pick ONE answer.
- Keep FINAL_ANSWER under 500 characters."""


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------
CRM_KEYWORDS = {
    "crm", "contact", "account", "opportunity", "lead", "pipeline",
    "salesforce", "deal", "revenue", "customer", "sales stage",
    "closed won", "closed lost", "prospect", "funnel",
}

FINANCIAL_KEYWORDS = {
    "treasury", "bulletin", "fiscal", "budget", "debt", "bond",
    "yield", "inflation", "gdp", "deficit", "surplus", "securities",
    "receipts", "expenditures", "appropriation", "federal",
}

# OfficeQA requires STRONG signals — must clearly reference Treasury Bulletin documents
OFFICEQA_STRONG_PHRASES = {
    "treasury bulletin", "u.s. treasury bulletin",
    "treasury department bulletin",
}
OFFICEQA_CONTEXT_PHRASES = {
    "public debt outstanding", "federal receipts", "treasury balance",
    "national debt", "u.s. treasury",
}
# These only count as OfficeQA when paired with question-style + fiscal/budget context
OFFICEQA_WEAK_SIGNALS = {
    "fiscal year", "treasury department",
}

BWIM_KEYWORDS = {
    "create", "build", "generate", "write", "convert", "transform",
    "rewrite", "format", "produce", "make", "compose", "draft",
    "translate", "restructure", "reorganize",
}


def _heuristic_task_routing(prompt: str) -> tuple[str, int]:
    """Heuristic task routing with a coarse confidence score."""
    lower = prompt.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    # 1. OfficeQA — strong signal: explicit Treasury Bulletin reference
    strong_hits = sum(1 for phrase in OFFICEQA_STRONG_PHRASES if phrase in lower)
    if strong_hits >= 1:
        return "officeqa", 4

    # OfficeQA — medium signal: Treasury context phrases (need >= 2)
    context_hits = sum(1 for phrase in OFFICEQA_CONTEXT_PHRASES if phrase in lower)
    if context_hits >= 2:
        return "officeqa", 3

    # OfficeQA — weak signal + fiscal context: only if also has budget/expenditure language
    weak_hits = sum(1 for phrase in OFFICEQA_WEAK_SIGNALS if phrase in lower)
    fiscal_words = words & {"expenditures", "receipts", "appropriation", "disbursements"}
    if weak_hits >= 1 and len(fiscal_words) >= 1:
        return "officeqa", 2

    # 2. CRM
    crm_score = len(words & CRM_KEYWORDS)
    if crm_score >= 2:
        return "crm", crm_score

    # 3. BWIM — but only if not also a financial question
    financial_score = len(words & FINANCIAL_KEYWORDS)
    bwim_score = len(words & BWIM_KEYWORDS)
    if bwim_score >= 2 or (bwim_score >= 1 and financial_score == 0):
        return "bwim", bwim_score

    # 4. Financial — general finance, NOT officeqa
    if financial_score >= 2:
        return "financial", financial_score

    # 5. Default — general
    return "general", 0


def detect_task_type(prompt: str) -> str:
    """Detect task type from prompt content to select the best system prompt."""
    task_type, _ = _heuristic_task_routing(prompt)
    return task_type


SYSTEM_PROMPTS = {
    "bwim": SYSTEM_PROMPT_BWIM,
    "crm": SYSTEM_PROMPT_CRM,
    "financial": SYSTEM_PROMPT_FINANCIAL,
    "officeqa": SYSTEM_PROMPT_OFFICEQA,
    "general": SYSTEM_PROMPT_GENERAL,
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS_ANTHROPIC = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports arithmetic, power (**), modulo (%), and common math. Input must be a valid Python arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression, e.g. '(100 * 1.05) / 3' or '2**10' or 'round(3.14159, 2)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "format_json",
        "description": "Parse, validate, and pretty-print a JSON string. Returns formatted JSON or an error if invalid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "A JSON string to parse and format"
                }
            },
            "required": ["json_string"]
        }
    },
    {
        "name": "search_data",
        "description": "Search and filter a JSON array of objects by field values. Returns matching records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "A JSON array of objects to search through"
                },
                "field": {
                    "type": "string",
                    "description": "The field name to filter on"
                },
                "value": {
                    "type": "string",
                    "description": "The value to match (case-insensitive substring match)"
                }
            },
            "required": ["data", "field", "value"]
        }
    },
    {
        "name": "aggregate",
        "description": "Aggregate a numeric field from a JSON array of objects. Supports sum, avg, min, max, count operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "A JSON array of objects"
                },
                "field": {
                    "type": "string",
                    "description": "The numeric field to aggregate"
                },
                "operation": {
                    "type": "string",
                    "enum": ["sum", "avg", "min", "max", "count"],
                    "description": "The aggregation operation"
                }
            },
            "required": ["data", "field", "operation"]
        }
    },
]

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
    }
    for t in TOOLS_ANTHROPIC
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a local tool and return the result as a string."""
    try:
        if tool_name == "calculate":
            expr = tool_input.get("expression", "")
            # Allow arithmetic + round/abs/min/max/pow
            import math
            safe_globals = {"__builtins__": {}}
            safe_locals = {
                "round": round, "abs": abs, "min": min, "max": max, "pow": pow,
                "sum": sum, "len": len, "int": int, "float": float,
                "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e,
            }
            result = eval(expr, safe_globals, safe_locals)  # noqa: S307
            return str(result)

        if tool_name == "format_json":
            raw = tool_input.get("json_string", "")
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2)

        if tool_name == "search_data":
            data = json.loads(tool_input.get("data", "[]"))
            field = tool_input.get("field", "")
            value = tool_input.get("value", "").lower()
            results = [
                r for r in data
                if field in r and value in str(r[field]).lower()
            ]
            return json.dumps(results, indent=2)

        if tool_name == "aggregate":
            data = json.loads(tool_input.get("data", "[]"))
            field = tool_input.get("field", "")
            operation = tool_input.get("operation", "sum")
            values = [
                float(r[field]) for r in data
                if field in r and r[field] is not None
            ]
            if not values:
                return "No numeric values found for field"
            ops = {
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            return str(ops.get(operation, "Unknown operation"))

    except Exception as e:
        return f"Tool error: {e}"

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Treasury source corpus — keyword-based retrieval (no answer data)
# ---------------------------------------------------------------------------
_treasury_source_files: list[str] = []  # cached list of .txt filenames


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


def _find_source_files(question: str) -> list[str]:
    """Find relevant Treasury source files by extracting dates from the question.

    Uses year/month mentions in the question to select matching files.
    Falls back to loading all files for the mentioned fiscal year(s).
    """
    available = _list_source_files()
    if not available:
        return []

    lower = question.lower()
    matched = []

    # Extract explicit year-month patterns: "January 2020", "March 1945", etc.
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    for month_name, month_num in month_map.items():
        pattern = rf'{month_name}\s+(\d{{4}})'
        for m in re.finditer(pattern, lower):
            year = m.group(1)
            fname = f"treasury_bulletin_{year}_{month_num}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)

    # Extract standalone years (4 digits between 1939-2025)
    years_found = set()
    for m in re.finditer(r'\b(19[3-9]\d|20[0-2]\d)\b', lower):
        years_found.add(m.group(1))

    # "fiscal year YYYY" typically ends in September of that year
    for m in re.finditer(r'fiscal\s+year\s+(\d{4})', lower):
        fy = m.group(1)
        years_found.add(fy)
        # Fiscal year YYYY runs Oct (YYYY-1) to Sep (YYYY) — add key months
        prev_year = str(int(fy) - 1)
        for month in ["10", "11", "12"]:
            fname = f"treasury_bulletin_{prev_year}_{month}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            fname = f"treasury_bulletin_{fy}_{month}.txt"
            if fname in available and fname not in matched:
                matched.append(fname)

    # For standalone years without month, add all months of that year
    if not matched:
        for year in sorted(years_found):
            for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
                fname = f"treasury_bulletin_{year}_{month}.txt"
                if fname in available and fname not in matched:
                    matched.append(fname)

    # Limit to avoid overloading context
    if len(matched) > 24:
        matched = matched[:24]

    if matched:
        logger.info(f"Source retrieval: {len(matched)} files for years {sorted(years_found)}")
    else:
        logger.warning(f"No source files matched for question: {question[:100]}...")

    return matched


def _load_source_context(question: str, max_chars: int = 120000) -> str:
    """Load relevant Treasury Bulletin source text for a question.

    Returns the source document text truncated to fit in LLM context.
    """
    source_files = _find_source_files(question)
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
            # Truncate if needed
            if len(content) > chars_per_file:
                content = content[:chars_per_file] + "\n[...truncated...]"
            context_parts.append(f"=== SOURCE: {sf} ===\n{content}")
            total_chars += len(content)
            if total_chars >= max_chars:
                break
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")

    if context_parts:
        return "\n\n".join(context_parts)
    return ""


# ---------------------------------------------------------------------------
# Conversation history per context_id (supports multi-turn)
# Max history entries to prevent memory leak across 246+ questions.
# ---------------------------------------------------------------------------
_conversation_history: dict[str, list[dict]] = {}
_MAX_HISTORY_ENTRIES = 100  # Evict oldest contexts when exceeded


def _get_history(context_id: str) -> list[dict]:
    return _conversation_history.setdefault(context_id, [])


def _append_history(context_id: str, role: str, content: str) -> None:
    _conversation_history.setdefault(context_id, []).append(
        {"role": role, "content": content}
    )
    # Evict oldest contexts to prevent memory leak
    if len(_conversation_history) > _MAX_HISTORY_ENTRIES:
        oldest_key = next(iter(_conversation_history))
        del _conversation_history[oldest_key]


def _clear_history(context_id: str) -> None:
    """Clear history for a context — called when a task completes."""
    _conversation_history.pop(context_id, None)


# ---------------------------------------------------------------------------
# Format Adapter — detect required output format from task text (zero API cost)
# ---------------------------------------------------------------------------
_FORMAT_RULES: list[tuple[str, str, str]] = [
    # OfficeQA / Treasury Bulletin
    (
        r'Treasury Bulletin|FINAL_ANSWER|treasury.*expenditure|'
        r'fiscal year.*(?:million|billion|thousand)|public debt.*outstanding',
        "xml_final_answer",
        (
            "OUTPUT FORMAT — end your response with this EXACT tag:\n"
            "<FINAL_ANSWER>\n[value only]\n</FINAL_ANSWER>\n"
            "The value inside the tag must be the bare answer. "
            "Show reasoning first, then the tag at the end."
        ),
    ),
    # JSON with specific schema (e.g. risk classification, trading)
    (
        r'Return JSON|return.*json.*format|answer.*json|provide.*json',
        "json_generic",
        (
            "OUTPUT FORMAT — return your answer as valid JSON.\n"
            "Match the exact structure requested in the task.\n"
            "No markdown fences, no prose outside the JSON object."
        ),
    ),
    (
        r'risk[_\s-]*classification|classif(?:y|ication).*(?:risk|risks)',
        "json_risk_classification",
        (
            "OUTPUT FORMAT — return valid JSON with a top-level "
            "\"risk_classification\" field.\n"
            "Do not wrap the JSON in markdown fences or extra prose."
        ),
    ),
    (
        r'trading decision|stop[_\s-]*loss|take[_\s-]*profit|position size|confidence score',
        "json_trading_decision",
        (
            "OUTPUT FORMAT — return valid JSON with the required trading fields "
            "(such as action, size, stop_loss, take_profit, reasoning, confidence) "
            "using the exact keys requested by the task."
        ),
    ),
    (
        r'\bcot\b|chain of thought|reasoning and answer|\"cot\"',
        "json_cot_answer",
        (
            "OUTPUT FORMAT — return valid JSON with explicit reasoning and answer "
            "fields, such as {\"cot\": \"...\", \"answer\": \"...\"}, matching the task."
        ),
    ),
    (
        r'options?|greeks|delta|gamma|theta|vega|implied volatility',
        "json_options",
        (
            "OUTPUT FORMAT — return valid JSON for the options task. "
            "If the prompt specifies a nested result object, preserve that exact structure."
        ),
    ),
    # CSV output
    (
        r'Return.*CSV|output.*CSV|CSV.*format',
        "csv_output",
        (
            "OUTPUT FORMAT — return the result as CSV.\n"
            "First line: header row with exact column names.\n"
            "Subsequent lines: data rows. Comma delimiter."
        ),
    ),
    (
        r'portfolio allocation|allocate.*portfolio|weights? across|target weights?',
        "portfolio_allocation",
        (
            "OUTPUT FORMAT — return each allocation on its own line in the exact "
            "portfolio format requested by the task. Keep ticker symbols, weights, "
            "and any required rationale concise and consistent."
        ),
    ),
    (
        r'business summary|industry.*products.*geography|business overview',
        "json_business_summary",
        (
            "OUTPUT FORMAT — return valid JSON for the business summary with the exact "
            "fields requested, such as industry, products, and geography."
        ),
    ),
    (
        r'consistency check|inconsisten|contradiction',
        "json_consistency_check",
        (
            "OUTPUT FORMAT — return valid JSON for the consistency check using the exact "
            "schema requested by the task."
        ),
    ),
]


def _detect_output_format(task_text: str) -> tuple[str | None, str]:
    """Detect required output format from task text. Returns (format_key, directive)."""
    for pattern, fmt_key, directive in _FORMAT_RULES:
        if re.search(pattern, task_text, re.IGNORECASE | re.DOTALL):
            return fmt_key, directive
    return None, ""


# ---------------------------------------------------------------------------
# Compute Verifier — lightweight arithmetic sanity check (no API cost)
# ---------------------------------------------------------------------------
def _verify_computation(question: str, answer: str) -> str:
    """Check if a numeric answer is self-consistent. Fix common issues.

    This is a zero-API-cost local check, not an LLM call.
    Returns the (possibly corrected) answer.
    """
    # Only verify numeric answers
    nums = re.findall(r'-?\d[\d,]*\.?\d*', answer)
    if not nums:
        return answer

    # Check: percentage answers should include % if question asks for percent
    q_lower = question.lower()
    if any(w in q_lower for w in ["percent", "percentage", "%"]):
        if "%" not in answer and "percent" not in answer.lower():
            # If answer is a bare number that looks like a percentage, add %
            clean = answer.strip().rstrip(".")
            if re.fullmatch(r'-?\d[\d,]*\.?\d*', clean):
                answer = clean + "%"
                logger.info(f"COMPUTE VERIFY: Added missing % sign -> {answer}")

    # Check: dollar answers should include $ if question asks for dollars
    if any(w in q_lower for w in ["in dollars", "dollar amount", "how much money"]):
        if "$" not in answer and "dollar" not in answer.lower():
            clean = answer.strip()
            if re.fullmatch(r'-?\d[\d,]*\.?\d*', clean):
                answer = "$" + clean
                logger.info(f"COMPUTE VERIFY: Added missing $ sign -> {answer}")

    # Check: unit consistency (billion/million) — if question says "in millions"
    # and answer has "billion", flag it
    if "in millions" in q_lower and "billion" in answer.lower():
        logger.warning(f"COMPUTE VERIFY: Unit mismatch — question says millions, answer says billion")
    if "in billions" in q_lower and "million" in answer.lower():
        logger.warning(f"COMPUTE VERIFY: Unit mismatch — question says billions, answer says million")

    return answer


# ---------------------------------------------------------------------------
# Hedge detection — mirrors the judge's contains_multiple_candidates()
# ---------------------------------------------------------------------------
_HEDGE_PATTERNS = [
    r'\beither\b.*\bor\b',
    r'\bcould be\b.*\bor\b',
    r'\bpossibly\b.*\bor\b',
    r'\bmaybe\b.*\bor\b',
    r'\bOne possibility is\b',
    r'\balternatively\b',
]


def _contains_hedge(answer: str) -> bool:
    """Check if FINAL_ANSWER contains hedging language (multiple candidates)."""
    lower = answer.lower()
    for pattern in _HEDGE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _contains_multiple_numbers(answer: str) -> bool:
    """Mirror the judge's hedge detection: count distinct non-year numbers in FINAL_ANSWER.
    If there are 2+ distinct numeric values (excluding years 1900-2100), it's a hedge."""
    # Extract all numbers (with optional commas/decimals)
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', answer)
    if len(numbers) <= 1:
        return False
    # Normalize: strip commas
    unique_values = set()
    for n in numbers:
        cleaned = n.replace(",", "")
        try:
            val = float(cleaned)
            # Skip years (1900-2100)
            if 1900 <= val <= 2100 and "." not in cleaned:
                continue
            unique_values.add(val)
        except ValueError:
            continue
    return len(unique_values) > 1


# ---------------------------------------------------------------------------
# Output cleaning and validation
# ---------------------------------------------------------------------------
def _extract_bare_answer(answer: str) -> str:
    """Strip preamble like 'The answer is ...' to get the bare value."""
    # Remove common preambles
    preambles = [
        r'^(?:the\s+)?answer\s+is\s*:?\s*',
        r'^based\s+on\s+.*?,\s*',
        r'^according\s+to\s+.*?,\s*',
        r'^the\s+total\s+.*?\s+(?:was|is|were)\s+',
    ]
    cleaned = answer
    for p in preambles:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else answer


def _looks_like_bare_value(text: str) -> bool:
    """Return True when the text is already a compact value-like answer."""
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
    """Use the last line only when it is clearly a standalone compact answer."""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) < 2:
        return answer
    last_line = lines[-1]
    return last_line if _looks_like_bare_value(last_line) else answer


def _clean_response(
    response: str,
    original_prompt: str = "",
    provider: str = "",
) -> str:
    """Ensure the response has well-formed FINAL_ANSWER tags and meets scoring requirements.

    Scoring rules (from judge/src/agent.py):
    - FINAL_ANSWER tags are required (case-insensitive)
    - Answer must be non-empty and under 500 characters
    - Must not contain "no answer found"
    - Multiple candidate answers = automatic fail (hedge detection)
    """
    # If no FINAL_ANSWER tag, wrap the entire response
    if "<FINAL_ANSWER>" not in response and "<final_answer>" not in response.lower():
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        answer = lines[-1] if lines else response
        return f"<REASONING>\n{response}\n</REASONING>\n<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>"

    # Validate FINAL_ANSWER content
    match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()

        # Reject "no answer found" — judge auto-fails this
        if "no answer found" in answer.lower():
            answer = "Unable to determine from available data"

        # Strip preamble to get bare value, then optionally keep a compact final line.
        answer = _extract_bare_answer(answer)
        answer = _extract_compact_final_line(answer)

        # Compute verifier — check units, signs, consistency (zero API cost)
        if original_prompt:
            answer = _verify_computation(original_prompt, answer)
            if _should_run_numeric_audit(original_prompt, answer):
                answer = _verify_with_llm(original_prompt, answer, provider)

        # Truncate to 500 chars
        if len(answer) > 500:
            answer = answer[:500].rsplit(" ", 1)[0] if " " in answer[:500] else answer[:500]

        # Check for multiple numbers (judge's hedge detection)
        if _contains_multiple_numbers(answer):
            logger.warning(f"MULTIPLE NUMBERS in FINAL_ANSWER: {answer[:100]}...")

        # Check text hedge patterns
        if _contains_hedge(answer):
            logger.warning(f"HEDGE DETECTED in FINAL_ANSWER: {answer[:100]}...")

        # Replace using re.sub (safer than string slicing)
        response = re.sub(
            r'(<FINAL_ANSWER>).*?(</FINAL_ANSWER>)',
            rf'\g<1>\n{answer}\n\g<2>',
            response,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )

    return response


# ---------------------------------------------------------------------------
# Global LLM call budget — all LLM calls (routing, audit, main loop) share this
# ---------------------------------------------------------------------------
_llm_calls_this_request: int = 0


def _reset_llm_budget() -> None:
    global _llm_calls_this_request
    _llm_calls_this_request = 0


def _consume_llm_call() -> bool:
    """Increment the global LLM call counter. Returns True if within budget."""
    global _llm_calls_this_request
    max_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    if _llm_calls_this_request >= max_calls:
        logger.warning(f"LLM budget exhausted ({_llm_calls_this_request}/{max_calls})")
        return False
    _llm_calls_this_request += 1
    return True


def _remaining_llm_budget() -> int:
    max_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    return max(0, max_calls - _llm_calls_this_request)


# ---------------------------------------------------------------------------
# Lightweight LLM routing / verification helpers
# ---------------------------------------------------------------------------
def _provider_is_available(provider: str) -> bool:
    key_map = {
        "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_AVAILABLE),
        "openai": ("OPENAI_API_KEY", OPENAI_AVAILABLE),
        "groq": ("GROQ_API_KEY", OPENAI_AVAILABLE),
        "nebius": ("NEBIUS_API_KEY", OPENAI_AVAILABLE),
        "deepinfra": ("DEEPINFRA_API_KEY", OPENAI_AVAILABLE),
    }
    env_var, sdk_ok = key_map.get(provider, ("", False))
    return sdk_ok and bool(os.environ.get(env_var))


def _classify_with_llm(prompt: str, provider: str) -> str | None:
    """Use a cheap single-shot LLM classification only when heuristics are uncertain."""
    if not _provider_is_available(provider):
        return None
    if not _consume_llm_call():
        return None

    classifier_prompt = (
        "Classify the task into exactly one label: officeqa, crm, bwim, financial, general.\n"
        "Return ONLY the label.\n\n"
        "Definitions:\n"
        "- officeqa: U.S. Treasury Bulletin / public debt / federal receipts type questions\n"
        "- crm: CRM records, leads, opportunities, pipeline, customer/account operations\n"
        "- bwim: create/transform/rewrite/format/generate content or structured output\n"
        "- financial: finance or quantitative business analysis not specific to OfficeQA\n"
        "- general: everything else\n\n"
        f"TASK:\n{prompt}"
    )

    try:
        if provider == "anthropic":
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=os.environ.get("ROUTER_MODEL", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")),
                max_tokens=8,
                system="Return only one routing label.",
                messages=[{"role": "user", "content": classifier_prompt}],
                temperature=0,
            )
            text = "\n".join(getattr(block, "text", "") for block in response.content if hasattr(block, "text"))
        else:
            base_url, default_model = PROVIDER_DEFAULTS[provider]
            key_map = {
                "openai": ("OPENAI_API_KEY", "ROUTER_MODEL", "OPENAI_MODEL"),
                "groq": ("GROQ_API_KEY", "ROUTER_MODEL", "GROQ_MODEL"),
                "nebius": ("NEBIUS_API_KEY", "ROUTER_MODEL", "NEBIUS_MODEL"),
                "deepinfra": ("DEEPINFRA_API_KEY", "ROUTER_MODEL", "DEEPINFRA_MODEL"),
            }
            api_key_var, router_var, model_var = key_map[provider]
            client = OpenAI(api_key=os.environ.get(api_key_var, ""), base_url=base_url)
            response = client.chat.completions.create(
                model=os.environ.get(router_var, os.environ.get(model_var, default_model)),
                messages=[
                    {"role": "system", "content": "Return only one routing label."},
                    {"role": "user", "content": classifier_prompt},
                ],
                temperature=0,
                max_tokens=8,
            )
            text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM classifier failed: {e}")
        return None

    label = (text or "").strip().lower()
    for candidate in SYSTEM_PROMPTS:
        if candidate in label:
            return candidate
    return None


def _should_run_numeric_audit(question: str, answer: str) -> bool:
    if not re.findall(r'-?\d[\d,]*\.?\d*', answer):
        return False
    return bool(
        re.search(
            r'calculate|difference|sum|total|ratio|percent|percentage|average|mean|median|growth|change|increase|decrease',
            question,
            re.IGNORECASE,
        )
    )


def _verify_with_llm(question: str, answer: str, provider: str) -> str:
    """Run a cheap numeric audit pass and return a corrected bare answer when needed."""
    if not _provider_is_available(provider):
        return answer
    if not _consume_llm_call():
        return answer

    verifier_prompt = (
        "Audit the proposed numeric answer for arithmetic, units, and missing symbols.\n"
        "If the answer is already correct, repeat it exactly.\n"
        "If it is wrong or missing a symbol like % or $, return the corrected bare answer only.\n"
        "Return ONLY the final bare answer, with no explanation.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PROPOSED ANSWER:\n{answer}"
    )

    try:
        if provider == "anthropic":
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=os.environ.get("VERIFIER_MODEL", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")),
                max_tokens=48,
                system="Return only the corrected bare answer.",
                messages=[{"role": "user", "content": verifier_prompt}],
                temperature=0,
            )
            text = "\n".join(getattr(block, "text", "") for block in response.content if hasattr(block, "text"))
        else:
            base_url, default_model = PROVIDER_DEFAULTS[provider]
            key_map = {
                "openai": ("OPENAI_API_KEY", "VERIFIER_MODEL", "OPENAI_MODEL"),
                "groq": ("GROQ_API_KEY", "VERIFIER_MODEL", "GROQ_MODEL"),
                "nebius": ("NEBIUS_API_KEY", "VERIFIER_MODEL", "NEBIUS_MODEL"),
                "deepinfra": ("DEEPINFRA_API_KEY", "VERIFIER_MODEL", "DEEPINFRA_MODEL"),
            }
            api_key_var, verifier_var, model_var = key_map[provider]
            client = OpenAI(api_key=os.environ.get(api_key_var, ""), base_url=base_url)
            response = client.chat.completions.create(
                model=os.environ.get(verifier_var, os.environ.get(model_var, default_model)),
                messages=[
                    {"role": "system", "content": "Return only the corrected bare answer."},
                    {"role": "user", "content": verifier_prompt},
                ],
                temperature=0,
                max_tokens=48,
            )
            text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Numeric audit failed: {e}")
        return answer

    verified = _extract_compact_final_line(_extract_bare_answer((text or "").strip()))
    return verified or answer


# ---------------------------------------------------------------------------
# LLM response with agentic tool loop
# ---------------------------------------------------------------------------
def get_llm_response(prompt: str, context_id: str = "") -> str:
    """Call the LLM with task-type-aware system prompt and tool loop."""
    _reset_llm_budget()
    logger.info(f"ENV CHECK — LLM_PROVIDER={os.environ.get('LLM_PROVIDER')}")

    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect provider from available keys
    if not provider:
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif OPENAI_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif OPENAI_AVAILABLE and os.environ.get("NEBIUS_API_KEY"):
            provider = "nebius"
        elif OPENAI_AVAILABLE and os.environ.get("DEEPINFRA_API_KEY"):
            provider = "deepinfra"

    # Detect task type and select system prompt
    task_type, routing_confidence = _heuristic_task_routing(prompt)
    if routing_confidence <= 1:
        llm_task_type = _classify_with_llm(prompt, provider)
        if llm_task_type:
            logger.info(f"LLM routing override: {task_type} -> {llm_task_type}")
            task_type = llm_task_type
    system_prompt = SYSTEM_PROMPTS[task_type]
    logger.info(f"Detected task type: {task_type} (confidence={routing_confidence})")

    # Format adapter — detect output format from task text (zero API cost)
    fmt_key, fmt_directive = _detect_output_format(prompt)
    if fmt_key:
        logger.info(f"Detected output format: {fmt_key}")
        system_prompt = system_prompt + "\n\n" + fmt_directive

    # Each OfficeQA question is independent (new_conversation=True from judge),
    # so skip history to save memory and avoid cross-contamination.

    # For OfficeQA: inject relevant Treasury source documents into the prompt
    source_context = ""
    if task_type in ("officeqa", "financial"):
        source_context = _load_source_context(prompt)
        if source_context:
            logger.info(f"Loaded {len(source_context)} chars of source context")

    if source_context:
        augmented_prompt = (
            f"REFERENCE DATA (from U.S. Treasury Bulletin — use this to find exact numbers):\n\n"
            f"{source_context}\n\n"
            f"---\n\n"
            f"QUESTION:\n{prompt}"
        )
    else:
        augmented_prompt = prompt

    messages = [{"role": "user", "content": augmented_prompt}]

    enable_tools = os.environ.get("ENABLE_TOOLS", "true").lower() == "true"

    if provider == "anthropic":
        response = _call_anthropic(messages, enable_tools, context_id, system_prompt)
    elif provider in PROVIDER_DEFAULTS:
        response = _call_openai_compatible(provider, messages, enable_tools, context_id, system_prompt)
    else:
        response = (
            "<FINAL_ANSWER>Error: No LLM provider configured. "
            "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, "
            "NEBIUS_API_KEY, or DEEPINFRA_API_KEY and set LLM_PROVIDER.</FINAL_ANSWER>"
        )

    return _clean_response(
        response,
        original_prompt=prompt,
        provider=provider,
    )


def _call_anthropic(messages: list[dict], enable_tools: bool, context_id: str, system_prompt: str) -> str:
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))
    enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"

    # Build tools list
    tools_list = []
    if enable_tools:
        tools_list.extend(TOOLS_ANTHROPIC)
    if enable_web_search:
        tools_list.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": 0,
    }
    if tools_list:
        kwargs["tools"] = tools_list

    final_text = ""
    last_error = None

    while _remaining_llm_budget() > 0:
        if not _consume_llm_call():
            break
        try:
            response = client.messages.create(**kwargs)
        except Exception as e:
            last_error = e
            logger.warning(f"Anthropic API error (calls={_llm_calls_this_request}): {e}")
            if _remaining_llm_budget() > 0:
                import time
                time.sleep(1)  # Brief retry delay
                continue
            break

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        current_text = "\n".join(text_parts)

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            final_text = current_text
            # History disabled for OfficeQA (independent questions)
            break

        # Execute tools and feed results back
        tool_results = []
        for tu in tool_uses:
            result = execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
            logger.info(f"Tool {tu.name}({tu.input}) -> {result[:200]}")

        kwargs["messages"] = kwargs["messages"] + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

    if not final_text:
        err_msg = str(last_error) if last_error else "Exceeded LLM call limit"
        final_text = f"<FINAL_ANSWER>Error: {err_msg}</FINAL_ANSWER>"

    return final_text


def _call_openai_compatible(provider: str, messages: list[dict], enable_tools: bool, context_id: str, system_prompt: str) -> str:
    """Handles OpenAI, Groq, Nebius, DeepInfra."""
    base_url, default_model = PROVIDER_DEFAULTS[provider]

    key_map = {
        "openai":    ("OPENAI_API_KEY",    "OPENAI_MODEL"),
        "groq":      ("GROQ_API_KEY",      "GROQ_MODEL"),
        "nebius":    ("NEBIUS_API_KEY",     "NEBIUS_MODEL"),
        "deepinfra": ("DEEPINFRA_API_KEY",  "DEEPINFRA_MODEL"),
    }
    api_key_var, model_var = key_map[provider]
    api_key = os.environ.get(api_key_var, "")
    model = os.environ.get(model_var, default_model)
    max_tokens = int(os.environ.get("MAX_TOKENS", "16000"))

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Only OpenAI supports function calling reliably
    supports_tools = provider == "openai" and enable_tools
    openai_tools = TOOLS_OPENAI if supports_tools else []

    base_messages = [{"role": "system", "content": system_prompt}] + messages
    current_messages = base_messages
    final_text = ""
    last_error = None

    while _remaining_llm_budget() > 0:
        if not _consume_llm_call():
            break
        kwargs: dict = {
            "model": model,
            "messages": current_messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            logger.warning(f"OpenAI API error (calls={_llm_calls_this_request}): {e}")
            if _remaining_llm_budget() > 0:
                import time
                time.sleep(1)
                continue
            break
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and supports_tools:
            tool_calls = choice.message.tool_calls or []
            current_messages = current_messages + [choice.message]
            tool_results_msgs = []
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                logger.info(f"Tool {tc.function.name}({args}) -> {result[:200]}")
                tool_results_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            current_messages = current_messages + tool_results_msgs
        else:
            final_text = choice.message.content or ""
            # History disabled for OfficeQA (independent questions)
            break

    if not final_text:
        err_msg = str(last_error) if last_error else "Exceeded LLM call limit"
        final_text = f"<FINAL_ANSWER>Error: {err_msg}</FINAL_ANSWER>"

    return final_text


def _extract_message_text(message: Message) -> str:
    """Extract all text from a Message's parts."""
    parts_text = []
    for part in message.parts:
        root = part.root if hasattr(part, "root") else part
        if isinstance(root, TextPart):
            parts_text.append(root.text)
        elif isinstance(root, DataPart):
            parts_text.append(json.dumps(root.data, indent=2))
    return "\n".join(parts_text)


# ---------------------------------------------------------------------------
# A2A Executor
# ---------------------------------------------------------------------------
class Executor(AgentExecutor):
    """Purple Agent executor — receives tasks from the judge, calls LLM, returns answers."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.message
        if not message or not message.parts:
            logger.warning("Received empty message, skipping")
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            logger.info(f"Task {task.id} already terminal, skipping")
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        # Signal we are working
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text="Analyzing task..."))],
                    ),
                ),
                final=False,
            )
        )

        prompt = _extract_message_text(message)
        logger.info(f"[{task_id}] Prompt: {prompt[:200]}...")

        try:
            response = await asyncio.to_thread(get_llm_response, prompt, context_id)
        except Exception as e:
            logger.exception(f"LLM error: {e}")
            response = f"<FINAL_ANSWER>Error processing request: {e}</FINAL_ANSWER>"

        logger.info(f"[{task_id}] Response: {response[:200]}...")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=response))],
                    ),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
