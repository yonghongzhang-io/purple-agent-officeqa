"""Purple Agent — core agent logic following the template pattern.

Handles OfficeQA task routing, LLM orchestration, tool execution,
and answer post-processing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile

import providers
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart, DataPart
from a2a.utils import new_agent_text_message

from messenger import Messenger
from postprocess import (
    _clean_response,
    _detect_output_format,
    _extract_final_answer_text,
    _vote_answers,
)
from retrieval import _load_source_context, build_officeqa_strategy


logger = logging.getLogger(__name__)


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

SYSTEM_PROMPT_OFFICEQA = """You answer U.S. Treasury Bulletin questions using ONLY the provided reference data.

RULES:
1. Find the exact number in the reference data. Do NOT guess.
2. FULL NUMERICAL PRECISION — never round unless asked.
3. NEVER say "not found", "unable to determine", "I need to verify", or "no data". If uncertain, give your best estimate from the data.
4. NEVER hedge or give multiple answers. ONE answer only.
5. Keep FINAL_ANSWER under 500 characters.
6. If the answer is numeric, prefer a bare number with no prose. Use symbols like % only when the question clearly asks for a percentage.
7. If the answer is a date or short text label, return only that date or label.
8. Read pipe-delimited tables carefully: match the column header to the row label to find the exact cell value.
9. Output ONLY the final answer tags — do NOT continue reasoning after <FINAL_ANSWER>.

OUTPUT FORMAT — you MUST use this exact format:
<REASONING>
[Brief work shown here — keep under 500 words]
</REASONING>
<FINAL_ANSWER>
[ONLY the bare value — a number, date, or short phrase. NO sentences. NO explanation.]
</FINAL_ANSWER>

EXAMPLES:
<FINAL_ANSWER>2602</FINAL_ANSWER>
<FINAL_ANSWER>47.3%</FINAL_ANSWER>
<FINAL_ANSWER>March 1977</FINAL_ANSWER>

WRONG (will score 0):
<FINAL_ANSWER>The total was approximately 2602 million</FINAL_ANSWER>
<FINAL_ANSWER>Based on my analysis, 2602</FINAL_ANSWER>
<FINAL_ANSWER>Not found in the provided reference data</FINAL_ANSWER>"""

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


SYSTEM_PROMPTS = {
    "bwim": SYSTEM_PROMPT_BWIM,
    "crm": SYSTEM_PROMPT_CRM,
    "financial": SYSTEM_PROMPT_FINANCIAL,
    "officeqa": SYSTEM_PROMPT_OFFICEQA,
    "general": SYSTEM_PROMPT_GENERAL,
}


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

OFFICEQA_STRONG_PHRASES = {
    "treasury bulletin", "u.s. treasury bulletin",
    "treasury department bulletin",
}
OFFICEQA_CONTEXT_PHRASES = {
    "public debt outstanding", "federal receipts", "treasury balance",
    "national debt", "u.s. treasury",
}
OFFICEQA_WEAK_SIGNALS = {
    "fiscal year", "treasury department",
}

BWIM_KEYWORDS = {
    "create", "build", "generate", "write", "convert", "transform",
    "rewrite", "format", "produce", "make", "compose", "draft",
    "translate", "restructure", "reorganize",
}


def detect_task_type(prompt: str) -> str:
    lower = prompt.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    strong_hits = sum(1 for phrase in OFFICEQA_STRONG_PHRASES if phrase in lower)
    if strong_hits >= 1:
        return "officeqa"

    context_hits = sum(1 for phrase in OFFICEQA_CONTEXT_PHRASES if phrase in lower)
    if context_hits >= 2:
        return "officeqa"

    weak_hits = sum(1 for phrase in OFFICEQA_WEAK_SIGNALS if phrase in lower)
    fiscal_words = words & {"expenditures", "receipts", "appropriation", "disbursements"}
    if weak_hits >= 1 and len(fiscal_words) >= 1:
        return "officeqa"

    crm_score = len(words & CRM_KEYWORDS)
    if crm_score >= 2:
        return "crm"

    financial_score = len(words & FINANCIAL_KEYWORDS)
    bwim_score = len(words & BWIM_KEYWORDS)
    if bwim_score >= 2 or (bwim_score >= 1 and financial_score == 0):
        return "bwim"

    if financial_score >= 2:
        return "financial"

    return "general"


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic + OpenAI format)
# ---------------------------------------------------------------------------
TOOLS_ANTHROPIC = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code for precise calculations and table parsing. "
            "Use print() to emit the final result. "
            "Available imports: math, decimal, statistics, re, csv, io. "
            "Use this tool to parse pipe-delimited tables and extract exact values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() for the final output."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports arithmetic, power (**), modulo (%), and common math.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression, e.g. '(100 * 1.05) / 3'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "format_json",
        "description": "Parse, validate, and pretty-print a JSON string.",
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
        "description": "Search and filter a JSON array of objects by field values.",
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
                    "description": "The value to match (case-insensitive substring)"
                }
            },
            "required": ["data", "field", "value"]
        }
    },
    {
        "name": "aggregate",
        "description": "Aggregate a numeric field from a JSON array. Supports sum, avg, min, max, count.",
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
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
    }
    for tool in TOOLS_ANTHROPIC
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "run_python":
            code = tool_input.get("code", "")
            full_code = (
                "import math\n"
                "import re\n"
                "import csv\n"
                "import io\n"
                "from decimal import Decimal, ROUND_HALF_UP\n"
                "import statistics\n\n"
                + code
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
                handle.write(full_code)
                tmp_path = handle.name
            try:
                timeout = int(os.environ.get("CODE_TIMEOUT", "30"))
                result = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                if result.returncode != 0:
                    return f"Tool error: {stderr[:500]}"
                return stdout if stdout else "(no output)"
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if tool_name == "calculate":
            import math

            expr = tool_input.get("expression", "")
            safe_globals = {"__builtins__": {}}
            safe_locals = {
                "round": round,
                "abs": abs,
                "min": min,
                "max": max,
                "pow": pow,
                "sum": sum,
                "len": len,
                "int": int,
                "float": float,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "pi": math.pi,
                "e": math.e,
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
                row for row in data
                if field in row and value in str(row[field]).lower()
            ]
            return json.dumps(results, indent=2)

        if tool_name == "aggregate":
            data = json.loads(tool_input.get("data", "[]"))
            field = tool_input.get("field", "")
            operation = tool_input.get("operation", "sum")
            values = [
                float(row[field]) for row in data
                if field in row and row[field] is not None
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
    except Exception as exc:
        return f"Tool error: {exc}"

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# LLM call wrappers
# ---------------------------------------------------------------------------
def _call_anthropic(
    messages: list[dict],
    enable_tools: bool,
    context_id: str,
    system_prompt: str,
    temperature: float = 0,
) -> str:
    return providers._call_anthropic(
        messages,
        enable_tools,
        context_id,
        system_prompt,
        temperature,
        tools_anthropic=TOOLS_ANTHROPIC,
        execute_tool=execute_tool,
    )


def _call_openai_compatible(
    provider: str,
    messages: list[dict],
    enable_tools: bool,
    context_id: str,
    system_prompt: str,
    temperature: float = 0,
) -> str:
    return providers._call_openai_compatible(
        provider,
        messages,
        enable_tools,
        context_id,
        system_prompt,
        temperature,
        tools_openai=TOOLS_OPENAI,
        execute_tool=execute_tool,
    )


# ---------------------------------------------------------------------------
# Core LLM response logic
# ---------------------------------------------------------------------------
def get_llm_response(prompt: str, context_id: str = "") -> str:
    logger.info(f"ENV CHECK — LLM_PROVIDER={os.environ.get('LLM_PROVIDER')}")

    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if not provider:
        if providers.ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif providers.OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif providers.OPENAI_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif providers.OPENAI_AVAILABLE and os.environ.get("NEBIUS_API_KEY"):
            provider = "nebius"
        elif providers.OPENAI_AVAILABLE and os.environ.get("KIMI_API_KEY"):
            provider = "kimi"
        elif providers.OPENAI_AVAILABLE and os.environ.get("DEEPINFRA_API_KEY"):
            provider = "deepinfra"

    # Current competition mode: OfficeQA-only submission agent.
    task_type = "officeqa"

    source_context = _load_source_context(prompt)
    if source_context:
        logger.info(f"OfficeQA source context loaded ({len(source_context)} chars)")
    else:
        logger.warning("No Treasury source context found for this question")
    strategy_hint = build_officeqa_strategy(prompt)

    system_prompt = SYSTEM_PROMPTS[task_type]

    fmt_key, fmt_directive = _detect_output_format(prompt, task_type=task_type)
    if fmt_key:
        logger.info(f"Detected output format: {fmt_key}")
        system_prompt = system_prompt + "\n\n" + fmt_directive

    if source_context:
        augmented_prompt = (
            "REFERENCE DATA (from U.S. Treasury Bulletin — use this to find exact numbers):\n\n"
            f"{source_context}\n\n"
            "---\n\n"
            "QUESTION-SHAPE SOLVER INSTRUCTIONS:\n"
            f"{strategy_hint}\n\n"
            "---\n\n"
            f"QUESTION:\n{prompt}"
        )
    else:
        augmented_prompt = (
            "QUESTION-SHAPE SOLVER INSTRUCTIONS:\n"
            f"{strategy_hint}\n\n"
            "---\n\n"
            f"QUESTION:\n{prompt}"
        )

    # ------------------------------------------------------------------
    # TWO-PASS STRATEGY: Read → Extract → Answer
    # Pass 1: LLM reads the full source and extracts the exact table/rows needed
    # Pass 2: LLM answers precisely from the extracted snippet
    # This mimics how a human expert would solve it: skim the full doc first,
    # then focus on the relevant section.
    # ------------------------------------------------------------------
    two_pass = os.environ.get("TWO_PASS", "true").lower() == "true"

    if two_pass and source_context and len(source_context) > 5000:
        extract_prompt = (
            "You are a Treasury Bulletin data extraction expert.\n\n"
            "REFERENCE DATA:\n\n"
            f"{source_context}\n\n"
            "---\n\n"
            f"QUESTION: {prompt}\n\n"
            "---\n\n"
            "TASK: Extract ONLY the specific table rows, column headers, and numeric values "
            "needed to answer this question. Include the table title, units (millions/billions), "
            "and any footnotes. Do NOT answer the question yet — just extract the relevant data.\n\n"
            "Format your extraction as:\n"
            "TABLE TITLE: ...\n"
            "UNITS: ...\n"
            "RELEVANT ROWS:\n"
            "- [row label]: [value]\n"
            "- [row label]: [value]\n"
            "KEY VALUES FOR CALCULATION: ..."
        )
        try:
            extraction = providers._call_openai_compatible(
                provider, [{"role": "user", "content": extract_prompt}],
                False, context_id, "You extract precise data from Treasury Bulletins.",
                0, tools_openai=None, execute_tool=None,
            )
            logger.info(f"Pass 1 extraction: {len(extraction)} chars")
            # Pass 2: answer from the focused extraction
            augmented_prompt = (
                "EXTRACTED DATA (from U.S. Treasury Bulletin — verified extraction):\n\n"
                f"{extraction}\n\n"
                "---\n\n"
                "ORIGINAL REFERENCE DATA (for cross-checking):\n\n"
                f"{source_context[:15000]}\n\n"
                "---\n\n"
                "QUESTION-SHAPE SOLVER INSTRUCTIONS:\n"
                f"{strategy_hint}\n\n"
                "---\n\n"
                f"QUESTION:\n{prompt}\n\n"
                "IMPORTANT: Use the EXTRACTED DATA above as your primary source. "
                "Cross-check against the ORIGINAL REFERENCE DATA. "
                "Show your calculation step by step."
            )
        except Exception as e:
            logger.warning(f"Two-pass extraction failed, falling back to single pass: {e}")

    messages = [{"role": "user", "content": augmented_prompt}]

    enable_tools = os.environ.get("ENABLE_TOOLS", "false").lower() == "true"
    officeqa_use_tools = os.environ.get("OFFICEQA_USE_TOOLS", "false").lower() == "true"
    if task_type == "officeqa":
        enable_tools = enable_tools and officeqa_use_tools

    num_rollouts = max(1, int(os.environ.get("NUM_ROLLOUTS", "1")))
    vote_temperature = float(os.environ.get("VOTE_TEMPERATURE", "0.2"))

    raw_responses: list[str] = []
    voted_answers: list[str] = []

    for idx in range(num_rollouts):
        providers._reset_llm_budget()
        rollout_temperature = 0 if idx == 0 else vote_temperature
        if provider == "anthropic":
            response = _call_anthropic(
                messages,
                enable_tools,
                context_id,
                system_prompt,
                rollout_temperature,
            )
        elif provider in providers.PROVIDER_DEFAULTS:
            response = _call_openai_compatible(
                provider,
                messages,
                enable_tools,
                context_id,
                system_prompt,
                rollout_temperature,
            )
        else:
            response = (
                "<FINAL_ANSWER>Error: No LLM provider configured. "
                "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, "
                "NEBIUS_API_KEY, KIMI_API_KEY, or DEEPINFRA_API_KEY and set LLM_PROVIDER.</FINAL_ANSWER>"
            )

        cleaned = _clean_response(
            response,
            original_prompt=prompt,
            provider=provider,
            task_type=task_type,
            verify_with_llm=providers._verify_with_llm,
            should_run_numeric_audit=providers._should_run_numeric_audit,
        )
        raw_responses.append(cleaned)
        voted_answers.append(_extract_final_answer_text(cleaned))

    if num_rollouts == 1:
        return raw_responses[0]

    voted = _vote_answers(voted_answers)
    ensemble_response = (
        "<REASONING>\n"
        f"Selected the most stable answer across {num_rollouts} rollouts.\n"
        "</REASONING>\n"
        "<FINAL_ANSWER>\n"
        f"{voted}\n"
        "</FINAL_ANSWER>"
    )
    return _clean_response(
        ensemble_response,
        original_prompt=prompt,
        provider=provider,
        task_type=task_type,
        verify_with_llm=providers._verify_with_llm,
        should_run_numeric_audit=providers._should_run_numeric_audit,
    )


# ---------------------------------------------------------------------------
# Helper to extract text from A2A Message
# ---------------------------------------------------------------------------
def _extract_message_text(message: Message) -> str:
    parts_text = []
    for part in message.parts:
        root = part.root if hasattr(part, "root") else part
        if isinstance(root, TextPart):
            parts_text.append(root.text)
        elif isinstance(root, DataPart):
            parts_text.append(json.dumps(root.data, indent=2))
    return "\n".join(parts_text)


# ---------------------------------------------------------------------------
# Agent class — template pattern
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement agent logic: extract question, call LLM, return answer."""
        prompt = _extract_message_text(message)
        logger.info(f"Agent processing: {prompt[:200]}...")

        await updater.update_status(
            TaskState.working, new_agent_text_message("Analyzing task...")
        )

        try:
            response = await asyncio.to_thread(get_llm_response, prompt, "")
        except Exception as exc:
            logger.exception(f"LLM error: {exc}")
            response = f"<FINAL_ANSWER>Error processing request: {exc}</FINAL_ANSWER>"

        logger.info(f"Agent response: {response[:200]}...")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response))],
            name="answer",
        )
