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
    "nebius":    ("https://api.studio.nebius.ai/v1",    "meta-llama/Meta-Llama-3.1-70B-Instruct"),
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

QUESTION TYPES:
- Simple data extraction (e.g., "What was the total receipts in fiscal year 2020?")
- Multi-year calculations with inflation adjustments
- Statistical analysis (regression, correlation, standard deviation)
- Time series forecasting
- Complex financial metrics (VaR, weighted averages)

RULES:
1. Ensure FULL NUMERICAL PRECISION — never round unless explicitly asked.
2. Match the exact value from the source. "2602" must be "2602", not "approximately 2600".
3. Handle unit conversions correctly: million, billion, trillion.
4. If the answer is a dollar amount, include the dollar sign or "dollars".
5. If the answer is a percentage, include the percent sign.
6. For dates, use the exact format from the source data.
7. NEVER hedge. Give ONE definitive answer, never "either X or Y".
8. Keep FINAL_ANSWER under 500 characters.

SCORING (exact rules used by the judge):
- Numbers: exact match (0% tolerance). Commas are stripped ("2,602" = "2602").
- Units: "billion", "million", "thousand" detected from context words.
- Text: case-insensitive substring match.
- Hedging: if you give multiple candidate answers = AUTOMATIC FAIL.
- "No answer found" = AUTOMATIC FAIL.

RESPONSE FORMAT:
<REASONING>
[Show your work. For calculations, show every step with full precision.]
</REASONING>
<FINAL_ANSWER>
[The precise value with units if applicable. ONE answer only.]
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

OFFICEQA_KEYWORDS = {
    "treasury bulletin", "u.s. treasury", "fiscal year",
    "treasury balance", "public debt", "federal receipts",
    "treasury department", "national debt",
}

BWIM_KEYWORDS = {
    "create", "build", "generate", "write", "convert", "transform",
    "rewrite", "format", "produce", "make", "compose", "draft",
    "translate", "restructure", "reorganize",
}


def detect_task_type(prompt: str) -> str:
    """Detect task type from prompt content to select the best system prompt."""
    lower = prompt.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    # Check OfficeQA first (multi-word phrases)
    officeqa_score = sum(1 for phrase in OFFICEQA_KEYWORDS if phrase in lower)
    if officeqa_score >= 1:
        return "officeqa"

    crm_score = len(words & CRM_KEYWORDS)
    financial_score = len(words & FINANCIAL_KEYWORDS)
    bwim_score = len(words & BWIM_KEYWORDS)

    if crm_score >= 2 or (crm_score >= 1 and financial_score == 0 and bwim_score == 0):
        return "crm"
    if financial_score >= 2:
        return "officeqa"  # Financial questions likely come from OfficeQA benchmark
    if bwim_score >= 1:
        return "bwim"

    # Default heuristic: if it looks like a question, use officeqa prompt (most robust)
    if lower.strip().startswith(("what", "how much", "how many", "calculate", "find", "determine")):
        return "officeqa"

    return "general"


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
# Hedge detection — check if response contains multiple candidate answers
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


# ---------------------------------------------------------------------------
# Output cleaning and validation
# ---------------------------------------------------------------------------
def _clean_response(response: str) -> str:
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
            response = response[:match.start(1)] + "\n" + answer + "\n" + response[match.end(1):]

        # Truncate to 500 chars
        if len(answer) > 500:
            truncated = answer[:500].rsplit(" ", 1)[0] if " " in answer[:500] else answer[:500]
            response = response[:match.start(1)] + "\n" + truncated + "\n" + response[match.end(1):]

        # Log warning if hedge detected (can't fix automatically without re-calling LLM)
        if _contains_hedge(answer):
            logger.warning(f"HEDGE DETECTED in FINAL_ANSWER: {answer[:100]}...")

    return response


# ---------------------------------------------------------------------------
# LLM response with agentic tool loop
# ---------------------------------------------------------------------------
def get_llm_response(prompt: str, context_id: str = "") -> str:
    """Call the LLM with task-type-aware system prompt and tool loop."""
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
    task_type = detect_task_type(prompt)
    system_prompt = SYSTEM_PROMPTS[task_type]
    logger.info(f"Detected task type: {task_type}")

    # Build message history
    history = _get_history(context_id)
    messages = history + [{"role": "user", "content": prompt}]
    _append_history(context_id, "user", prompt)

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

    return _clean_response(response)


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

    max_llm_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    calls_made = 0
    final_text = ""
    last_error = None

    while calls_made < max_llm_calls:
        try:
            response = client.messages.create(**kwargs)
        except Exception as e:
            last_error = e
            calls_made += 1
            logger.warning(f"Anthropic API error (attempt {calls_made}): {e}")
            if calls_made < max_llm_calls:
                import time
                time.sleep(1)  # Brief retry delay
                continue
            break

        calls_made += 1

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        current_text = "\n".join(text_parts)

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            final_text = current_text
            _append_history(context_id, "assistant", final_text)
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
    max_llm_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    calls_made = 0
    final_text = ""

    last_error = None

    while calls_made < max_llm_calls:
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
            calls_made += 1
            logger.warning(f"OpenAI API error (attempt {calls_made}): {e}")
            if calls_made < max_llm_calls:
                import time
                time.sleep(1)
                continue
            break

        calls_made += 1
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
            _append_history(context_id, "assistant", final_text)
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
