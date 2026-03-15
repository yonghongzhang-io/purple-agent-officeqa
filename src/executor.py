import asyncio
import json
import logging
import os
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
    # provider name  → (base_url,                          default model)
    "openai":        (None,                                 "gpt-4o"),
    "groq":          ("https://api.groq.com/openai/v1",    "llama-3.3-70b-versatile"),
    "nebius":        ("https://api.studio.nebius.ai/v1",   "meta-llama/Meta-Llama-3.1-70B-Instruct"),
    "deepinfra":     ("https://api.deepinfra.com/v1/openai","meta-llama/Llama-3.3-70B-Instruct-Turbo"),
}


# ---------------------------------------------------------------------------
# System prompt — tuned for BWIM (natural-language task completion) and
# CRMArena (CRM data manipulation tasks).
# Keep reasoning explicit so the judge can follow your logic.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI agent competing in the AgentX-AgentBeats competition.
You excel at:
1. Following natural-language instructions precisely (Build What I Mean tasks)
2. Manipulating and reasoning about CRM data (CRMArena tasks)
3. Writing, editing, and transforming structured data (JSON, CSV, tables)
4. Step-by-step problem solving with high accuracy

RESPONSE FORMAT — always use this structure:
<REASONING>
[Think step by step. Show your work clearly.]
</REASONING>
<FINAL_ANSWER>
[Your complete, accurate final answer here. No hedging, no multiple options — one definitive answer.]
</FINAL_ANSWER>

CRITICAL RULES:
- Always produce a <FINAL_ANSWER> tag or your response counts as wrong.
- Be precise: numbers, names, and formats must match exactly what is asked.
- For CRM tasks: output valid JSON when structured data is requested.
- For instruction-following tasks: do exactly what is asked, nothing more or less.
- If a task is ambiguous, pick the most reasonable interpretation and state it briefly in REASONING.
"""


# ---------------------------------------------------------------------------
# Tool definitions (used when tools are enabled)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Input must be a valid Python arithmetic expression string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A Python arithmetic expression, e.g. '(100 * 1.05) / 3'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "format_json",
        "description": "Parse and pretty-print a JSON string. Useful for validating and formatting CRM data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "json_string": {"type": "string", "description": "A JSON string to parse and format"}
            },
            "required": ["json_string"]
        }
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a local tool and return the result as a string."""
    try:
        if tool_name == "calculate":
            expr = tool_input.get("expression", "")
            # Safe eval: only allow arithmetic
            allowed = set("0123456789+-*/().,% eE")
            if not all(c in allowed for c in expr.replace(" ", "")):
                return f"Error: unsafe expression: {expr}"
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return str(result)

        if tool_name == "format_json":
            raw = tool_input.get("json_string", "")
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2)

    except Exception as e:
        return f"Tool error: {e}"

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Conversation history per context_id (supports multi-turn)
# ---------------------------------------------------------------------------
_conversation_history: dict[str, list[dict]] = {}


def _get_history(context_id: str) -> list[dict]:
    return _conversation_history.setdefault(context_id, [])


def _append_history(context_id: str, role: str, content: str) -> None:
    _conversation_history.setdefault(context_id, []).append(
        {"role": role, "content": content}
    )


# ---------------------------------------------------------------------------
# LLM response with agentic tool loop
# ---------------------------------------------------------------------------
def get_llm_response(prompt: str, context_id: str = "") -> str:
    logger.info(f"ENV CHECK — LLM_PROVIDER={os.environ.get('LLM_PROVIDER')} GROQ_KEY={'SET' if os.environ.get('GROQ_API_KEY') else 'MISSING'}")
    """
    Call the LLM with full conversation history and an optional tool-use loop.
    Supports: anthropic, openai, groq, nebius, deepinfra.
    Returns the final text response (including <FINAL_ANSWER> tags).
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect provider from available keys if not explicitly set
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

    # Build message history
    history = _get_history(context_id)
    messages = history + [{"role": "user", "content": prompt}]
    _append_history(context_id, "user", prompt)

    enable_tools = os.environ.get("ENABLE_TOOLS", "true").lower() == "true"

    if provider == "anthropic":
        return _call_anthropic(messages, enable_tools, context_id)

    if provider in PROVIDER_DEFAULTS:
        return _call_openai_compatible(provider, messages, enable_tools, context_id)

    return (
        "<FINAL_ANSWER>Error: No LLM provider configured. "
        "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, "
        "NEBIUS_API_KEY, or DEEPINFRA_API_KEY and set LLM_PROVIDER.</FINAL_ANSWER>"
    )


def _call_anthropic(messages: list[dict], enable_tools: bool, context_id: str) -> str:
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8000"))

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "temperature": 0,
    }
    if enable_tools:
        kwargs["tools"] = TOOLS

    # Agentic tool-use loop (max 4 LLM calls to stay within competition limit)
    max_llm_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    calls_made = 0
    final_text = ""

    while calls_made < max_llm_calls:
        response = client.messages.create(**kwargs)
        calls_made += 1

        # Collect text blocks
        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        current_text = "\n".join(text_parts)

        # Check for tool use
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses or not enable_tools:
            # No more tool calls — this is the final answer
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
            logger.info(f"Tool {tu.name}({tu.input}) → {result}")

        # Append assistant turn + tool results to conversation
        kwargs["messages"] = kwargs["messages"] + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

    if not final_text:
        final_text = "<FINAL_ANSWER>Error: Exceeded LLM call limit without producing an answer.</FINAL_ANSWER>"

    return final_text


def _call_openai_compatible(provider: str, messages: list[dict], enable_tools: bool, context_id: str) -> str:
    """Handles OpenAI, Groq, Nebius, DeepInfra — all use the same OpenAI client."""
    base_url, default_model = PROVIDER_DEFAULTS[provider]

    # Resolve API key and model from env
    key_map = {
        "openai":    ("OPENAI_API_KEY",    "OPENAI_MODEL"),
        "groq":      ("GROQ_API_KEY",      "GROQ_MODEL"),
        "nebius":    ("NEBIUS_API_KEY",     "NEBIUS_MODEL"),
        "deepinfra": ("DEEPINFRA_API_KEY",  "DEEPINFRA_MODEL"),
    }
    api_key_var, model_var = key_map[provider]
    api_key = os.environ.get(api_key_var, "")
    model = os.environ.get(model_var, default_model)
    max_tokens = int(os.environ.get("MAX_TOKENS", "8000"))

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Groq/Nebius/DeepInfra don't support function calling well — disable tools
    supports_tools = provider == "openai" and enable_tools
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        }
        for t in TOOLS
    ] if supports_tools else []

    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    current_messages = base_messages
    max_llm_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    calls_made = 0
    final_text = ""

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

        response = client.chat.completions.create(**kwargs)
        calls_made += 1
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and supports_tools:
            tool_calls = choice.message.tool_calls or []
            current_messages = current_messages + [choice.message]
            tool_results_msgs = []
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                logger.info(f"Tool {tc.function.name}({args}) → {result}")
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
        final_text = "<FINAL_ANSWER>Error: Exceeded LLM call limit without producing an answer.</FINAL_ANSWER>"

    return final_text


def _extract_message_text(message: Message) -> str:
    """Extract all text from a Message's parts (handles TextPart and DataPart)."""
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
