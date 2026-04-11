from __future__ import annotations

import contextvars

import json
import logging
import os
import re
import threading
import time

from postprocess import _extract_bare_answer, _extract_compact_final_line


logger = logging.getLogger(__name__)


try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


PROVIDER_DEFAULTS = {
    "openai": (None, "gpt-4o"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "nebius": ("https://api.studio.nebius.ai/v1", "meta-llama/Llama-3.3-70B-Instruct"),
    "kimi": ("https://api.moonshot.cn/v1", "kimi-k2.5"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
}


_budget_var: contextvars.ContextVar[int] = contextvars.ContextVar("llm_budget", default=0)
_semaphore_cache: dict[int, threading.BoundedSemaphore] = {}

# ---------------------------------------------------------------------------
# Round-robin API key rotation (Kimi rate-limit avoidance)
# ---------------------------------------------------------------------------
_kimi_keys: list[str] = []
_kimi_key_index = 0
_kimi_key_lock = threading.Lock()


def _init_kimi_keys() -> list[str]:
    """Build round-robin key pool from KIMI_RESEARCH_KEY_01..20 + primary key."""
    global _kimi_keys
    if _kimi_keys:
        return _kimi_keys
    keys: list[str] = []
    for i in range(1, 20):
        k = os.environ.get(f"KIMI_RESEARCH_KEY_{i:02d}", "")
        if k:
            keys.append(k)
    if not keys:
        primary = os.environ.get("KIMI_API_KEY", "")
        if primary:
            keys.append(primary)
    _kimi_keys = keys
    if len(keys) > 1:
        logger.info("Kimi round-robin initialized with %d API keys", len(keys))
    return keys


def _get_next_kimi_key() -> str:
    """Return the next Kimi API key in round-robin order."""
    global _kimi_key_index
    keys = _init_kimi_keys()
    if not keys:
        return os.environ.get("KIMI_API_KEY", "")
    with _kimi_key_lock:
        key = keys[_kimi_key_index % len(keys)]
        _kimi_key_index += 1
    return key


def _reset_llm_budget() -> None:
    _budget_var.set(0)


def _consume_llm_call() -> bool:
    max_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    current = _budget_var.get()
    if current >= max_calls:
        logger.warning(f"LLM budget exhausted ({current}/{max_calls})")
        return False
    _budget_var.set(current + 1)
    return True


def _remaining_llm_budget() -> int:
    max_calls = int(os.environ.get("MAX_LLM_CALLS", "4"))
    return max(0, max_calls - _budget_var.get())


def _get_api_semaphore() -> threading.BoundedSemaphore:
    max_concurrency = max(1, int(os.environ.get("LLM_MAX_CONCURRENCY", "2")))
    if max_concurrency not in _semaphore_cache:
        _semaphore_cache[max_concurrency] = threading.BoundedSemaphore(max_concurrency)
    return _semaphore_cache[max_concurrency]


def _resolve_temperature(provider: str, model: str, requested: float, *, thinking: bool = True) -> float:
    """Normalize temperature for providers/models with strict requirements.

    Kimi K2.5 rules:
      - thinking enabled  → temperature must be 1.0
      - thinking disabled → temperature must be 0.6
    """
    normalized = float(requested)
    if provider == "kimi" and "k2" in model.lower():
        return 1.0 if thinking else 0.6
    # Older kimi models accept the usual [0, 1] range.
    if provider == "kimi" and model.startswith("kimi-"):
        return max(0.0, min(1.0, normalized))
    return normalized


def _provider_is_available(provider: str) -> bool:
    key_map = {
        "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_AVAILABLE),
        "openai": ("OPENAI_API_KEY", OPENAI_AVAILABLE),
        "groq": ("GROQ_API_KEY", OPENAI_AVAILABLE),
        "nebius": ("NEBIUS_API_KEY", OPENAI_AVAILABLE),
        "kimi": ("KIMI_API_KEY", OPENAI_AVAILABLE),
        "deepinfra": ("DEEPINFRA_API_KEY", OPENAI_AVAILABLE),
    }
    env_var, sdk_ok = key_map.get(provider, ("", False))
    if not sdk_ok:
        return False
    # Kimi: also check round-robin keys
    if provider == "kimi":
        return bool(os.environ.get(env_var)) or bool(_init_kimi_keys())
    return bool(os.environ.get(env_var))


def _retry_delay(retry_count: int, provider: str) -> float:
    base = float(os.environ.get("API_RETRY_BASE_DELAY", "1.0"))
    if provider == "anthropic":
        base = max(base, 2.0)
    return min(8.0, base * retry_count)


def _classify_with_llm(prompt: str, provider: str, valid_labels: list[str] | tuple[str, ...]) -> str | None:
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
                "kimi": ("KIMI_API_KEY", "ROUTER_MODEL", "KIMI_MODEL"),
                "deepinfra": ("DEEPINFRA_API_KEY", "ROUTER_MODEL", "DEEPINFRA_MODEL"),
            }
            api_key_var, router_var, model_var = key_map[provider]
            api_key = _get_next_kimi_key() if provider == "kimi" else os.environ.get(api_key_var, "")
            client = OpenAI(api_key=api_key, base_url=base_url)
            model = os.environ.get(router_var, os.environ.get(model_var, default_model))
            is_k2 = provider == "kimi" and "k2" in model.lower()
            clf_kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only one routing label."},
                    {"role": "user", "content": classifier_prompt},
                ],
                "temperature": _resolve_temperature(provider, model, 0, thinking=False),
                "max_tokens": 8,
            }
            if is_k2:
                clf_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(**clf_kwargs)
            text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM classifier failed: {e}")
        return None

    label = (text or "").strip().lower()
    for candidate in valid_labels:
        if candidate in label:
            return candidate
    return None


def _should_run_numeric_audit(question: str, answer: str) -> bool:
    if not any(char.isdigit() for char in answer):
        return False
    return bool(
        re.search(
            r"calculate|difference|sum|total|ratio|percent|percentage|average|mean|median|growth|change|increase|decrease",
            question,
            re.IGNORECASE,
        )
    )


def _verify_with_llm(question: str, answer: str, provider: str) -> str:
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
                "kimi": ("KIMI_API_KEY", "VERIFIER_MODEL", "KIMI_MODEL"),
                "deepinfra": ("DEEPINFRA_API_KEY", "VERIFIER_MODEL", "DEEPINFRA_MODEL"),
            }
            api_key_var, verifier_var, model_var = key_map[provider]
            api_key = _get_next_kimi_key() if provider == "kimi" else os.environ.get(api_key_var, "")
            client = OpenAI(api_key=api_key, base_url=base_url)
            model = os.environ.get(verifier_var, os.environ.get(model_var, default_model))
            is_k2 = provider == "kimi" and "k2" in model.lower()
            ver_kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only the corrected bare answer."},
                    {"role": "user", "content": verifier_prompt},
                ],
                "temperature": _resolve_temperature(provider, model, 0, thinking=False),
                "max_tokens": 48,
            }
            if is_k2:
                ver_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(**ver_kwargs)
            text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Numeric audit failed: {e}")
        return answer

    verified = _extract_compact_final_line(_extract_bare_answer((text or "").strip()))
    return verified or answer


def _call_anthropic(
    messages: list[dict],
    enable_tools: bool,
    context_id: str,
    system_prompt: str,
    temperature: float = 0,
    *,
    tools_anthropic: list[dict],
    execute_tool,
) -> str:
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))
    enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"

    tools_list = []
    if enable_tools:
        tools_list.extend(tools_anthropic)
    if enable_web_search:
        tools_list.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
    }
    if tools_list:
        kwargs["tools"] = tools_list

    final_text = ""
    last_error = None
    max_retries = int(os.environ.get("MAX_API_RETRIES", "2"))
    retry_count = 0

    while _remaining_llm_budget() > 0:
        try:
            with _get_api_semaphore():
                response = client.messages.create(**kwargs)
        except Exception as e:
            last_error = e
            retry_count += 1
            logger.warning(f"Anthropic API error (budget={_budget_var.get()}): {e}")
            if retry_count <= max_retries and _remaining_llm_budget() > 0:
                time.sleep(_retry_delay(retry_count, "anthropic"))
                continue
            break
        retry_count = 0
        if not _consume_llm_call():
            break

        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        current_text = "\n".join(text_parts)
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            final_text = current_text
            break

        tool_results = []
        for tool_use in tool_uses:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })
            logger.info(f"Tool {tool_use.name}({tool_use.input}) -> {result[:200]}")

        kwargs["messages"] = kwargs["messages"] + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

    if not final_text:
        err_msg = str(last_error) if last_error else "Exceeded LLM call limit"
        final_text = f"<FINAL_ANSWER>Error: {err_msg}</FINAL_ANSWER>"

    return final_text


def _call_openai_compatible(
    provider: str,
    messages: list[dict],
    enable_tools: bool,
    context_id: str,
    system_prompt: str,
    temperature: float = 0,
    *,
    tools_openai: list[dict],
    execute_tool,
) -> str:
    base_url, default_model = PROVIDER_DEFAULTS[provider]

    key_map = {
        "openai": ("OPENAI_API_KEY", "OPENAI_MODEL"),
        "groq": ("GROQ_API_KEY", "GROQ_MODEL"),
        "nebius": ("NEBIUS_API_KEY", "NEBIUS_MODEL"),
        "kimi": ("KIMI_API_KEY", "KIMI_MODEL"),
        "deepinfra": ("DEEPINFRA_API_KEY", "DEEPINFRA_MODEL"),
    }
    api_key_var, model_var = key_map[provider]
    model = os.environ.get(model_var, default_model)
    max_tokens = int(os.environ.get("MAX_TOKENS", "6000"))

    # Round-robin key for Kimi, plain env var for others
    if provider == "kimi":
        api_key = _get_next_kimi_key()
    else:
        api_key = os.environ.get(api_key_var, "")

    client = OpenAI(api_key=api_key, base_url=base_url)
    is_k2 = provider == "kimi" and "k2" in model.lower()

    # Kimi K2.5 supports tool calls via OpenAI-compatible API
    supports_tools = (provider in ("openai", "kimi")) and enable_tools
    openai_tools = tools_openai if supports_tools else []

    base_messages = [{"role": "system", "content": system_prompt}] + messages
    current_messages = base_messages
    final_text = ""
    last_error = None
    max_retries = int(os.environ.get("MAX_API_RETRIES", "2"))
    retry_count = 0

    # Kimi K2.5 thinking mode: configurable via KIMI_THINKING env var
    # Default disabled — our system prompt already has <REASONING> tags for CoT
    kimi_thinking_enabled = os.environ.get("KIMI_THINKING", "false").lower() == "true"

    # Track whether thinking was disabled for tool follow-ups
    _thinking_disabled_for_tools = False

    while _remaining_llm_budget() > 0:
        thinking = is_k2 and kimi_thinking_enabled and not _thinking_disabled_for_tools
        kwargs: dict = {
            "model": model,
            "messages": current_messages,
            "temperature": _resolve_temperature(provider, model, temperature, thinking=thinking),
            "max_tokens": max_tokens,
        }
        if is_k2:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        try:
            with _get_api_semaphore():
                response = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            retry_count += 1
            logger.warning(f"OpenAI API error (budget={_budget_var.get()}): {e}")
            if retry_count <= max_retries and _remaining_llm_budget() > 0:
                # Rotate key on retry for Kimi round-robin
                if provider == "kimi":
                    client.api_key = _get_next_kimi_key()
                time.sleep(_retry_delay(retry_count, provider))
                continue
            break
        retry_count = 0
        if not _consume_llm_call():
            break
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and supports_tools:
            tool_calls = choice.message.tool_calls or []
            # For Kimi K2.5 tool-call follow-ups, disable thinking to save tokens
            if is_k2:
                _thinking_disabled_for_tools = True
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                kwargs["temperature"] = _resolve_temperature(provider, model, temperature, thinking=False)
            current_messages = current_messages + [choice.message]
            tool_results_msgs = []
            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_call.function.name, args)
                logger.info(f"Tool {tool_call.function.name}({args}) -> {result[:200]}")
                tool_results_msgs.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            current_messages = current_messages + tool_results_msgs
            # Rotate key before follow-up call
            if provider == "kimi":
                client.api_key = _get_next_kimi_key()
        else:
            final_text = choice.message.content or ""
            # Kimi K2.5 thinking mode: reasoning goes to reasoning_content
            if is_k2 and not final_text:
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
                if reasoning:
                    final_text = reasoning
            break

    if not final_text:
        err_msg = str(last_error) if last_error else "Exceeded LLM call limit"
        final_text = f"<FINAL_ANSWER>Error: {err_msg}</FINAL_ANSWER>"

    return final_text
