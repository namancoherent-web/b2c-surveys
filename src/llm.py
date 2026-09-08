"""Central LLM factory — provider-agnostic.

Supported: ``deepseek`` (native API, default), ``openrouter``, ``anthropic``.

DeepSeek note
-------------
DeepSeek's OpenAI-compatible tool/function-calling often emits slightly
invalid JSON (extra trailing braces). Native ``json_object`` mode also
requires the word "json" in the prompt. For DeepSeek we therefore:
  1) ask for a single JSON object matching the Pydantic schema, and
  2) repair common malformations before validation.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Optional

from src import config
from src.ratelimit import acquire

# Shared across all workers via Redis. Tune independently of worker count.
_LLM_RATE = float(os.getenv("LLM_RATE_PER_SEC", "5"))
_LLM_BURST = float(os.getenv("LLM_BURST", "10"))
_LLM_MAX_ATTEMPTS = 5


def _is_rate_or_timeout(exc: BaseException) -> bool:
    """True for 429 / timeout-class failures worth retrying with backoff."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError"):
        return True
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 429


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(8, 2 ** attempt) + random.uniform(0, 1))


async def _backoff_asleep(attempt: int) -> None:
    await asyncio.sleep(min(8, 2 ** attempt) + random.uniform(0, 1))


def _call_provider_sync(fn, *args, **kwargs):
    """Acquire LLM token, call provider, retry 429/timeout with jittered backoff."""
    last: BaseException | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        acquire("llm", _LLM_RATE, _LLM_BURST)
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — classify then re-raise or retry
            last = e
            if not _is_rate_or_timeout(e) or attempt == _LLM_MAX_ATTEMPTS - 1:
                raise
            _backoff_sleep(attempt)
    assert last is not None
    raise last


async def _call_provider_async(fn, *args, **kwargs):
    """Async variant of ``_call_provider_sync``."""
    last: BaseException | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        # Token-bucket acquire is sync Redis; run off the event loop briefly.
        await asyncio.to_thread(acquire, "llm", _LLM_RATE, _LLM_BURST)
        try:
            return await fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_rate_or_timeout(e) or attempt == _LLM_MAX_ATTEMPTS - 1:
                raise
            await _backoff_asleep(attempt)
    assert last is not None
    raise last


def get_llm(temperature: float = 0.0, max_tokens=None):
    """Build a chat model for the configured provider."""
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
        )

    from langchain_openai import ChatOpenAI

    capped = min(max_tokens, config.MAX_OUTPUT_TOKENS) if max_tokens else None

    if config.LLM_PROVIDER == "deepseek":
        return ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=capped,
        )

    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=capped,
        default_headers={"X-Title": "survey-agent"},
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text") or "")
        return "".join(parts)
    return str(content)


def repair_json_text(text: str) -> str:
    """Extract/repair a JSON object from model output (fences, extra braces)."""
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model output; expected JSON.")

    # Strip markdown fences.
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()

    # Prefer the outermost {...} span.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s)
        return json.dumps(obj)
    except json.JSONDecodeError:
        pass

    # Peel trailing braces until it parses (DeepSeek sometimes adds extras).
    candidate = s
    for _ in range(5):
        try:
            obj, _ = decoder.raw_decode(candidate)
            return json.dumps(obj)
        except json.JSONDecodeError:
            if candidate.endswith("}"):
                candidate = candidate[:-1].rstrip()
                continue
            break
    raise ValueError(f"Could not parse JSON from model output: {s[:240]!r}")


# change for b2c questionarie — cost/time optimization step 1. A caller MAY
# pass a (static_text, dynamic_text) tuple instead of a flat string wherever
# this module accepts "input"/"user_input". `static_text` is the part that is
# byte-identical across many calls (e.g. a fixed prompt template with its
# per-call variables already substituted out into `dynamic_text`); it is only
# ever used to place an Anthropic `cache_control` breakpoint. No caller in
# this codebase builds that split today, so this is inert unless a caller
# opts in — it changes nothing about what any provider is asked to do. On
# every path but Anthropic, the two halves are simply concatenated back into
# the exact same single string that was sent before this change.
PromptInput = Any  # str | tuple[str, str]


def _split_prompt(input: PromptInput) -> tuple[str, str]:
    if isinstance(input, tuple):
        static_text, dynamic_text = input
        return static_text, dynamic_text
    return "", input


def _flatten_prompt(input: PromptInput) -> str:
    static_text, dynamic_text = _split_prompt(input)
    return f"{static_text}{dynamic_text}" if static_text else dynamic_text


def _anthropic_cached_messages(input: PromptInput) -> Optional[list]:
    """Two-block message with a cache breakpoint after the static half.

    Returns None when there is no static/dynamic split to make (the common
    case), so callers fall back to sending a plain string exactly as before.
    """
    static_text, dynamic_text = _split_prompt(input)
    if not static_text:
        return None
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": static_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_text},
            ],
        }
    ]


class _DeepSeekStructured:
    """Runnable-like wrapper: prompt → JSON → Pydantic model."""

    def __init__(self, schema, temperature: float = 0.0, max_tokens=None):
        self.schema = schema
        self.llm = get_llm(temperature=temperature, max_tokens=max_tokens)

    def _build_prompt(self, user_input: PromptInput) -> str:
        schema_json = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        return (
            f"{_flatten_prompt(user_input)}\n\n"
            "Return your answer as a single valid JSON object only "
            "(no markdown, no commentary) that matches this JSON schema:\n"
            f"{schema_json}\n"
        )

    def _parse(self, message) -> Any:
        text = _content_to_text(getattr(message, "content", message))
        repaired = repair_json_text(text)
        return self.schema.model_validate_json(repaired)

    def invoke(self, input: PromptInput, config: Optional[dict] = None):
        # Limiter + backoff wrap the HTTP call only; JSON repair stays below.
        msg = _call_provider_sync(
            self.llm.invoke, self._build_prompt(input), config=config
        )
        return self._parse(msg)

    async def ainvoke(self, input: PromptInput, config: Optional[dict] = None):
        msg = await _call_provider_async(
            self.llm.ainvoke, self._build_prompt(input), config=config
        )
        return self._parse(msg)


class _RateLimitedStructured:
    """Wrap any structured-output runnable with the shared LLM rate limiter.

    When the configured provider is Anthropic and the caller passed a
    (static_text, dynamic_text) tuple, the static half is sent as a separate
    cached message block. Every other provider, and every caller that still
    passes a plain string, is unaffected byte-for-byte.
    """

    # Anthropic-only: a malformed/truncated tool call raises a validation
    # error from with_structured_output, which is NOT a rate-limit/timeout
    # case and so _call_provider_sync/async re-raise it immediately. Callers
    # like survey_simulator._simulate_batch catch that with a bare
    # `except Exception: by_id = {}` and silently fall back to low-confidence
    # placeholder answers for the WHOLE batch — observed live: a Sonnet run's
    # last 4 revisions all showed 0 confident questions / 0% grounded, which
    # was this failure being mistaken for a genuine quality judgement. A
    # couple of quick retries on the SAME input is enough to distinguish a
    # transient tool-call hiccup from an actual persistent failure, without
    # touching DeepSeek's own (already-repairing) JSON path at all.
    _STRUCTURED_RETRY_ATTEMPTS = 2

    def __init__(self, inner):
        self._inner = inner

    def _resolve(self, input: PromptInput):
        if config.LLM_PROVIDER == "anthropic":
            cached = _anthropic_cached_messages(input)
            if cached is not None:
                return cached
        return _flatten_prompt(input) if isinstance(input, tuple) else input

    def invoke(self, input: PromptInput, config: Optional[dict] = None):
        resolved = self._resolve(input)
        if self._is_anthropic():
            last: BaseException | None = None
            for attempt in range(self._STRUCTURED_RETRY_ATTEMPTS + 1):
                try:
                    return _call_provider_sync(self._inner.invoke, resolved, config=config)
                except Exception as e:  # noqa: BLE001
                    last = e
                    if attempt == self._STRUCTURED_RETRY_ATTEMPTS:
                        raise
            assert last is not None
            raise last
        return _call_provider_sync(self._inner.invoke, resolved, config=config)

    async def ainvoke(self, input: PromptInput, config: Optional[dict] = None):
        resolved = self._resolve(input)
        if self._is_anthropic():
            last: BaseException | None = None
            for attempt in range(self._STRUCTURED_RETRY_ATTEMPTS + 1):
                try:
                    return await _call_provider_async(
                        self._inner.ainvoke, resolved, config=config
                    )
                except Exception as e:  # noqa: BLE001
                    last = e
                    if attempt == self._STRUCTURED_RETRY_ATTEMPTS:
                        raise
            assert last is not None
            raise last
        return await _call_provider_async(self._inner.ainvoke, resolved, config=config)

    @staticmethod
    def _is_anthropic() -> bool:
        return config.LLM_PROVIDER == "anthropic"


def get_structured_llm(schema, temperature: float = 0.0, max_tokens=None):
    """Return an LLM bound to a Pydantic schema via structured output.

    DeepSeek uses a dedicated JSON+repair path. Other providers use LangChain's
    native ``with_structured_output``. All paths share the Redis LLM bucket.
    """
    if config.LLM_PROVIDER == "deepseek":
        return _DeepSeekStructured(schema, temperature=temperature, max_tokens=max_tokens)

    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    return _RateLimitedStructured(llm.with_structured_output(schema))
