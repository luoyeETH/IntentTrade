"""LLM client for IntentTrade — Anthropic-compatible API (e.g. grok-4.5 via base URL)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


def llm_enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("INTENT_TRADE_LLM_KEY"))


def default_model() -> str:
    return (
        os.getenv("INTENT_TRADE_LLM_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or "grok-4.5"
    )


def chat_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 1200,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Call chat model and parse a JSON object from the response."""
    return chat_json_content(
        system,
        user,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_json_content(
    system: str,
    content: str | list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 1200,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Call the model with text or native multimodal content and parse JSON."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("INTENT_TRADE_LLM_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    import anthropic

    timeout = timeout if timeout is not None else float(
        os.getenv("INTENT_TRADE_LLM_TIMEOUT", "60")
    )
    # anthropic SDK reads ANTHROPIC_BASE_URL from env if present
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    msg = client.messages.create(
        model=model or default_model(),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    raw = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw += block.text
    return _parse_json_object(raw)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # extract first {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON object in LLM response: {text[:200]}")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data
