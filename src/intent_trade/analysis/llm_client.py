"""LLM client for IntentTrade — Anthropic-compatible API (e.g. grok-4.5 via base URL)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional


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


def chat_json_agent(
    system: str,
    user: str,
    *,
    tools: list[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: int = 1600,
    max_rounds: int = 6,
    timeout: Optional[float] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run an Anthropic-compatible tool loop and parse the final JSON object."""

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("INTENT_TRADE_LLM_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    import anthropic

    timeout = timeout if timeout is not None else float(
        os.getenv("INTENT_TRADE_LLM_TIMEOUT", "60")
    )
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    trace: list[dict[str, Any]] = []

    for round_index in range(1, max(1, max_rounds) + 1):
        msg = client.messages.create(
            model=model or default_model(),
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        blocks = [_content_block_dict(block) for block in msg.content]
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        if not tool_uses:
            raw = "".join(
                str(block.get("text") or "")
                for block in blocks
                if block.get("type") == "text"
            )
            return _parse_json_object(raw), trace

        messages.append({"role": "assistant", "content": blocks})
        results: list[dict[str, Any]] = []
        for block in tool_uses:
            tool_name = str(block.get("name") or "")
            tool_input = block.get("input") or {}
            if not isinstance(tool_input, dict):
                tool_input = {"value": tool_input}
            output = execute_tool(tool_name, tool_input)
            trace.append(
                {
                    "round": round_index,
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                }
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(block.get("id") or ""),
                    "content": json.dumps(output, ensure_ascii=False, default=str),
                    "is_error": output.get("ok") is False,
                }
            )
        messages.append({"role": "user", "content": results})

    # Give the model one tool-free turn to synthesize what it already learned.
    # This prevents repeated synonym searches from discarding successful calls.
    msg = client.messages.create(
        model=model or default_model(),
        max_tokens=max_tokens,
        system=(
            system
            + "\n\n工具调用预算已用完。不得再请求工具；请立即基于已有工具结果输出最终 JSON。"
        ),
        messages=messages,
    )
    raw = "".join(
        str(block.get("text") or "")
        for block in (_content_block_dict(item) for item in msg.content)
        if block.get("type") == "text"
    )
    return _parse_json_object(raw), trace


def _content_block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    block_type = str(getattr(block, "type", ""))
    if block_type == "text":
        return {"type": "text", "text": str(getattr(block, "text", ""))}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": str(getattr(block, "id", "")),
            "name": str(getattr(block, "name", "")),
            "input": getattr(block, "input", {}) or {},
        }
    return {"type": block_type or "unknown"}


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
