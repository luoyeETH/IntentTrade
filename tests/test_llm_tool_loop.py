from __future__ import annotations

import sys
from types import SimpleNamespace

from intent_trade.analysis.llm_client import chat_json_agent


def test_anthropic_tool_loop_returns_json_and_trace(monkeypatch) -> None:
    calls = []
    responses = [
        SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "get_market_snapshot",
                    "input": {"symbol": "BTC-USD"},
                }
            ]
        ),
        SimpleNamespace(
            content=[
                {
                    "type": "text",
                    "text": '{"canonical_symbols":["BTC-USD"],"summary":"ok"}',
                }
            ]
        ),
    ]

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("INTENT_TRADE_LLM_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )

    data, trace = chat_json_agent(
        "system",
        "user",
        tools=[
            {
                "name": "get_market_snapshot",
                "description": "quote",
                "input_schema": {"type": "object"},
            }
        ],
        execute_tool=lambda name, args: {
            "ok": True,
            "symbol": args["symbol"],
            "price": 100,
        },
    )

    assert data["canonical_symbols"] == ["BTC-USD"]
    assert trace[0]["tool"] == "get_market_snapshot"
    assert calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_tool_budget_forces_a_final_tool_free_response(monkeypatch) -> None:
    responses = [
        SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "search_instruments",
                    "input": {"query": "SK Hynix"},
                }
            ]
        ),
        SimpleNamespace(content=[{"type": "text", "text": '{"summary":"final"}'}]),
    ]
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("INTENT_TRADE_LLM_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )
    data, trace = chat_json_agent(
        "system",
        "user",
        tools=[
            {
                "name": "search_instruments",
                "description": "search",
                "input_schema": {"type": "object"},
            }
        ],
        execute_tool=lambda name, args: {"ok": True, "results": []},
        max_rounds=1,
    )

    assert data == {"summary": "final"}
    assert len(trace) == 1
    assert "tools" not in calls[-1]
    assert "工具调用预算已用完" in calls[-1]["system"]
