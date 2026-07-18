"""Regression tests for semantic intent and price-triggered paper fills."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_trade.analysis.intent import IntentAnalyzer
import intent_trade.analysis.intent as intent_module
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.config import AnalysisConfig, ExecutionConfig
from intent_trade.execution.paper import PaperBroker
from intent_trade.execution.timing import evaluate_signal
from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    MarketSnapshot,
    PositionState,
    SignalState,
    SocialPost,
    TradingSignal,
)
from intent_trade.storage.db import Storage
from intent_trade.time_utils import format_display_time


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _signal(**overrides) -> TradingSignal:
    values = dict(
        post_id="post-1",
        kol_username="kol",
        symbol="SNDK",
        direction=Direction.LONG,
        action=IntentAction.OPEN,
        position_state=PositionState.PLANNED,
        entry_mode=EntryMode.LIMIT,
        entry_price=1300,
        stop_loss=1200,
        take_profit=1500,
        confidence=0.9,
        signal_time=_now(),
    )
    values.update(overrides)
    return TradingSignal(**values)


def test_fallback_extracts_entry_without_using_stop_price() -> None:
    analyzer = IntentAnalyzer(
        TickerMap(ROOT / "config" / "ticker_aliases.yaml"),
        AnalysisConfig(mode="rule_based"),
    )
    analysis = analyzer.analyze(
        SocialPost(
            id="post-1",
            author_username="kol",
            text="闪迪 1300 买入，止损1200，目标1500",
            created_at=_now(),
        )
    )
    assert analysis.canonical_symbols == ["SNDK"]
    assert analysis.entry_price == 1300
    assert analysis.stop_loss == 1200
    assert analysis.take_profit == 1500
    assert analysis.entry_mode == EntryMode.LIMIT
    assert analysis.position_state == PositionState.PLANNED


def test_display_time_uses_beijing_for_utc_storage() -> None:
    assert format_display_time(datetime(2026, 7, 18, 0, 0)) == "2026-07-18 08:00"


def test_llm_prompt_and_normalization_accept_structured_response() -> None:
    original = intent_module.chat_json

    def fake_chat_json(system, user, **kwargs):
        assert '"field_confidence"' in user
        assert '"evidence"' in user
        return {
            "mentions": ["闪迪"],
            "canonical_symbols": ["SNDK"],
            "direction": "long",
            "action": "open",
            "position_state": "planned",
            "entry_mode": "limit",
            "signal_type": "structured",
            "entry_price": 1300,
            "stop_loss": 1200,
            "take_profit": 1500,
            "confidence": 0.95,
            "field_confidence": {"entry": 0.99},
            "evidence": {"entry": "1300买"},
            "summary": "limit long",
        }

    intent_module.chat_json = fake_chat_json
    try:
        analyzer = IntentAnalyzer(
            TickerMap(ROOT / "config" / "ticker_aliases.yaml"),
            AnalysisConfig(mode="llm"),
        )
        analysis = analyzer._analyze_llm(
            SocialPost(
                id="llm-1",
                author_username="kol",
                text="闪迪1300买",
                created_at=_now(),
            )
        )
        assert analysis.entry_mode == EntryMode.LIMIT
        assert analysis.entry_price == 1300
        assert analysis.field_confidence["entry"] == 0.99
    finally:
        intent_module.chat_json = original


def test_limit_long_waits_above_requested_price() -> None:
    signal = _signal()
    waiting = evaluate_signal(
        signal,
        MarketSnapshot(symbol="SNDK", price=1350, source="test", is_live=True),
        require_live=True,
    )
    assert waiting.state == SignalState.WAITING_ENTRY
    assert not waiting.can_execute
    assert "1300" in waiting.reason

    ready = evaluate_signal(
        signal,
        MarketSnapshot(symbol="SNDK", price=1290, source="test", is_live=True),
        require_live=True,
    )
    assert ready.state == SignalState.READY
    assert ready.can_execute


def test_short_limit_and_stop_have_opposite_triggers() -> None:
    short_limit = _signal(
        direction=Direction.SHORT,
        entry_price=1300,
        stop_loss=1400,
        take_profit=1100,
    )
    assert evaluate_signal(
        short_limit,
        MarketSnapshot(symbol="SNDK", price=1250, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.WAITING_ENTRY
    assert evaluate_signal(
        short_limit,
        MarketSnapshot(symbol="SNDK", price=1350, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.READY

    long_stop = _signal(entry_mode=EntryMode.STOP, entry_price=1300)
    assert evaluate_signal(
        long_stop,
        MarketSnapshot(symbol="SNDK", price=1290, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.WAITING_ENTRY
    assert evaluate_signal(
        long_stop,
        MarketSnapshot(symbol="SNDK", price=1310, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.READY


def test_range_entry_and_expiry_are_explicit() -> None:
    range_signal = _signal(
        entry_mode=EntryMode.RANGE,
        entry_price=1300,
        entry_price_low=1280,
        entry_price_high=1320,
    )
    assert evaluate_signal(
        range_signal,
        MarketSnapshot(symbol="SNDK", price=1350, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.WAITING_ENTRY
    assert evaluate_signal(
        range_signal,
        MarketSnapshot(symbol="SNDK", price=1300, source="test", is_live=True),
        require_live=True,
    ).state == SignalState.READY

    old = _signal(signal_time=_now() - timedelta(hours=2))
    expired = evaluate_signal(
        old,
        MarketSnapshot(symbol="SNDK", price=1290, source="test", is_live=True),
        require_live=True,
        max_age_hours=1,
    )
    assert expired.state == SignalState.EXPIRED


def test_observed_position_and_stale_quote_do_not_execute() -> None:
    observed = _signal(position_state=PositionState.ENTERED, entry_mode=EntryMode.MARKET)
    decision = evaluate_signal(
        observed,
        MarketSnapshot(symbol="SNDK", price=1350, source="test", is_live=True),
        require_live=True,
    )
    assert decision.state == SignalState.OBSERVED_POSITION
    assert not decision.can_execute

    stale = evaluate_signal(
        _signal(),
        MarketSnapshot(symbol="SNDK", price=1290, source="sample_fallback", stale=True),
        require_live=True,
    )
    assert stale.state == SignalState.WAITING_MARKET_DATA
    assert not stale.can_execute


def test_exit_condition_keeps_trigger_price_without_becoming_an_entry() -> None:
    analyzer = IntentAnalyzer(
        TickerMap(ROOT / "config" / "ticker_aliases.yaml"),
        AnalysisConfig(mode="rule_based"),
    )
    analysis = analyzer.analyze(
        SocialPost(
            id="exit-1",
            author_username="kol",
            text="比特币若跌破61000我会减仓",
            created_at=_now(),
        )
    )
    assert analysis.action == IntentAction.REDUCE
    assert analysis.trigger_price == 61000
    assert analysis.entry_price is None
    assert analysis.signal_type.value == "structured"

    signal = _signal(
        action=IntentAction.REDUCE,
        direction=Direction.UNKNOWN,
        entry_price=None,
        trigger_price=61000,
        entry_mode=EntryMode.STOP,
    )
    decision = evaluate_signal(
        signal,
        MarketSnapshot(symbol="BTC-USD", price=60000, source="test", is_live=True),
        require_live=True,
    )
    assert decision.state == SignalState.EXIT_INTENT
    assert not decision.can_execute


def test_paper_broker_fills_only_after_limit_trigger() -> None:
    class FakeMarket:
        def __init__(self, price: float) -> None:
            self.price = price

        def get_current_snapshot(self, symbol: str) -> MarketSnapshot:
            return MarketSnapshot(
                symbol=symbol,
                price=self.price,
                source="test",
                is_live=True,
                stale=False,
            )

        def get_price_at_or_after(self, symbol: str, when: datetime):
            return None

        def get_latest_price(self, symbol: str):
            return self.price

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "intent.db")
        market = FakeMarket(1350)
        broker = PaperBroker(
            storage,
            market,
            ExecutionConfig(),
            require_live_market=True,
        )
        signal = _signal()
        storage.insert_signal(signal)
        assert broker.execute_signal(signal) is None
        assert storage.list_trades() == []
        assert storage.list_signals()[0].state == SignalState.WAITING_ENTRY

        market.price = 1290
        loaded = storage.list_signals()[0]
        trade = broker.execute_signal(loaded)
        assert trade is not None
        assert trade.entry_price == 1290
        assert storage.list_signals()[0].state == SignalState.EXECUTED
        assert storage.list_signals()[0].executed is True


if __name__ == "__main__":
    test_fallback_extracts_entry_without_using_stop_price()
    test_display_time_uses_beijing_for_utc_storage()
    test_llm_prompt_and_normalization_accept_structured_response()
    test_limit_long_waits_above_requested_price()
    test_short_limit_and_stop_have_opposite_triggers()
    test_range_entry_and_expiry_are_explicit()
    test_observed_position_and_stale_quote_do_not_execute()
    test_exit_condition_keeps_trigger_price_without_becoming_an_entry()
    test_paper_broker_fills_only_after_limit_trigger()
    print("timing tests passed")
