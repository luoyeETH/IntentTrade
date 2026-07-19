"""Regression tests for semantic intent and price-triggered paper fills."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_trade.analysis.intent import IntentAnalyzer
import intent_trade.analysis.intent as intent_module
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.config import (
    AnalysisConfig,
    AppConfig,
    ExecutionConfig,
    Settings,
    TwitterConfig,
)
from intent_trade.execution.paper import PaperBroker
from intent_trade.execution.timing import evaluate_signal
from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    MarketSnapshot,
    IntentAnalysis,
    PositionState,
    SignalState,
    SignalType,
    SocialPost,
    TradingSignal,
)
from intent_trade.pipeline.runner import Pipeline
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


def test_explicit_already_entered_overrides_planned_model_output() -> None:
    analyzer = IntentAnalyzer(
        TickerMap(ROOT / "config" / "ticker_aliases.yaml"),
        AnalysisConfig(mode="llm"),
    )
    analysis = analyzer._analyze_llm(
        SocialPost(
            id="entered-guard",
            author_username="kol",
            text="1345 闪迪我已经上车了",
            created_at=_now(),
        ),
        data_override={
            "mentions": ["闪迪"],
            "canonical_symbols": ["SNDK"],
            "direction": "long",
            "action": "open",
            "position_state": "planned",
            "entry_mode": "unknown",
            "signal_type": "structured",
            "entry_price": 1345,
            "confidence": 0.95,
            "summary": "模型错误标成计划入场",
        },
    )

    assert analysis.position_state == PositionState.ENTERED
    assert analysis.entry_mode == EntryMode.MARKET


@pytest.mark.parametrize(("image_count", "expected_calls"), [(1, 3), (2, 4)])
def test_multimodal_analyzes_original_images_independently_then_merges(
    monkeypatch: pytest.MonkeyPatch,
    image_count: int,
    expected_calls: int,
) -> None:
    calls: list[tuple[str, object]] = []

    text_result = {
        "mentions": ["闪迪"],
        "canonical_symbols": ["SNDK"],
        "direction": "unknown",
        "action": "watch",
        "position_state": "unknown",
        "entry_mode": "unknown",
        "signal_type": "descriptive",
        "confidence": 0.6,
        "summary": "正文提到闪迪，等待图片给出计划",
        "evidence": {"symbol": "text: 闪迪"},
    }
    final_result = {
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
        "summary": "正文标的与图片计划合并",
        "evidence": {"entry": "image_1: 图中入场线 1300"},
    }

    def fake_chat_json(system, user, **kwargs):
        calls.append(("text" if system == intent_module.SYSTEM_PROMPT else "merge", user))
        if system == intent_module.MERGE_SYSTEM_PROMPT:
            assert '"image_analyses"' in user
            return final_result
        assert "legacy OCR text" not in user
        return text_result

    def fake_chat_json_content(system, content, **kwargs):
        calls.append(("image", content))
        assert system == intent_module.IMAGE_SYSTEM_PROMPT
        image_block = content[0]
        assert image_block["type"] == "image"
        assert image_block["source"] == {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "encoded-image",
        }
        return {
            **final_result,
            "summary": "图片独立交易计划",
            "evidence": {"entry": "图中入场线 1300"},
        }

    monkeypatch.setenv("INTENT_TRADE_LLM_KEY", "test-key")
    monkeypatch.setattr(intent_module, "chat_json", fake_chat_json)
    monkeypatch.setattr(intent_module, "chat_json_content", fake_chat_json_content)
    monkeypatch.setattr(
        intent_module,
        "image_source_from_url",
        lambda url: {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "encoded-image",
        },
    )

    analyzer = IntentAnalyzer(
        TickerMap(ROOT / "config" / "ticker_aliases.yaml"),
        AnalysisConfig(mode="llm"),
    )
    post = SocialPost(
        id=f"multimodal-{image_count}",
        author_username="kol",
        text="闪迪看这张图",
        created_at=_now(),
        media_urls=[f"https://img.test/{index}.jpg" for index in range(image_count)],
        media_transcripts=["legacy OCR text must not enter the text stage"],
    )

    analysis = analyzer.analyze(
        post,
        history=[
            {
                "kind": "note",
                "post_id": "older-note",
                "time": (_now() - timedelta(hours=1)).isoformat(),
                "symbol": "SNDK",
                "content": "此前关注闪迪",
            }
        ],
    )

    assert len(calls) == expected_calls
    assert [kind for kind, _ in calls].count("text") == 1
    assert [kind for kind, _ in calls].count("image") == image_count
    assert [kind for kind, _ in calls].count("merge") == 1
    assert analysis.analyzer == "llm_multimodal"
    assert analysis.analysis_text == post.text
    assert analysis.entry_price == 1300
    assert len(analysis.extracted_fields["image_analyses"]) == image_count
    assert analysis.extracted_fields["memory"]["relation"] == "uncertain"


def test_memory_review_supersedes_old_pending_plan_after_entry_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = _now()
    settings = Settings(
        app=AppConfig(db_path=str(tmp_path / "memory.db")),
        twitter=TwitterConfig(source="mock", auto_poll=False),
        analysis=AnalysisConfig(
            mode="llm",
            memory_enabled=True,
            agent_tools_enabled=False,
        ),
    )
    pipe = Pipeline(settings)
    old_post = SocialPost(
        id="old-plan",
        author_username="kol",
        text="闪迪 1290-1300 抄底",
        created_at=now - timedelta(hours=2),
    )
    pipe.storage.upsert_post(old_post)
    old_signal = _signal(
        id="old-signal",
        post_id=old_post.id,
        kol_username="kol",
        symbol="SNDK",
        entry_mode=EntryMode.RANGE,
        entry_price=1295,
        entry_price_low=1290,
        entry_price_high=1300,
        signal_time=old_post.created_at,
        state=SignalState.WAITING_ENTRY,
        source_text=old_post.text,
    )
    pipe.storage.insert_signal(old_signal)
    current_post = SocialPost(
        id="entry-confirmation",
        author_username="kol",
        text="1345 闪迪我上车了",
        created_at=now,
    )
    pipe.storage.upsert_post(current_post)

    calls: list[str] = []
    current_result = {
        "mentions": ["闪迪"],
        "canonical_symbols": ["SNDK"],
        "direction": "long",
        "action": "open",
        "position_state": "entered",
        "entry_mode": "market",
        "signal_type": "structured",
        "entry_price": 1345,
        "confidence": 0.95,
        "summary": "1345 已上车闪迪",
        "evidence": {"entry": "1345 闪迪我上车了"},
    }

    def fake_chat_json(system, user, **kwargs):
        calls.append(system)
        if system == intent_module.MEMORY_SYSTEM_PROMPT:
            assert '"entry_price_low": 1290' in user
            assert '"state": "waiting_entry"' in user
            return {
                **current_result,
                "memory_relation": "confirms_entry",
                "memory_confidence": 0.96,
                "related_symbol": "SNDK",
                "supersede_signal_ids": [old_signal.id],
                "memory_summary": "1345 已上车，取代 1290-1300 的旧等待计划",
                "reasoning": "当前推文确认已经入场，旧限价计划不应继续等待",
            }
        return current_result

    monkeypatch.setenv("INTENT_TRADE_LLM_KEY", "test-key")
    monkeypatch.setattr(intent_module, "chat_json", fake_chat_json)

    analysis = pipe.analyze_post(current_post)

    assert calls == [intent_module.SYSTEM_PROMPT, intent_module.MEMORY_SYSTEM_PROMPT]
    assert analysis.position_state == PositionState.ENTERED
    assert analysis.entry_price == 1345
    assert analysis.extracted_fields["memory"]["relation"] == "confirms_entry"
    assert analysis.extracted_fields["memory"]["applied_signal_ids"] == [old_signal.id]
    stored_old = next(s for s in pipe.storage.list_signals() if s.id == old_signal.id)
    assert stored_old.state == SignalState.SUPERSEDED
    assert "1345 已上车" in stored_old.decision_reason
    assert old_signal.id not in {
        signal.id for signal in pipe.storage.list_signals(unexecuted_only=True)
    }


def test_unchanged_continuation_does_not_create_a_duplicate_signal(
    tmp_path: Path,
) -> None:
    now = _now()
    pipe = Pipeline(
        Settings(
            app=AppConfig(db_path=str(tmp_path / "continuation.db")),
            twitter=TwitterConfig(source="mock", auto_poll=False),
        )
    )
    old_signal = _signal(
        id="continuing-signal",
        post_id="old-plan",
        signal_time=now - timedelta(hours=1),
        state=SignalState.WAITING_ENTRY,
    )
    pipe.storage.insert_signal(old_signal)
    current_post = SocialPost(
        id="continue-post",
        author_username="kol",
        text="闪迪继续等 1300",
        created_at=now,
    )
    history = pipe._recent_kol_history(current_post)
    analysis = IntentAnalysis(
        post_id=current_post.id,
        kol_username="kol",
        raw_text=current_post.text,
        canonical_symbols=["SNDK"],
        direction=Direction.LONG,
        action=IntentAction.OPEN,
        position_state=PositionState.PLANNED,
        entry_mode=EntryMode.LIMIT,
        signal_type=SignalType.STRUCTURED,
        entry_price=1300,
        confidence=0.95,
        summary="继续等待 1300",
        extracted_fields={
            "memory": {
                "relation": "continues",
                "confidence": 0.96,
                "related_symbol": "SNDK",
                "supersede_signal_ids": [],
                "summary": "原 1300 等待计划保持不变",
            }
        },
    )

    assert pipe._apply_memory_actions(analysis, history) == []
    assert analysis.signal_type == SignalType.DESCRIPTIVE
    assert analysis.extracted_fields["memory"]["suppressed_duplicate_signal"] is True
    assert pipe.storage.list_signals()[0].state == SignalState.WAITING_ENTRY


def test_batch_persists_oldest_post_before_reviewing_the_next(
    tmp_path: Path,
) -> None:
    now = _now()
    pipe = Pipeline(
        Settings(
            app=AppConfig(db_path=str(tmp_path / "ordered.db")),
            twitter=TwitterConfig(source="mock", auto_poll=False),
            analysis=AnalysisConfig(mode="rule_based"),
        )
    )
    first = SocialPost(
        id="batch-first",
        author_username="kol",
        text="闪迪 1300 买",
        created_at=now - timedelta(minutes=10),
    )
    second = SocialPost(
        id="batch-second",
        author_username="kol",
        text="闪迪继续等待",
        created_at=now,
    )
    pipe.storage.upsert_post(second)
    pipe.storage.upsert_post(first)
    seen_history: list[list[dict]] = []

    class FakeAnalyzer:
        def analyze(self, post, *, history=None):
            seen_history.append(list(history or []))
            if post.id == first.id:
                return IntentAnalysis(
                    post_id=post.id,
                    kol_username="kol",
                    raw_text=post.text,
                    analysis_text=post.text,
                    canonical_symbols=["SNDK"],
                    direction=Direction.LONG,
                    action=IntentAction.OPEN,
                    position_state=PositionState.PLANNED,
                    entry_mode=EntryMode.LIMIT,
                    signal_type=SignalType.STRUCTURED,
                    entry_price=1300,
                    confidence=0.95,
                    summary="1300 计划买入",
                )
            return IntentAnalysis(
                post_id=post.id,
                kol_username="kol",
                raw_text=post.text,
                analysis_text=post.text,
                canonical_symbols=["SNDK"],
                direction=Direction.LONG,
                action=IntentAction.WATCH,
                signal_type=SignalType.DESCRIPTIVE,
                confidence=0.8,
                summary="继续等待",
                descriptive_note="继续等待",
            )

    pipe.analyzer = FakeAnalyzer()
    analyses, signals, notes = pipe.analyze_new_posts()

    assert [item.post_id for item in analyses] == [first.id, second.id]
    assert seen_history[0] == []
    assert any(item.get("post_id") == first.id for item in seen_history[1])
    assert len(signals) == 1
    assert len(notes) == 1


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
