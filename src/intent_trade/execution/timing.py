"""Entry timing decisions for paper execution.

This module is deliberately deterministic. The LLM explains what a KOL said;
this layer decides whether the current quote satisfies that instruction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    MarketSnapshot,
    PositionState,
    SignalDecision,
    SignalState,
    TradingSignal,
)


def _price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.8g}"


def _age_expired(
    signal: TradingSignal,
    now: datetime,
    max_age_hours: Optional[float],
) -> bool:
    if signal.expires_at is not None and now >= signal.expires_at:
        return True
    if max_age_hours is None or max_age_hours <= 0:
        return False
    return now - signal.signal_time >= timedelta(hours=max_age_hours)


def evaluate_signal(
    signal: TradingSignal,
    market: Optional[MarketSnapshot],
    *,
    now: Optional[datetime] = None,
    require_live: bool = False,
    max_age_hours: Optional[float] = None,
    tolerance_pct: float = 0.0,
) -> SignalDecision:
    """Evaluate one signal without mutating storage or placing a trade."""

    now = now or datetime.utcnow()
    if signal.state == SignalState.SUPERSEDED:
        return SignalDecision(
            state=SignalState.SUPERSEDED,
            reason=signal.decision_reason or "该计划已被后续推文更新取代",
            evaluated_at=now,
        )
    if signal.executed or signal.state == SignalState.EXECUTED:
        return SignalDecision(
            state=SignalState.EXECUTED,
            reason="该信号已经有纸面成交记录",
            evaluated_at=now,
        )

    if signal.action in (IntentAction.CLOSE, IntentAction.REDUCE):
        return SignalDecision(
            state=SignalState.EXIT_INTENT,
            reason="识别为退出/减仓意图；当前版本仅记录，不自动执行",
            evaluated_at=now,
        )

    if signal.position_state == PositionState.ENTERED:
        return SignalDecision(
            state=SignalState.OBSERVED_POSITION,
            reason="KOL 表示已经持仓，当前记录为观察状态，不追价跟单",
            evaluated_at=now,
        )

    if _age_expired(signal, now, max_age_hours):
        return SignalDecision(
            state=SignalState.EXPIRED,
            reason="信号超过有效期，避免追踪过期喊单",
            evaluated_at=now,
        )

    if signal.action not in (IntentAction.OPEN, IntentAction.ADD):
        return SignalDecision(
            state=SignalState.REJECTED,
            reason="未识别为可跟踪的开仓或加仓意图",
            evaluated_at=now,
        )
    if signal.direction not in (Direction.LONG, Direction.SHORT):
        return SignalDecision(
            state=SignalState.REJECTED,
            reason="缺少明确的多空方向",
            evaluated_at=now,
        )

    if market is None or market.price is None or market.price <= 0:
        return SignalDecision(
            state=SignalState.WAITING_MARKET_DATA,
            market_source=market.source if market else "",
            market_is_live=market.is_live if market else None,
            reason="没有可用的当前行情，暂不模拟成交",
            evaluated_at=now,
        )

    if require_live and (not market.is_live or market.stale):
        return SignalDecision(
            state=SignalState.WAITING_MARKET_DATA,
            current_price=market.price,
            market_timestamp=market.ts,
            market_source=market.source,
            market_is_live=market.is_live,
            reason=f"行情来源 {market.source} 不满足实时执行要求，暂不成交",
            evaluated_at=now,
        )

    current = market.price
    mode = signal.entry_mode
    if mode == EntryMode.UNKNOWN:
        mode = EntryMode.LIMIT if signal.entry_price is not None else EntryMode.MARKET

    target = (
        signal.trigger_price
        if mode == EntryMode.STOP and signal.trigger_price is not None
        else signal.entry_price
    )
    low = signal.entry_price_low
    high = signal.entry_price_high
    if mode == EntryMode.RANGE:
        low = low if low is not None else target
        high = high if high is not None else target
        if low is None or high is None:
            return SignalDecision(
                state=SignalState.REJECTED,
                current_price=current,
                market_timestamp=market.ts,
                market_source=market.source,
                market_is_live=market.is_live,
                reason="区间入场缺少上下限",
                evaluated_at=now,
            )
        low, high = min(low, high), max(low, high)
        target = (low + high) / 2
    elif target is None and mode != EntryMode.MARKET:
        return SignalDecision(
            state=SignalState.REJECTED,
            current_price=current,
            market_timestamp=market.ts,
            market_source=market.source,
            market_is_live=market.is_live,
            reason="条件入场缺少价格，无法建立可验证的触发条件",
            evaluated_at=now,
        )

    distance = ((current - target) / target * 100) if target else None
    tolerance = max(0.0, tolerance_pct) / 100
    ready = False
    reason = ""

    if mode == EntryMode.MARKET:
        ready = True
        reason = f"市价入场条件满足，当前价 {_price(current)}"
    elif mode == EntryMode.LIMIT:
        assert target is not None
        if signal.direction == Direction.LONG:
            ready = current <= target * (1 + tolerance)
            reason = (
                f"当前价 {_price(current)} 已到多头限价 {_price(target)}，允许在更优价模拟成交"
                if ready
                else f"等待多头回落至 <= {_price(target)}，当前价 {_price(current)}"
            )
        else:
            ready = current >= target * (1 - tolerance)
            reason = (
                f"当前价 {_price(current)} 已到空头限价 {_price(target)}，允许在更优价模拟成交"
                if ready
                else f"等待空头反弹至 >= {_price(target)}，当前价 {_price(current)}"
            )
    elif mode == EntryMode.STOP:
        assert target is not None
        if signal.direction == Direction.LONG:
            ready = current >= target * (1 - tolerance)
            reason = (
                f"当前价 {_price(current)} 已突破多头触发价 {_price(target)}"
                if ready
                else f"等待多头突破 >= {_price(target)}，当前价 {_price(current)}"
            )
        else:
            ready = current <= target * (1 + tolerance)
            reason = (
                f"当前价 {_price(current)} 已跌破空头触发价 {_price(target)}"
                if ready
                else f"等待空头跌破 <= {_price(target)}，当前价 {_price(current)}"
            )
    elif mode == EntryMode.RANGE:
        assert low is not None and high is not None
        if signal.direction == Direction.LONG:
            ready = current <= high * (1 + tolerance)
            reason = (
                f"当前价 {_price(current)} 位于/低于多头入场区间 {_price(low)}-{_price(high)}"
                if ready
                else f"等待多头回落至入场区间 {_price(low)}-{_price(high)}，当前价 {_price(current)}"
            )
        else:
            ready = current >= low * (1 - tolerance)
            reason = (
                f"当前价 {_price(current)} 位于/高于空头入场区间 {_price(low)}-{_price(high)}"
                if ready
                else f"等待空头反弹至入场区间 {_price(low)}-{_price(high)}，当前价 {_price(current)}"
            )
    else:
        return SignalDecision(
            state=SignalState.REJECTED,
            current_price=current,
            market_timestamp=market.ts,
            market_source=market.source,
            market_is_live=market.is_live,
            reason=f"不支持的入场方式 {mode.value}",
            evaluated_at=now,
        )

    return SignalDecision(
        state=SignalState.READY if ready else SignalState.WAITING_ENTRY,
        can_execute=ready,
        current_price=current,
        market_timestamp=market.ts,
        market_source=market.source,
        market_is_live=market.is_live,
        price_distance_pct=distance,
        reason=reason,
        evaluated_at=now,
    )


def apply_decision(signal: TradingSignal, decision: SignalDecision) -> TradingSignal:
    """Copy a decision onto a signal before it is persisted."""

    signal.state = decision.state
    signal.current_price = decision.current_price
    signal.market_timestamp = decision.market_timestamp
    signal.market_source = decision.market_source
    signal.market_is_live = decision.market_is_live
    signal.price_distance_pct = decision.price_distance_pct
    signal.decision_reason = decision.reason
    signal.last_evaluated_at = decision.evaluated_at
    return signal
