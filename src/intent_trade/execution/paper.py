"""Paper trading: open positions from structured signals, simulate fills."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from intent_trade.config import ExecutionConfig
from intent_trade.execution.timing import apply_decision, evaluate_signal
from intent_trade.market.prices import MarketDataService
from intent_trade.models.domain import (
    Direction,
    IntentAction,
    MarketSnapshot,
    PaperTrade,
    TradeSide,
    TradeStatus,
    TradingSignal,
    SignalState,
)
from intent_trade.storage.db import Storage


def _format_price(value: float) -> str:
    return f"{value:.8g}"


class PaperBroker:
    def __init__(
        self,
        storage: Storage,
        market: MarketDataService,
        config: ExecutionConfig,
        *,
        require_live_market: bool = False,
        max_signal_age_hours: Optional[float] = None,
    ) -> None:
        self.storage = storage
        self.market = market
        self.config = config
        self.require_live_market = require_live_market
        self.max_signal_age_hours = max_signal_age_hours

    def evaluate_signal(self, signal: TradingSignal):
        """Refresh one signal's state from the latest quote and persist it."""

        quote_getter = getattr(self.market, "get_current_snapshot", None)
        if quote_getter is not None:
            snapshot = quote_getter(signal.symbol)
        else:
            # Compatibility for older/test market adapters that only expose
            # historical bars. It is deliberately marked non-live.
            bar = self.market.get_price_at_or_after(signal.symbol, signal.signal_time)
            snapshot = MarketSnapshot(
                symbol=signal.symbol,
                price=bar.close if bar else self.market.get_latest_price(signal.symbol),
                ts=bar.ts if bar else signal.signal_time,
                source="legacy_history",
                is_live=False,
                stale=True,
            )
        decision = evaluate_signal(
            signal,
            snapshot,
            require_live=self.require_live_market,
            max_age_hours=self.max_signal_age_hours,
            tolerance_pct=self.config.entry_tolerance_pct,
        )
        apply_decision(signal, decision)
        self.storage.update_signal_decision(signal)
        return decision, snapshot

    def execute_signal(self, signal: TradingSignal) -> Optional[PaperTrade]:
        if signal.executed:
            return None

        decision, snapshot = self.evaluate_signal(signal)
        if not decision.can_execute:
            return None
        if signal.action not in (IntentAction.OPEN, IntentAction.ADD):
            return None
        if signal.direction not in (Direction.LONG, Direction.SHORT):
            return None

        open_trades = self.storage.list_trades(
            status=TradeStatus.OPEN, symbol=signal.symbol
        )
        if len(open_trades) >= self.config.max_open_trades_per_symbol:
            signal.state = SignalState.WAITING_RISK_LIMIT
            signal.decision_reason = (
                f"标的已有 {len(open_trades)} 个模拟仓，达到上限 "
                f"{self.config.max_open_trades_per_symbol}，等待风险额度释放"
            )
            self.storage.update_signal_decision(signal)
            return None

        fill = self._resolve_fill(signal, snapshot=snapshot)
        if fill is None:
            return None
        entry_price, entry_time = fill

        sl = signal.stop_loss if signal.stop_loss and signal.stop_loss > 0 else None
        tp = signal.take_profit if signal.take_profit and signal.take_profit > 0 else None

        # Drop SL/TP that are on a totally different price scale than fill
        # (e.g. community quote 1345 vs yfinance single-digit proxy).
        def _scale_ok(level: Optional[float]) -> bool:
            if level is None or entry_price <= 0:
                return False
            ratio = max(level, entry_price) / min(level, entry_price)
            return ratio <= 20

        if sl is not None and not _scale_ok(sl):
            sl = None
        if tp is not None and not _scale_ok(tp):
            tp = None

        if sl is None:
            if signal.direction == Direction.LONG:
                sl = entry_price * (1 - self.config.default_stop_loss_pct)
            else:
                sl = entry_price * (1 + self.config.default_stop_loss_pct)
        if tp is None:
            if signal.direction == Direction.LONG:
                tp = entry_price * (1 + self.config.default_take_profit_pct)
            else:
                tp = entry_price * (1 - self.config.default_take_profit_pct)

        # A malformed model output must not create a trade whose risk levels
        # are on the profitable side of the entry.
        if signal.direction == Direction.LONG:
            if sl >= entry_price:
                sl = entry_price * (1 - self.config.default_stop_loss_pct)
            if tp <= entry_price:
                tp = entry_price * (1 + self.config.default_take_profit_pct)
        else:
            if sl <= entry_price:
                sl = entry_price * (1 + self.config.default_stop_loss_pct)
            if tp >= entry_price:
                tp = entry_price * (1 - self.config.default_take_profit_pct)

        notional = self.config.default_position_size_usd
        qty = notional / entry_price if entry_price else 0
        commission = notional * (self.config.commission_bps / 10_000)
        side = TradeSide.BUY if signal.direction == Direction.LONG else TradeSide.SELL

        trade = PaperTrade(
            signal_id=signal.id,
            kol_username=signal.kol_username,
            symbol=signal.symbol,
            side=side,
            direction=signal.direction,
            quantity=qty,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            status=TradeStatus.OPEN,
            entry_time=entry_time,
            commission_usd=commission,
            notes=f"paper fill from signal {signal.id}; conf={signal.confidence}",
        )
        self.storage.insert_trade(trade)
        signal.executed = True
        signal.state = SignalState.EXECUTED
        signal.decision_reason = (
            signal.decision_reason or "entry condition satisfied"
        ) + f"；已按 {_format_price(entry_price)} 模拟成交"
        self.storage.mark_signal_executed(signal.id)
        self.storage.update_signal_decision(signal)
        return trade

    def _resolve_fill(
        self,
        signal: TradingSignal,
        *,
        snapshot: Optional[MarketSnapshot] = None,
    ) -> Optional[tuple[float, datetime]]:
        # Once a limit/stop condition is satisfied, paper-fill at the latest
        # observable market price. The KOL level remains the trigger/reference.
        if snapshot is not None and snapshot.price is not None and snapshot.price > 0:
            return snapshot.price, snapshot.ts

        # Fallback for callers/tests using a market implementation without the
        # quote API. This path is still only reached after timing evaluation.
        bar = self.market.get_price_at_or_after(signal.symbol, signal.signal_time)
        if bar is None:
            px = self.market.get_latest_price(signal.symbol)
            if px is None:
                # last resort: use SL/TP midpoint if both present
                if (
                    signal.stop_loss
                    and signal.take_profit
                    and signal.stop_loss > 0
                    and signal.take_profit > 0
                ):
                    mid = (signal.stop_loss + signal.take_profit) / 2
                    return mid, signal.signal_time
                return None
            ts = signal.signal_time
        else:
            mode = (self.config.fill_price or "close").lower()
            if mode == "open":
                px = bar.open
            elif mode == "mid":
                px = (bar.high + bar.low) / 2
            else:
                px = bar.close
            ts = bar.ts

        for level in (signal.stop_loss, signal.take_profit):
            if level and level > 0 and px > 0:
                ratio = max(level, px) / min(level, px)
                if ratio > 20:
                    # e.g. TP=500 with market 50 or market 60000 — skip paper fill
                    return None
        return px, ts

    def settle_open_trades(self, path_resolution: str = "conservative") -> list[PaperTrade]:
        """Walk subsequent OHLC bars to hit SL/TP for open trades."""
        updated: list[PaperTrade] = []
        for trade in self.storage.list_trades(status=TradeStatus.OPEN):
            history_is_live = getattr(self.market, "history_is_live", None)
            if self.require_live_market and (
                history_is_live is None or not history_is_live(trade.symbol)
            ):
                # Never close a live-configured paper position against sample
                # bars or an old cache after a market-data outage.
                continue
            bars = self.market.bars_after(trade.symbol, trade.entry_time)
            if not bars:
                # try mark with latest only — leave open
                continue
            closed = self._path_close(trade, bars, path_resolution)
            if closed:
                self.storage.update_trade(closed)
                updated.append(closed)
        return updated

    def _path_close(
        self,
        trade: PaperTrade,
        bars: list,
        path_resolution: str,
    ) -> Optional[PaperTrade]:
        for bar in bars:
            if bar.ts < trade.entry_time:
                continue
            hit_sl = False
            hit_tp = False
            if trade.direction == Direction.LONG:
                if trade.stop_loss is not None and bar.low <= trade.stop_loss:
                    hit_sl = True
                if trade.take_profit is not None and bar.high >= trade.take_profit:
                    hit_tp = True
            else:
                if trade.stop_loss is not None and bar.high >= trade.stop_loss:
                    hit_sl = True
                if trade.take_profit is not None and bar.low <= trade.take_profit:
                    hit_tp = True

            if hit_sl and hit_tp:
                # same bar ambiguity
                if path_resolution == "conservative":
                    return self._close(trade, trade.stop_loss, bar.ts, TradeStatus.CLOSED_SL)
                return self._close(trade, trade.take_profit, bar.ts, TradeStatus.CLOSED_TP)
            if hit_sl:
                return self._close(trade, trade.stop_loss, bar.ts, TradeStatus.CLOSED_SL)
            if hit_tp:
                return self._close(trade, trade.take_profit, bar.ts, TradeStatus.CLOSED_TP)
        return None

    def _close(
        self,
        trade: PaperTrade,
        exit_price: Optional[float],
        exit_time: datetime,
        status: TradeStatus,
    ) -> PaperTrade:
        assert exit_price is not None
        if trade.direction == Direction.LONG:
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
        notional = trade.entry_price * trade.quantity
        pnl_usd = notional * pnl_pct - trade.commission_usd
        # exit commission
        exit_comm = notional * (self.config.commission_bps / 10_000)
        pnl_usd -= exit_comm
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = status
        trade.pnl_pct = round(pnl_pct * 100, 4)
        trade.pnl_usd = round(pnl_usd, 4)
        trade.commission_usd = trade.commission_usd + exit_comm
        trade.notes = (trade.notes or "") + f" | closed {status.value}"
        return trade
