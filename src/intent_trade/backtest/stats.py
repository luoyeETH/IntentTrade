"""KOL / system performance stats for paper trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from intent_trade.models.domain import PaperTrade, TradeStatus
from intent_trade.storage.db import Storage


CLOSED = {
    TradeStatus.CLOSED_TP,
    TradeStatus.CLOSED_SL,
    TradeStatus.CLOSED_MANUAL,
    TradeStatus.EXPIRED,
}


@dataclass
class KOLStats:
    kol_username: str
    total_trades: int = 0
    closed_trades: int = 0
    open_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_usd: float = 0.0
    avg_pnl_pct: float = 0.0
    best_trade_pct: Optional[float] = None
    worst_trade_pct: Optional[float] = None
    symbols: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kol": self.kol_username,
            "total_trades": self.total_trades,
            "closed": self.closed_trades,
            "open": self.open_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "avg_pnl_pct": round(self.avg_pnl_pct, 3),
            "best_trade_pct": self.best_trade_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "symbols": self.symbols,
            "summary": (
                f"跟单 {self.kol_username}: {self.closed_trades} 单已平, "
                f"盈利 {self.wins} 单, 胜率 {self.win_rate:.1%}, "
                f"累计PnL ${self.total_pnl_usd:.2f}"
            ),
        }


class PerformanceReporter:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def kol_stats(self, kol: Optional[str] = None) -> list[KOLStats]:
        trades = self.storage.list_trades(kol=kol)
        by_kol: dict[str, list[PaperTrade]] = {}
        for t in trades:
            by_kol.setdefault(t.kol_username, []).append(t)
        return [self._calc(k, v) for k, v in sorted(by_kol.items())]

    def overall(self) -> dict[str, Any]:
        trades = self.storage.list_trades()
        stats = self._calc("_all_", trades)
        d = stats.as_dict()
        d["kol"] = "ALL"
        d["summary"] = (
            f"全市场模拟: {stats.closed_trades} 单已平, "
            f"盈利 {stats.wins} 单, 胜率 {stats.win_rate:.1%}, "
            f"累计PnL ${stats.total_pnl_usd:.2f}"
        )
        return d

    def _calc(self, kol: str, trades: list[PaperTrade]) -> KOLStats:
        closed = [t for t in trades if t.status in CLOSED]
        open_t = [t for t in trades if t.status == TradeStatus.OPEN]
        wins = [t for t in closed if (t.pnl_usd or 0) > 0]
        losses = [t for t in closed if (t.pnl_usd or 0) <= 0]
        pnls = [t.pnl_usd or 0 for t in closed]
        pcts = [t.pnl_pct for t in closed if t.pnl_pct is not None]
        win_rate = (len(wins) / len(closed)) if closed else 0.0
        symbols = sorted({t.symbol for t in trades})
        return KOLStats(
            kol_username=kol,
            total_trades=len(trades),
            closed_trades=len(closed),
            open_trades=len(open_t),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            total_pnl_usd=sum(pnls),
            avg_pnl_pct=(sum(pcts) / len(pcts)) if pcts else 0.0,
            best_trade_pct=max(pcts) if pcts else None,
            worst_trade_pct=min(pcts) if pcts else None,
            symbols=symbols,
        )
