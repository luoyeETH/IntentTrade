"""Tests: ticker registry + AI path structure (rules only as fallback unit)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_trade.analysis.intent import IntentAnalyzer
from intent_trade.analysis.agent_tools import IntentAgentTools
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.config import AnalysisConfig
from intent_trade.market.klines import KlineResult
from intent_trade.models.domain import Direction, MarketBar, MarketSnapshot, SocialPost


def test_daping_maps_to_btc():
    tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml")
    assert tm.resolve("大饼") == "BTC-USD"
    assert tm.resolve("比特币") == "BTC-USD"
    assert tm.resolve("闪迪") == "SNDK"


def test_learn_alias_persists(tmp_path: Path | None = None):
    learned = ROOT / "config" / "ticker_aliases.learned.yaml"
    # use isolated learned file
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "learned.yaml"
        tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml", learned_path=p)
        tm.learn_alias("大饼饼", "BTC-USD", reason="unit-test", persist=True)
        assert tm.resolve("大饼饼") == "BTC-USD"
        tm2 = TickerMap(ROOT / "config" / "ticker_aliases.yaml", learned_path=p)
        assert tm2.resolve("大饼饼") == "BTC-USD"


def test_rule_fallback_does_not_crash():
    tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml")
    az = IntentAnalyzer(tm, AnalysisConfig(mode="rule_based"))
    post = SocialPost(
        id="t1",
        author_username="k",
        text="大饼看到7万",
        created_at=datetime(2026, 7, 18),
    )
    a = az.analyze(post)
    # fallback may be descriptive; mapping via catalog still works for 大饼 if in map
    assert "BTC-USD" in (a.canonical_symbols or tm.find_in_text(post.text))


def test_agent_tools_discover_register_and_measure_drawdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    learned = tmp_path / "learned.yaml"
    ticker_map = TickerMap(
        ROOT / "config" / "ticker_aliases.yaml",
        learned_path=learned,
    )

    class FakeSearch:
        quotes = [
            {
                "symbol": "HXSCF",
                "quoteType": "EQUITY",
                "longname": "SK hynix Inc.",
                "exchDisp": "OTC Markets OTCPK",
            }
        ]

    monkeypatch.setattr(yf, "Search", lambda *args, **kwargs: FakeSearch())

    class FakeMarket:
        def __init__(self) -> None:
            self.registered = None

        def register_instrument(self, symbol, *, asset_class, yfinance_symbol):
            self.registered = (symbol, asset_class, yfinance_symbol)

        def get_current_snapshot(self, symbol, **kwargs):
            return MarketSnapshot(
                symbol=symbol,
                price=60,
                source="yfinance",
                is_live=True,
            )

        def get_klines(self, symbol, **kwargs):
            return KlineResult(
                symbol=symbol,
                interval="1d",
                source="yfinance",
                provider_symbol="HXSCF",
                is_live=True,
                bars=[
                    MarketBar(
                        symbol=symbol,
                        ts=datetime(2026, 1, 1),
                        open=95,
                        high=100,
                        low=90,
                        close=95,
                    ),
                    MarketBar(
                        symbol=symbol,
                        ts=datetime(2026, 7, 1),
                        open=65,
                        high=68,
                        low=55,
                        close=60,
                    ),
                ],
            )

    market = FakeMarket()
    tools = IntentAgentTools(ticker_map, market)
    search = tools.execute("search_instruments", {"query": "SK Hynix"})
    assert search["results"][0]["symbol"] == "HXSCF"

    registered = tools.execute(
        "register_instrument",
        {
            "symbol": "HXSCF",
            "name": "SK hynix Inc.",
            "asset_class": "equity",
            "aliases": ["海力士", "SK Hynix"],
            "yfinance_symbol": "HXSCF",
        },
    )
    assert registered["ok"] is True
    assert ticker_map.resolve("海力士") == "HXSCF"
    assert market.registered == ("HXSCF", "equity", "HXSCF")

    stats = tools.execute(
        "get_price_statistics",
        {"symbol": "海力士", "lookback_days": 365},
    )
    assert stats["period_high"] == 100
    assert stats["current_price"] == 60
    assert stats["drawdown_from_high_pct"] == -40
    assert stats["current_source"] == "yfinance"
    cached_stats = tools.execute(
        "get_price_statistics",
        {"symbol": "海力士", "lookback_days": 365},
    )
    assert cached_stats["cached"] is True

    reloaded = TickerMap(
        ROOT / "config" / "ticker_aliases.yaml",
        learned_path=learned,
    )
    assert reloaded.resolve("海力士") == "HXSCF"
    assert reloaded.by_symbol["HXSCF"].yfinance_symbol == "HXSCF"


def test_price_statistics_uses_calendar_days(tmp_path: Path) -> None:
    ticker_map = TickerMap(
        ROOT / "config" / "ticker_aliases.yaml",
        learned_path=tmp_path / "learned.yaml",
    )

    class FakeMarket:
        def get_current_snapshot(self, symbol, **kwargs):
            return MarketSnapshot(symbol=symbol, price=80, source="yfinance")

        def get_klines(self, symbol, **kwargs):
            return KlineResult(
                symbol=symbol,
                interval="1d",
                source="yfinance",
                bars=[
                    MarketBar(
                        symbol=symbol,
                        ts=datetime(2025, 1, 1),
                        open=190,
                        high=200,
                        low=180,
                        close=190,
                    ),
                    MarketBar(
                        symbol=symbol,
                        ts=datetime(2026, 1, 1),
                        open=95,
                        high=100,
                        low=90,
                        close=95,
                    ),
                    MarketBar(
                        symbol=symbol,
                        ts=datetime(2026, 7, 1),
                        open=85,
                        high=90,
                        low=75,
                        close=80,
                    ),
                ],
            )

    stats = IntentAgentTools(ticker_map, FakeMarket()).execute(
        "get_price_statistics",
        {"symbol": "HXSCF", "lookback_days": 365},
    )

    assert stats["period_high"] == 100
    assert stats["period_start_at"].startswith("2026-01-01")


if __name__ == "__main__":
    test_daping_maps_to_btc()
    test_learn_alias_persists()
    test_rule_fallback_does_not_crash()
    print("all tests passed")
