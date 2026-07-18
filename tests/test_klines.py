"""K-line providers + /api/klines endpoint."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from intent_trade.market.klines import (
    SUPPORTED_INTERVALS,
    KlineProvider,
    KlineResult,
    crypto_base_from_canonical,
    is_crypto_symbol,
    normalize_interval,
)
from intent_trade.models.domain import MarketBar
from intent_trade.web.app import app


def test_normalize_interval_aliases():
    assert normalize_interval("1D") == "1d"
    assert normalize_interval("1wk") == "1w"
    assert normalize_interval("60m") == "1h"
    assert normalize_interval("1day") == "1d"
    with pytest.raises(ValueError):
        normalize_interval("3m")


def test_proxy_helpers_redact_credentials(monkeypatch):
    from intent_trade.market.klines import proxy_label, resolve_http_proxy

    monkeypatch.delenv("INTENT_TRADE_HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)

    assert resolve_http_proxy(None) is None
    monkeypatch.setenv(
        "INTENT_TRADE_HTTP_PROXY", "socks5h://user:secret@proxy.example:1080"
    )
    resolved = resolve_http_proxy()
    assert resolved and "secret" in resolved
    label = proxy_label(resolved)
    assert "secret" not in label
    assert "proxy.example" in label
    assert label.startswith("socks5h://")


def test_crypto_detection():
    assert crypto_base_from_canonical("BTC-USD") == "BTC"
    assert crypto_base_from_canonical("ETH-USDT") == "ETH"
    assert crypto_base_from_canonical("SOLUSDT") == "SOL"
    assert is_crypto_symbol("BTC-USD", "crypto") is True
    assert is_crypto_symbol("TSLA", "equity") is False
    assert is_crypto_symbol("SNDK", "equity") is False


def test_binance_parse_mocked():
    provider = KlineProvider(cache_ttl_seconds=0)
    fake_rows = [
        [
            1_700_000_000_000,
            "100",
            "110",
            "90",
            "105",
            "12.5",
            1_700_000_060_000,
            "0",
            0,
            "0",
            "0",
            "0",
        ],
        [
            1_700_086_400_000,
            "105",
            "120",
            "100",
            "115",
            "20",
            1_700_172_799_999,
            "0",
            0,
            "0",
            "0",
            "0",
        ],
    ]

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return fake_rows

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            assert "klines" in url
            assert params["symbol"] == "BTCUSDT"
            return FakeResp()

    with patch("intent_trade.market.klines.httpx.Client", FakeClient):
        bars, psym = provider._binance_pairs(
            "BTC-USD", "1d", 10, ["BTCUSDT"]
        )
    assert psym == "BTCUSDT"
    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[-1].close == 115.0


def test_okx_parse_mocked():
    provider = KlineProvider(cache_ttl_seconds=0)
    # OKX newest-first
    fake_payload = {
        "code": "0",
        "msg": "",
        "data": [
            ["1700086400000", "105", "120", "100", "115", "20", "0", "0", "1"],
            ["1700000000000", "100", "110", "90", "105", "12.5", "0", "0", "1"],
        ],
    }

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return fake_payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            assert "candles" in url
            return FakeResp()

    with patch("intent_trade.market.klines.httpx.Client", FakeClient):
        bars, psym = provider._okx("BTC-USD", "1d", 10)
    assert psym == "BTC-USDT"
    assert len(bars) == 2
    assert bars[0].ts < bars[1].ts
    assert bars[0].open == 100.0


def test_bstock_pair_candidates():
    from intent_trade.market.klines import bstock_pair_candidates

    assert "SNDKBUSDT" in bstock_pair_candidates("SNDK")
    assert "TSLABUSDT" in bstock_pair_candidates("TSLA")
    assert "NVDABUSDT" in bstock_pair_candidates("NVDA")


def test_get_klines_equity_prefers_bstock_then_yfinance():
    provider = KlineProvider(cache_ttl_seconds=0)
    sample = [
        MarketBar(
            symbol="TSLA",
            ts=datetime(2024, 1, 2),
            open=200,
            high=210,
            low=195,
            close=205,
            volume=1e6,
        )
    ]
    # bStock hit
    with patch.object(
        provider, "_binance_pairs", return_value=(sample, "TSLABUSDT")
    ) as bn:
        with patch.object(provider, "_yfinance") as yf:
            result = provider.get_klines(
                "TSLA", interval="1d", limit=50, asset_class="equity"
            )
    bn.assert_called_once()
    yf.assert_not_called()
    assert result.source == "binance_bstock"
    assert result.provider_symbol == "TSLABUSDT"

    # bStock miss → yfinance
    with patch.object(provider, "_binance_pairs", return_value=([], "")) as bn2:
        with patch.object(provider, "_yfinance", return_value=(sample, "TSLA")) as yf2:
            result2 = provider.get_klines(
                "TSLA", interval="1d", limit=50, asset_class="equity", force=True
            )
    yf2.assert_called_once()
    assert result2.source == "yfinance"


def test_api_klines_mocked(monkeypatch):
    """Hit FastAPI route with a stubbed Pipeline.market.get_klines."""

    bars = [
        MarketBar(
            symbol="BTC-USD",
            ts=datetime(2024, 6, 1, 0, 0, 0),
            open=1,
            high=2,
            low=0.5,
            close=1.5,
            volume=10,
        )
    ]
    fake_result = KlineResult(
        symbol="BTC-USD",
        interval="1d",
        bars=bars,
        source="binance",
        provider_symbol="BTCUSDT",
        is_live=True,
    )

    class FakeTickerMap:
        def resolve(self, s):
            return s if s else None

        def asset_class_of(self, s):
            return "crypto"

    class FakeMarket:
        def get_klines(self, symbol, interval="1d", limit=300, force=False):
            return fake_result

        def get_current_snapshot(self, symbol):
            from intent_trade.models.domain import MarketSnapshot

            return MarketSnapshot(
                symbol=symbol, price=1.5, source="test", is_live=True
            )

    class FakeStorage:
        def symbol_snapshot(self, symbol, timezone_name="Asia/Shanghai"):
            return {
                "symbol": symbol,
                "structured_signals": [],
                "notes": [],
                "trades": [],
            }

        def get_posts_by_ids(self, post_ids):
            return {}

    class FakeSettings:
        class app:
            timezone = "Asia/Shanghai"

        class market:
            kline_default_limit = 300

    class FakePipe:
        settings = FakeSettings()
        ticker_map = FakeTickerMap()
        market = FakeMarket()
        storage = FakeStorage()

    monkeypatch.setattr("intent_trade.web.app._pipe", lambda: FakePipe())
    client = TestClient(app)
    r = client.get("/api/klines/BTC-USD?interval=1d&limit=50")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "BTC-USD"
    assert body["source"] == "binance"
    assert body["interval"] == "1d"
    assert body["count"] == 1
    assert body["bars"][0]["close"] == 1.5
    assert "time" in body["bars"][0]
    assert set(body["intervals"]) == set(SUPPORTED_INTERVALS)


def test_api_klines_bad_interval(monkeypatch):
    class FakePipe:
        class ticker_map:
            @staticmethod
            def resolve(s):
                return s

            @staticmethod
            def asset_class_of(s):
                return "crypto"

        class settings:
            class app:
                timezone = "Asia/Shanghai"

        market = MagicMock()
        storage = MagicMock()

    monkeypatch.setattr("intent_trade.web.app._pipe", lambda: FakePipe())
    client = TestClient(app)
    r = client.get("/api/klines/BTC-USD?interval=3m")
    assert r.status_code == 400


def test_api_klines_can_skip_duplicate_quote(monkeypatch):
    """Timeline chart loads can rely on /api/symbol for the quote."""

    from intent_trade.web import app as web_app

    bars = [
        MarketBar(
            symbol="BTC-USD",
            ts=datetime(2024, 6, 1, 0, 0, 0),
            open=1,
            high=2,
            low=0.5,
            close=1.5,
            volume=10,
        )
    ]
    fake_result = KlineResult(
        symbol="BTC-USD",
        interval="1d",
        bars=bars,
        source="test",
        is_live=True,
    )

    class FakeTickerMap:
        def resolve(self, symbol):
            return symbol

        def asset_class_of(self, symbol):
            return "crypto"

    class FakeMarket:
        quote_calls = 0

        def get_klines(self, symbol, interval="1d", limit=300, force=False):
            return fake_result

        def get_current_snapshot(self, symbol):
            self.quote_calls += 1
            raise AssertionError("quote should be skipped")

    class FakeStorage:
        def symbol_snapshot(self, symbol, timezone_name="Asia/Shanghai"):
            return {"symbol": symbol, "structured_signals": [], "notes": [], "trades": []}

        def get_posts_by_ids(self, post_ids):
            return {}

    class FakeSettings:
        class app:
            timezone = "Asia/Shanghai"

        class market:
            kline_default_limit = 300

    class FakePipe:
        settings = FakeSettings()
        ticker_map = FakeTickerMap()
        market = FakeMarket()
        storage = FakeStorage()

    monkeypatch.setattr(web_app, "_pipe", lambda: FakePipe())
    payload = web_app.klines_view(
        "BTC-USD",
        interval="1d",
        limit=300,
        markers=True,
        include_quote=False,
    )

    assert payload["quote"] is None


def test_web_pipeline_is_reused(monkeypatch):
    from intent_trade.web import app as web_app

    created = []

    def fake_pipeline(settings):
        instance = object()
        created.append(instance)
        return instance

    monkeypatch.setattr(web_app, "_PIPELINE", None)
    monkeypatch.setattr(web_app, "load_settings", lambda: object())
    monkeypatch.setattr(web_app, "Pipeline", fake_pipeline)

    first = web_app._pipe()
    second = web_app._pipe()

    assert first is second
    assert len(created) == 1


def test_static_has_chart_hooks():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    static = root / "src" / "intent_trade" / "web" / "static"
    frontend = root / "frontend" / "src"
    html = (static / "index.html").read_text(encoding="utf-8")
    styles = (frontend / "styles.css").read_text(encoding="utf-8")
    app_jsx = (frontend / "App.jsx").read_text(encoding="utf-8")
    # SPA shell + chart vendor
    assert "lightweight-charts" in html
    assert 'id="root"' in html
    assert "/static/assets/" in html
    # React chart surface
    assert ".kline-chart" in styles
    assert "KlineChart" in app_jsx
    # default 15m interval in UI config
    assert 'value: "15m"' in app_jsx or "value: '15m'" in app_jsx
    # no debug health dump strings
    assert "source=rapidapi" not in app_jsx
    assert "auto_poll=ok" not in app_jsx


def test_binance_quote_preferred_for_equity(monkeypatch):
    from intent_trade.config import load_settings
    from intent_trade.market.prices import MarketDataService
    from intent_trade.models.domain import MarketSnapshot

    s = load_settings()
    m = MarketDataService(
        s,
        yf_symbol_map={"SNDK": "SNDK"},
        asset_class_map={"SNDK": "equity"},
        prefer_fallback=False,
    )

    def fake_price(symbol, asset_class=None, yfinance_symbol=None):
        return 1345.5, "SNDKBUSDT", "binance_bstock"

    monkeypatch.setattr(m.klines, "get_last_price", fake_price)
    monkeypatch.setattr(m, "_fetch_yfinance_quote", lambda symbol: None)
    snap = m.get_current_snapshot("SNDK", force=True)
    assert snap.price == 1345.5
    assert snap.source == "binance_bstock"
    assert snap.is_live is True
