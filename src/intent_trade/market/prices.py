"""Market price data via yfinance + optional local cache/fallback.

K-line charts use :class:`~intent_trade.market.klines.KlineProvider`
(Binance data-api / OKX / yfinance) via :meth:`get_klines`.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta
from typing import Optional

from intent_trade.config import Settings
from intent_trade.market.klines import KlineProvider, KlineResult
from intent_trade.models.domain import MarketBar, MarketSnapshot


class MarketDataService:
    def __init__(
        self,
        settings: Settings,
        yf_symbol_map: dict[str, str] | None = None,
        prefer_fallback: bool | None = None,
        asset_class_map: dict[str, str] | None = None,
    ):
        self.settings = settings
        self.yf_symbol_map = yf_symbol_map or {}
        self.asset_class_map = asset_class_map or {}
        self.cache_dir = settings.data_dir / "cache" / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, tuple[float, list[MarketBar]]] = {}
        self._history_sources: dict[str, str] = {}
        self._quotes: dict[str, tuple[float, MarketSnapshot]] = {}
        self._quote_errors: dict[str, str] = {}
        self.fallback_path = settings.data_dir / "sample" / "price_fallback.json"
        self._fallback: dict[str, list[dict]] | None = None
        # Mock Twitter demos should settle against sample OHLC, not live yfinance levels.
        if prefer_fallback is None:
            prefer_fallback = (settings.twitter.source or "mock").lower() == "mock"
        self.prefer_fallback = prefer_fallback
        m = settings.market
        proxy = str(getattr(m, "http_proxy", "") or "").strip() or None
        self.klines = KlineProvider(
            binance_base_url=getattr(
                m, "binance_data_base_url", "https://data-api.binance.vision"
            ),
            okx_base_url=getattr(m, "okx_base_url", "https://www.okx.com"),
            cache_ttl_seconds=int(getattr(m, "kline_cache_ttl_seconds", 30) or 30),
            preferred_crypto=str(
                getattr(m, "kline_crypto_provider", "binance") or "binance"
            ),
            proxy=proxy,
            use_proxy_for_okx=bool(getattr(m, "kline_proxy_okx", False)),
        )

    def resolve_yf_symbol(self, symbol: str) -> str:
        return self.yf_symbol_map.get(symbol, symbol)

    def asset_class_of(self, symbol: str) -> str:
        return self.asset_class_map.get(symbol, "other")

    def register_instrument(
        self,
        symbol: str,
        *,
        asset_class: str,
        yfinance_symbol: str,
    ) -> None:
        """Make a newly learned instrument immediately available to providers."""

        self.asset_class_map[symbol] = asset_class
        self.yf_symbol_map[symbol] = yfinance_symbol
        self._quotes.pop(symbol, None)

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        limit: int | None = None,
        force: bool = False,
    ) -> KlineResult:
        """OHLCV for charts — multi-source (Binance/OKX/yfinance)."""

        limit = limit or int(
            getattr(self.settings.market, "kline_default_limit", 300) or 300
        )
        return self.klines.get_klines(
            symbol,
            interval=interval,
            limit=limit,
            asset_class=self.asset_class_of(symbol),
            yfinance_symbol=self.resolve_yf_symbol(symbol),
            force=force,
        )

    def get_history(
        self,
        symbol: str,
        days: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> list[MarketBar]:
        days = days or self.settings.market.history_days
        interval = interval or self.settings.market.default_interval
        cache_key = f"{symbol}:{days}:{interval}"
        ttl = self.settings.market.cache_ttl_seconds
        now = time.time()
        if cache_key in self._mem:
            ts, bars = self._mem[cache_key]
            if now - ts < ttl:
                return bars

        bars: list[MarketBar] = []
        source = "unavailable"
        if self.prefer_fallback:
            bars = self._from_fallback(symbol)
            if bars:
                source = "sample_fallback"
        if not bars:
            bars = self._fetch_yfinance(symbol, days, interval)
            if bars:
                source = "yfinance"
        if not bars:
            bars = self._from_disk_cache(symbol)
            if bars:
                source = "yfinance_disk_cache"
        if not bars:
            bars = self._from_fallback(symbol)
            if bars:
                source = "sample_fallback"
        self._mem[cache_key] = (now, bars)
        self._history_sources[cache_key] = source
        return bars

    def history_source(
        self,
        symbol: str,
        days: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> str:
        days = days or self.settings.market.history_days
        interval = interval or self.settings.market.default_interval
        key = f"{symbol}:{days}:{interval}"
        if key not in self._history_sources:
            self.get_history(symbol, days=days, interval=interval)
        return self._history_sources.get(key, "unavailable")

    def history_is_live(
        self,
        symbol: str,
        days: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> bool:
        return self.history_source(symbol, days=days, interval=interval) == "yfinance"

    def get_latest_price(self, symbol: str) -> Optional[float]:
        snapshot = self.get_current_snapshot(symbol)
        return snapshot.price

    def get_current_snapshot(
        self,
        symbol: str,
        *,
        allow_fallback: Optional[bool] = None,
        force: bool = False,
    ) -> MarketSnapshot:
        """Return the latest quote with provenance and freshness metadata.

        Live path: Binance (crypto / bStock equities) → yfinance → cache →
        sample fallback. Fallback is marked non-live so demos stay visible
        without enabling paper fills under require_live_for_execution.
        """

        allow_fallback = (
            self.settings.market.allow_fallback
            if allow_fallback is None
            else allow_fallback
        )
        now = time.time()
        ttl = max(0, self.settings.market.quote_ttl_seconds)
        cached = self._quotes.get(symbol)
        if cached and not force and now - cached[0] < ttl:
            return cached[1]

        snapshot: Optional[MarketSnapshot] = None
        if self.prefer_fallback:
            snapshot = self._snapshot_from_fallback(symbol)
        # Prefer Binance (crypto USDT / equity bStock) when not in mock-fallback mode
        if snapshot is None:
            snapshot = self._fetch_binance_quote(symbol)
        if snapshot is None:
            snapshot = self._fetch_yfinance_quote(symbol)
        if snapshot is None:
            snapshot = self._snapshot_from_disk_cache(symbol)
        if snapshot is None and allow_fallback:
            snapshot = self._snapshot_from_fallback(symbol)
        if snapshot is None:
            snapshot = MarketSnapshot(
                symbol=symbol,
                source="unavailable",
                is_live=False,
                stale=True,
                error=self._quote_errors.get(symbol)
                or "no quote returned by configured market source",
            )
        self._quotes[symbol] = (now, snapshot)
        return snapshot

    def _fetch_binance_quote(self, symbol: str) -> Optional[MarketSnapshot]:
        """Live quote via Binance ticker (crypto) or bStock (*BUSDT for equities)."""

        try:
            price, psym, tag = self.klines.get_last_price(
                symbol,
                asset_class=self.asset_class_of(symbol),
                yfinance_symbol=self.resolve_yf_symbol(symbol),
            )
            if price is None:
                return None
            return self._quote(
                symbol,
                float(price),
                datetime.utcnow(),
                source=tag or "binance",
                is_live=True,
            )
        except Exception as exc:
            self._quote_errors[symbol] = str(exc)[:240]
            return None

    # Short alias for callers that think in terms of quotes.
    get_quote = get_current_snapshot

    def get_price_at_or_after(
        self, symbol: str, when: datetime
    ) -> Optional[MarketBar]:
        bars = self.get_history(symbol)
        for b in bars:
            if b.ts >= when:
                return b
        return bars[-1] if bars else None

    def bars_after(self, symbol: str, when: datetime) -> list[MarketBar]:
        bars = self.get_history(symbol)
        return [b for b in bars if b.ts >= when]

    def _fetch_yfinance(
        self, symbol: str, days: int, interval: str
    ) -> list[MarketBar]:
        yf_sym = self.resolve_yf_symbol(symbol)
        try:
            import yfinance as yf
        except ImportError:
            return []
        try:
            end = datetime.utcnow() + timedelta(days=1)
            start = end - timedelta(days=days + 5)
            ticker = yf.Ticker(yf_sym)
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                return []
            bars: list[MarketBar] = []
            for idx, row in df.iterrows():
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.replace(tzinfo=None)
                bars.append(
                    MarketBar(
                        symbol=symbol,
                        ts=ts,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row.get("Volume") or 0),
                    )
                )
            # disk cache
            cache_file = self.cache_dir / f"{symbol.replace('/', '_')}.json"
            cache_file.write_text(
                json.dumps(
                    [b.model_dump(mode="json") for b in bars],
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            return bars
        except Exception:
            return []

    @staticmethod
    def _coerce_datetime(value: object) -> Optional[datetime]:
        if value is None:
            return None
        try:
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(float(value))
            elif hasattr(value, "to_pydatetime"):
                dt = value.to_pydatetime()
            else:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except (TypeError, ValueError, OverflowError):
            return None

    def _fetch_yfinance_quote(self, symbol: str) -> Optional[MarketSnapshot]:
        """Fetch a current/delayed quote without depending on ``Ticker.info``."""

        yf_sym = self.resolve_yf_symbol(symbol)
        try:
            import yfinance as yf

            ticker = yf.Ticker(yf_sym)
            try:
                fast = ticker.fast_info
            except Exception:
                fast = {}
            get_fast = fast.get if hasattr(fast, "get") else lambda key, default=None: getattr(fast, key, default)
            price = get_fast("last_price")
            bid = get_fast("bid")
            ask = get_fast("ask")
            previous = get_fast("previous_close")
            currency = get_fast("currency")
            trade_ts = self._coerce_datetime(get_fast("last_trade_time"))
            last_price = self._float_or_none(price)
            if last_price is not None:
                ts = trade_ts or datetime.utcnow()
                return self._quote(
                    symbol,
                    last_price,
                    ts,
                    source="yfinance_fast_info",
                    is_live=True,
                    bid=self._float_or_none(bid),
                    ask=self._float_or_none(ask),
                    previous_close=self._float_or_none(previous),
                    currency=str(currency) if currency else None,
                )

            # fast_info can be incomplete for some exchanges. A 1m bar is a
            # better current read than a daily close when it is available.
            intraday = ticker.history(period="1d", interval="1m", auto_adjust=True)
            if intraday is not None and not intraday.empty:
                row = intraday.iloc[-1]
                ts = self._coerce_datetime(intraday.index[-1]) or datetime.utcnow()
                return self._quote(
                    symbol,
                    float(row["Close"]),
                    ts,
                    source="yfinance_1m",
                    is_live=True,
                    previous_close=self._float_or_none(previous),
                    currency=str(currency) if currency else None,
                )

            # Last close is still valuable for display, but must not be called
            # a live quote by the timing engine.
            daily = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if daily is not None and not daily.empty:
                row = daily.iloc[-1]
                ts = self._coerce_datetime(daily.index[-1]) or datetime.utcnow()
                return self._quote(
                    symbol,
                    float(row["Close"]),
                    ts,
                    source="yfinance_daily_close",
                    is_live=False,
                    previous_close=self._float_or_none(previous),
                    currency=str(currency) if currency else None,
                )
        except Exception as exc:
            self._quote_errors[symbol] = str(exc)[:240]
            return None
        return None

    @staticmethod
    def _float_or_none(value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            result = float(value)
            return result if result > 0 and math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    def _quote(
        self,
        symbol: str,
        price: float,
        ts: datetime,
        *,
        source: str,
        is_live: bool,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        previous_close: Optional[float] = None,
        currency: Optional[str] = None,
    ) -> MarketSnapshot:
        self._quote_errors.pop(symbol, None)
        age = max(0.0, (datetime.utcnow() - ts).total_seconds())
        stale = age > max(0, self.settings.market.max_quote_age_seconds)
        return MarketSnapshot(
            symbol=symbol,
            price=price,
            ts=ts,
            source=source,
            is_live=is_live,
            stale=stale,
            age_seconds=age,
            bid=bid,
            ask=ask,
            previous_close=previous_close,
            currency=currency,
        )

    def _snapshot_from_fallback(self, symbol: str) -> Optional[MarketSnapshot]:
        bars = self._from_fallback(symbol)
        if not bars:
            return None
        bar = bars[-1]
        age = max(0.0, (datetime.utcnow() - bar.ts).total_seconds())
        return MarketSnapshot(
            symbol=symbol,
            price=bar.close,
            ts=bar.ts,
            source="sample_fallback",
            is_live=False,
            stale=True,
            age_seconds=age,
            previous_close=bars[-2].close if len(bars) > 1 else None,
            error=self._quote_errors.get(symbol),
        )

    def _snapshot_from_disk_cache(self, symbol: str) -> Optional[MarketSnapshot]:
        """Use previously downloaded yfinance bars for display during outages."""

        bars = self._from_disk_cache(symbol)
        try:
            if not bars:
                return None
            bar = bars[-1]
            age = max(0.0, (datetime.utcnow() - bar.ts).total_seconds())
            return MarketSnapshot(
                symbol=symbol,
                price=bar.close,
                ts=bar.ts,
                source="yfinance_disk_cache",
                is_live=False,
                stale=True,
                age_seconds=age,
                previous_close=bars[-2].close if len(bars) > 1 else None,
                error=self._quote_errors.get(symbol),
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None

    def _from_disk_cache(self, symbol: str) -> list[MarketBar]:
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}.json"
        if not cache_file.exists():
            return []
        try:
            rows = json.loads(cache_file.read_text(encoding="utf-8"))
            bars: list[MarketBar] = []
            for row in rows:
                ts = self._coerce_datetime(row.get("ts"))
                if ts is None:
                    continue
                bars.append(
                    MarketBar(
                        symbol=symbol,
                        ts=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0),
                    )
                )
            return bars
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _load_fallback(self) -> dict[str, list[dict]]:
        if self._fallback is not None:
            return self._fallback
        if not self.fallback_path.exists():
            self._fallback = {}
            return self._fallback
        self._fallback = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        return self._fallback

    def _from_fallback(self, symbol: str) -> list[MarketBar]:
        data = self._load_fallback()
        rows = data.get(symbol) or data.get(self.resolve_yf_symbol(symbol)) or []
        bars: list[MarketBar] = []
        for r in rows:
            ts = r["ts"]
            if isinstance(ts, str):
                ts_dt = datetime.fromisoformat(ts.replace("Z", ""))
            else:
                ts_dt = datetime.utcnow()
            bars.append(
                MarketBar(
                    symbol=symbol,
                    ts=ts_dt,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("volume") or 0),
                )
            )
        return bars
