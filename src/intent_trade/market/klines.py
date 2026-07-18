"""Multi-source K-line (candlestick) providers.

Priority (crypto):
  1. Binance public data API (``data-api.binance.vision``) — geo-friendly
  2. OKX public candles
  3. yfinance

Equities / ETFs: yfinance only (real US/A-share quotes). Optional tokenized
stock venues (OKX ``XTSLA-USDT``, Binance ``TSLABUSDT``) are not used for
canonical equity symbols — they track synthetics, not the cash stock.

Binance.com main API returns HTTP 451 from some regions; the data-api host
is used deliberately for market data only (no trading).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urljoin

import httpx

from intent_trade.models.domain import MarketBar

log = logging.getLogger("intent_trade.market.klines")


def _utc_from_ts(seconds: float | int) -> datetime:
    """Naive UTC datetime (matches rest of codebase storage convention)."""
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).replace(tzinfo=None)


def resolve_http_proxy(explicit: str | None = None) -> Optional[str]:
    """Return SOCKS/HTTP proxy URL for market HTTP clients.

    Order: explicit arg → INTENT_TRADE_HTTP_PROXY → HTTPS_PROXY → ALL_PROXY → HTTP_PROXY.
    Prefer ``socks5h://`` so DNS resolves on the proxy side.
    """

    for key in (
        explicit,
        os.getenv("INTENT_TRADE_HTTP_PROXY"),
        os.getenv("HTTPS_PROXY"),
        os.getenv("https_proxy"),
        os.getenv("ALL_PROXY"),
        os.getenv("all_proxy"),
        os.getenv("HTTP_PROXY"),
        os.getenv("http_proxy"),
    ):
        if key and str(key).strip():
            return str(key).strip()
    return None


def proxy_label(proxy_url: str | None) -> str:
    """Safe log label without credentials."""

    if not proxy_url:
        return "direct"
    try:
        from urllib.parse import urlparse

        p = urlparse(proxy_url)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        scheme = p.scheme or "proxy"
        return f"{scheme}://{host}{port}"
    except Exception:
        return "proxy"

# Canonical UI/API intervals → provider-specific codes
SUPPORTED_INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d", "1w")

BINANCE_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}
OKX_BAR = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}
YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "60m",  # yfinance has no 4h; resample later if needed
    "1d": "1d",
    "1w": "1wk",
}
# How much history to request per interval for a given limit (rough upper bound)
YF_PERIOD_FOR_INTERVAL = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "max",
    "1w": "max",
}


@dataclass(frozen=True)
class KlineResult:
    symbol: str
    interval: str
    bars: list[MarketBar]
    source: str
    provider_symbol: str = ""
    is_live: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.bars) and not self.error


def normalize_interval(interval: str | None) -> str:
    raw = (interval or "1d").strip().lower()
    aliases = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "60m": "1h",
        "60min": "1h",
        "1hour": "1h",
        "4hour": "4h",
        "1day": "1d",
        "d": "1d",
        "day": "1d",
        "1week": "1w",
        "week": "1w",
        "1wk": "1w",
        "wk": "1w",
    }
    raw = aliases.get(raw, raw)
    if raw not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"unsupported interval {interval!r}; use one of {', '.join(SUPPORTED_INTERVALS)}"
        )
    return raw


def crypto_base_from_canonical(symbol: str) -> Optional[str]:
    """``BTC-USD`` / ``BTCUSDT`` / ``BTC`` → ``BTC`` for known crypto forms."""

    s = symbol.strip().upper().replace("/", "-")
    if s.endswith("-USD"):
        return s[: -len("-USD")] or None
    if s.endswith("-USDT"):
        return s[: -len("-USDT")] or None
    if s.endswith("USDT") and len(s) > 4:
        return s[: -len("USDT")]
    if s.endswith("USD") and len(s) > 3 and s not in {"USD"}:
        # avoid treating equities; only pure crypto bases without dots
        base = s[: -len("USD")]
        if base.isalpha() and base not in {"TSLA", "AAPL", "NVDA", "SNDK", "SPY", "QQQ"}:
            # Heuristic only used when asset_class is crypto; callers should prefer asset_class.
            return base
    if s in {"BTC", "ETH", "SOL", "XBT", "BNB", "XRP", "DOGE", "ADA", "AVAX"}:
        return "BTC" if s == "XBT" else s
    return None


def is_crypto_symbol(symbol: str, asset_class: str | None = None) -> bool:
    if asset_class and asset_class.lower() == "crypto":
        return True
    s = symbol.strip().upper()
    if s.endswith("-USD") and crypto_base_from_canonical(s):
        # Equity never uses -USD in our registry; crypto does.
        return "." not in s and not s[0].isdigit()
    return crypto_base_from_canonical(s) is not None and s in {
        "BTC",
        "ETH",
        "SOL",
        "XBT",
        "BNB",
        "XRP",
        "DOGE",
        "ADA",
        "AVAX",
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }


def is_equity_like(symbol: str, asset_class: str | None = None) -> bool:
    if asset_class and asset_class.lower() in {"equity", "etf", "stock"}:
        return True
    if is_crypto_symbol(symbol, asset_class):
        return False
    s = symbol.strip().upper()
    # A-shares like 600519.SS — not on Binance bStocks
    if "." in s:
        return True
    return bool(s) and s not in {"UNKNOWN", "N/A"}


def bstock_pair_candidates(symbol: str, yfinance_symbol: str | None = None) -> list[str]:
    """Map equity ticker → Binance bStock USDT pairs.

    Rule from product: append ``B`` before the quote, e.g. ``SNDK`` → ``SNDKBUSDT``.
    Also try a few aliases (already-suffixed forms, yfinance root).
    """

    roots: list[str] = []
    for raw in (symbol, yfinance_symbol or ""):
        s = str(raw or "").strip().upper().replace("/", "")
        if not s or s in {"UNKNOWN", "N/A"}:
            continue
        # drop exchange suffixes (.SS/.SZ/.HK) — bStocks are US-style roots only
        if "." in s:
            s = s.split(".", 1)[0]
        if s.endswith("-USD"):
            continue  # crypto form
        if s.endswith("USDT"):
            roots.append(s[: -len("USDT")])
            continue
        roots.append(s)

    pairs: list[str] = []
    for root in roots:
        # strip trailing B if user already passed SNDKB
        base = root[:-1] if root.endswith("B") and len(root) > 2 else root
        for candidate in (
            f"{base}BUSDT",  # SNDK → SNDKBUSDT
            f"{root}USDT",  # if root already SNDKB
            f"{base}BUSD",
        ):
            if candidate not in pairs:
                pairs.append(candidate)
    return pairs

def _coerce_ts_ms(ms: int | float | str) -> datetime:
    return _utc_from_ts(int(ms) / 1000.0)


def _bar(
    symbol: str,
    ts: datetime,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 0.0,
) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        ts=ts,
        open=float(o),
        high=float(h),
        low=float(l),
        close=float(c),
        volume=float(v or 0),
    )


class KlineProvider:
    """Fetch OHLCV bars for dashboard charts (not used for paper fill path)."""

    def __init__(
        self,
        *,
        binance_base_url: str = "https://data-api.binance.vision",
        okx_base_url: str = "https://www.okx.com",
        timeout: float = 12.0,
        cache_ttl_seconds: int = 30,
        preferred_crypto: str = "binance",
        proxy: str | None = None,
        use_proxy_for_okx: bool = False,
    ) -> None:
        self.binance_base_url = binance_base_url.rstrip("/") + "/"
        self.okx_base_url = okx_base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.cache_ttl_seconds = max(0, cache_ttl_seconds)
        self.preferred_crypto = (preferred_crypto or "binance").lower()
        self.proxy = resolve_http_proxy(proxy)
        # OKX is usually reachable direct; proxy only if explicitly enabled or forced.
        self.use_proxy_for_okx = use_proxy_for_okx
        self._mem: dict[str, tuple[float, KlineResult]] = {}
        if self.proxy:
            log.info(
                "kline HTTP proxy enabled: %s (binance base %s)",
                proxy_label(self.proxy),
                self.binance_base_url.rstrip("/"),
            )

    def _client(self, *, via_proxy: bool = False) -> httpx.Client:
        kwargs: dict = {"timeout": self.timeout}
        if via_proxy and self.proxy:
            kwargs["proxy"] = self.proxy
        return httpx.Client(**kwargs)

    def get_last_price(
        self,
        symbol: str,
        *,
        asset_class: str | None = None,
        yfinance_symbol: str | None = None,
    ) -> tuple[Optional[float], str, str]:
        """Return (price, provider_symbol, source_tag) via Binance ticker.

        Crypto → BASEUSDT; equities → bStock ``TICKERBUSDT``. Empty on miss.
        """

        crypto = is_crypto_symbol(symbol, asset_class)
        pairs: list[str] = []
        source_tag = "binance"
        if crypto:
            base = crypto_base_from_canonical(symbol)
            if base:
                pairs = [f"{base}USDT"]
        elif is_equity_like(symbol, asset_class):
            pairs = bstock_pair_candidates(symbol, yfinance_symbol)
            source_tag = "binance_bstock"
        if not pairs:
            return None, "", ""

        url = urljoin(self.binance_base_url, "api/v3/ticker/price")
        attempts: list[bool] = [True, False] if self.proxy else [False]
        for psym in pairs:
            for via_proxy in attempts:
                try:
                    with self._client(via_proxy=via_proxy) as client:
                        r = client.get(url, params={"symbol": psym})
                    if r.status_code == 400:
                        break  # try next pair
                    if r.status_code == 451:
                        continue
                    r.raise_for_status()
                    body = r.json()
                    price = float(body.get("price"))
                    if price > 0:
                        return price, psym, source_tag
                except Exception:
                    continue
        return None, "", ""

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        limit: int = 200,
        asset_class: str | None = None,
        yfinance_symbol: str | None = None,
        force: bool = False,
    ) -> KlineResult:
        interval = normalize_interval(interval)
        limit = max(1, min(int(limit), 1000))
        cache_key = f"{symbol}:{interval}:{limit}:{asset_class}:{yfinance_symbol}"
        now = time.time()
        if not force and cache_key in self._mem:
            ts, cached = self._mem[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return cached

        result = self._fetch(
            symbol,
            interval=interval,
            limit=limit,
            asset_class=asset_class,
            yfinance_symbol=yfinance_symbol,
        )
        self._mem[cache_key] = (now, result)
        return result

    def _fetch(
        self,
        symbol: str,
        *,
        interval: str,
        limit: int,
        asset_class: str | None,
        yfinance_symbol: str | None,
    ) -> KlineResult:
        crypto = is_crypto_symbol(symbol, asset_class)
        equity = (not crypto) and is_equity_like(symbol, asset_class)
        errors: list[str] = []

        if crypto:
            order = (
                ("binance", "okx", "yfinance")
                if self.preferred_crypto != "okx"
                else ("okx", "binance", "yfinance")
            )
            for name in order:
                try:
                    if name == "binance":
                        base = crypto_base_from_canonical(symbol)
                        pairs = [f"{base}USDT"] if base else []
                        bars, psym = self._binance_pairs(
                            symbol, interval, limit, pairs
                        )
                        if bars:
                            return KlineResult(
                                symbol=symbol,
                                interval=interval,
                                bars=bars[-limit:],
                                source="binance",
                                provider_symbol=psym,
                                is_live=True,
                            )
                    elif name == "okx":
                        bars, psym = self._okx(symbol, interval, limit)
                        if bars:
                            return KlineResult(
                                symbol=symbol,
                                interval=interval,
                                bars=bars[-limit:],
                                source="okx",
                                provider_symbol=psym,
                                is_live=True,
                            )
                    else:
                        bars, psym = self._yfinance(
                            symbol,
                            interval,
                            limit,
                            yfinance_symbol=yfinance_symbol or symbol,
                        )
                        if bars:
                            return KlineResult(
                                symbol=symbol,
                                interval=interval,
                                bars=bars[-limit:],
                                source="yfinance",
                                provider_symbol=psym,
                                is_live=True,
                            )
                except Exception as exc:
                    msg = f"{name}: {exc}"[:200]
                    errors.append(msg)
                    log.debug("kline provider %s failed for %s: %s", name, symbol, exc)
        elif equity:
            # US equities / ETFs: Binance bStocks (TICKER + B + USDT) first, then yfinance
            try:
                pairs = bstock_pair_candidates(symbol, yfinance_symbol)
                bars, psym = self._binance_pairs(symbol, interval, limit, pairs)
                if bars:
                    return KlineResult(
                        symbol=symbol,
                        interval=interval,
                        bars=bars[-limit:],
                        source="binance_bstock",
                        provider_symbol=psym,
                        is_live=True,
                    )
            except Exception as exc:
                errors.append(f"binance_bstock: {exc}"[:200])
                log.debug("bstock failed for %s: %s", symbol, exc)
            try:
                bars, psym = self._yfinance(
                    symbol,
                    interval,
                    limit,
                    yfinance_symbol=yfinance_symbol or symbol,
                )
                if bars:
                    return KlineResult(
                        symbol=symbol,
                        interval=interval,
                        bars=bars[-limit:],
                        source="yfinance",
                        provider_symbol=psym,
                        is_live=True,
                    )
            except Exception as exc:
                errors.append(f"yfinance: {exc}"[:200])
        else:
            try:
                bars, psym = self._yfinance(
                    symbol,
                    interval,
                    limit,
                    yfinance_symbol=yfinance_symbol or symbol,
                )
                if bars:
                    return KlineResult(
                        symbol=symbol,
                        interval=interval,
                        bars=bars[-limit:],
                        source="yfinance",
                        provider_symbol=psym,
                        is_live=True,
                    )
            except Exception as exc:
                errors.append(f"yfinance: {exc}"[:200])

        return KlineResult(
            symbol=symbol,
            interval=interval,
            bars=[],
            source="unavailable",
            is_live=False,
            error="; ".join(errors) if errors else "no klines returned",
        )

    def _binance_pairs(
        self,
        symbol: str,
        interval: str,
        limit: int,
        pairs: list[str],
    ) -> tuple[list[MarketBar], str]:
        """Try one or more Binance symbols (crypto USDT or bStock *BUSDT)."""

        if not pairs:
            return [], ""
        url = urljoin(self.binance_base_url, "api/v3/klines")
        last_err: Optional[Exception] = None
        # Prefer proxy for Binance main host; vision host works direct too.
        attempts: list[bool] = [True, False] if self.proxy else [False]
        for psym in pairs:
            params = {
                "symbol": psym,
                "interval": BINANCE_INTERVAL[interval],
                "limit": str(limit),
            }
            data = None
            for via_proxy in attempts:
                try:
                    with self._client(via_proxy=via_proxy) as client:
                        r = client.get(url, params=params)
                    if r.status_code == 451:
                        last_err = RuntimeError("binance restricted location (451)")
                        continue
                    if r.status_code == 400:
                        # invalid symbol — try next pair
                        try:
                            body = r.json()
                            last_err = RuntimeError(
                                f"{psym}: {body.get('msg') or body}"
                            )
                        except Exception:
                            last_err = RuntimeError(f"{psym}: HTTP 400")
                        data = None
                        break
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as exc:
                    last_err = exc
                    continue
            if data is None:
                continue
            if isinstance(data, dict) and data.get("code") not in (None, 0, "0"):
                last_err = RuntimeError(str(data.get("msg") or data)[:200])
                continue
            if not isinstance(data, list) or not data:
                continue
            bars: list[MarketBar] = []
            for row in data:
                bars.append(
                    _bar(
                        symbol,
                        _coerce_ts_ms(row[0]),
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                    )
                )
            bars.sort(key=lambda b: b.ts)
            return bars, psym
        if last_err:
            raise last_err
        return [], ""

    def _binance(
        self, symbol: str, interval: str, limit: int
    ) -> tuple[list[MarketBar], str]:
        """Crypto convenience wrapper: SYMBOL-USD → BASEUSDT."""

        base = crypto_base_from_canonical(symbol)
        if not base:
            return [], ""
        return self._binance_pairs(symbol, interval, limit, [f"{base}USDT"])

    def _okx(
        self, symbol: str, interval: str, limit: int
    ) -> tuple[list[MarketBar], str]:
        base = crypto_base_from_canonical(symbol)
        if not base:
            return [], ""
        # Prefer USDT pair; fall back to USD if needed
        candidates = [f"{base}-USDT", f"{base}-USD"]
        url = urljoin(self.okx_base_url, "api/v5/market/candles")
        last_err: Optional[Exception] = None
        with self._client(via_proxy=self.use_proxy_for_okx and bool(self.proxy)) as client:
            for psym in candidates:
                # OKX max 300 per request; paginate if limit > 300
                remaining = limit
                after: Optional[str] = None
                rows: list[list] = []
                while remaining > 0:
                    batch = min(remaining, 300)
                    params: dict[str, str] = {
                        "instId": psym,
                        "bar": OKX_BAR[interval],
                        "limit": str(batch),
                    }
                    if after:
                        params["after"] = after
                    r = client.get(url, params=params)
                    r.raise_for_status()
                    payload = r.json()
                    if str(payload.get("code")) != "0":
                        last_err = RuntimeError(
                            f"{psym}: {payload.get('msg') or payload}"
                        )
                        rows = []
                        break
                    chunk = payload.get("data") or []
                    if not chunk:
                        break
                    rows.extend(chunk)
                    # OKX returns newest first; `after` = older pagination using ts
                    after = str(chunk[-1][0])
                    remaining -= len(chunk)
                    if len(chunk) < batch:
                        break
                if rows:
                    bars: list[MarketBar] = []
                    for row in rows:
                        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                        bars.append(
                            _bar(
                                symbol,
                                _coerce_ts_ms(row[0]),
                                row[1],
                                row[2],
                                row[3],
                                row[4],
                                row[5],
                            )
                        )
                    # de-dupe by ts, sort ascending
                    by_ts = {b.ts: b for b in bars}
                    ordered = [by_ts[k] for k in sorted(by_ts)]
                    return ordered[-limit:], psym
        if last_err:
            raise last_err
        return [], ""

    def _yfinance(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        yfinance_symbol: str,
    ) -> tuple[list[MarketBar], str]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance not installed") from exc

        yf_interval = YF_INTERVAL[interval]
        period = YF_PERIOD_FOR_INTERVAL[interval]
        ticker = yf.Ticker(yfinance_symbol)
        # Prefer period for short intervals; history days for daily+
        if interval in {"1d", "1w"}:
            end = datetime.utcnow() + timedelta(days=1)
            # ~limit * bar length with buffer
            if interval == "1d":
                start = end - timedelta(days=limit + 30)
            else:
                start = end - timedelta(weeks=limit + 10)
            df = ticker.history(
                start=start, end=end, interval=yf_interval, auto_adjust=True
            )
        else:
            df = ticker.history(period=period, interval=yf_interval, auto_adjust=True)

        if df is None or df.empty:
            return [], yfinance_symbol

        bars: list[MarketBar] = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.replace(tzinfo=None)
            bars.append(
                _bar(
                    symbol,
                    ts,
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    row.get("Volume") or 0,
                )
            )
        bars.sort(key=lambda b: b.ts)

        # Approximate 4h by resampling 1h when requested
        if interval == "4h" and bars:
            bars = _resample_ohlc(bars, hours=4)

        return bars[-limit:], yfinance_symbol


def _resample_ohlc(bars: list[MarketBar], *, hours: int) -> list[MarketBar]:
    if not bars:
        return []
    bucket_sec = hours * 3600
    groups: dict[int, list[MarketBar]] = {}
    for b in bars:
        key = int(b.ts.timestamp()) // bucket_sec * bucket_sec
        groups.setdefault(key, []).append(b)
    out: list[MarketBar] = []
    for key in sorted(groups):
        chunk = groups[key]
        out.append(
            MarketBar(
                symbol=chunk[0].symbol,
                ts=_utc_from_ts(key),
                open=chunk[0].open,
                high=max(x.high for x in chunk),
                low=min(x.low for x in chunk),
                close=chunk[-1].close,
                volume=sum(x.volume for x in chunk),
            )
        )
    return out
