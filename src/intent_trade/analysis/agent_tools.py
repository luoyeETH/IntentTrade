"""Whitelisted market tools exposed to the intent-analysis agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from intent_trade.analysis.ticker_map import TickerMap


class IntentAgentTools:
    def __init__(
        self,
        ticker_map: TickerMap,
        market: Any,
        *,
        default_lookback_days: int = 365,
    ) -> None:
        self.ticker_map = ticker_map
        self.market = market
        self.default_lookback_days = max(7, min(default_lookback_days, 1000))
        self._discovered: dict[str, dict[str, Any]] = {}
        self._verified_symbols: set[str] = set()
        self._cache: dict[str, dict[str, Any]] = {}

    def start_session(self) -> None:
        """Reset per-post verification and quote caches."""

        self._discovered.clear()
        self._verified_symbols.clear()
        self._cache.clear()

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_instruments",
                "description": (
                    "Search Yahoo Finance for a company, ADR, ETF, or crypto symbol. "
                    "Use this before declaring a clearly named asset unmapped."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_market_snapshot",
                "description": (
                    "Get the latest price and source. Binance crypto/bStock is tried "
                    "before yfinance whenever available."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_price_statistics",
                "description": (
                    "Get recent high, low, current price, high date, and drawdown from "
                    "the high for claims such as 'down 40% from the peak'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "lookback_days": {
                            "type": "integer",
                            "minimum": 7,
                            "maximum": 1000,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "register_instrument",
                "description": (
                    "Persist a newly discovered instrument. Call only after search or a "
                    "successful quote/statistics lookup verified the provider symbol."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "name": {"type": "string"},
                        "asset_class": {
                            "type": "string",
                            "enum": ["crypto", "equity", "etf", "other"],
                        },
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "yfinance_symbol": {"type": "string"},
                    },
                    "required": ["symbol", "name", "asset_class", "aliases"],
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        cacheable = name in {
            "search_instruments",
            "get_market_snapshot",
            "get_price_statistics",
        }
        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        if cacheable and cache_key in self._cache:
            return {**self._cache[cache_key], "cached": True}
        try:
            if name == "search_instruments":
                output = self.search_instruments(
                    str(arguments.get("query") or ""),
                    int(arguments.get("limit") or 8),
                )
            elif name == "get_market_snapshot":
                output = self.get_market_snapshot(str(arguments.get("symbol") or ""))
            elif name == "get_price_statistics":
                output = self.get_price_statistics(
                    str(arguments.get("symbol") or ""),
                    int(arguments.get("lookback_days") or self.default_lookback_days),
                )
            elif name == "register_instrument":
                output = self.register_instrument(arguments)
            else:
                output = {"ok": False, "error": f"unknown tool: {name}"}
            if cacheable:
                self._cache[cache_key] = output
            return output
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:500]}

    def search_instruments(self, query: str, limit: int = 8) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"ok": False, "error": "query is required", "results": []}
        limit = max(1, min(limit, 10))
        import yfinance as yf

        search = yf.Search(query, max_results=limit, news_count=0, raise_errors=True)
        results: list[dict[str, Any]] = []
        for row in getattr(search, "quotes", []) or []:
            provider_symbol = str(row.get("symbol") or "").strip().upper()
            quote_type = str(row.get("quoteType") or "").strip().upper()
            if not provider_symbol or quote_type not in {
                "EQUITY",
                "ETF",
                "CRYPTOCURRENCY",
            }:
                continue
            asset_class = (
                "crypto"
                if quote_type == "CRYPTOCURRENCY"
                else "etf"
                if quote_type == "ETF"
                else "equity"
            )
            canonical = self.ticker_map.canonicalize_symbol(
                provider_symbol, asset_class
            )
            candidate = {
                "symbol": canonical,
                "provider_symbol": provider_symbol,
                "name": str(
                    row.get("longname")
                    or row.get("shortname")
                    or row.get("name")
                    or provider_symbol
                ),
                "asset_class": asset_class,
                "exchange": str(row.get("exchDisp") or row.get("exchange") or ""),
                "quote_type": quote_type,
            }
            results.append(candidate)
            self._discovered[canonical.upper()] = candidate
            self._discovered[provider_symbol.upper()] = candidate
        return {"ok": True, "query": query, "results": results[:limit]}

    def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        canonical = self.ticker_map.resolve(symbol) or symbol.strip().upper()
        if not canonical:
            return {"ok": False, "error": "symbol is required"}
        snapshot = self.market.get_current_snapshot(
            canonical,
            allow_fallback=False,
            force=True,
        )
        if snapshot.price is not None and snapshot.price > 0:
            self._verified_symbols.add(canonical.upper())
        return {
            "ok": snapshot.price is not None,
            "symbol": canonical,
            "price": snapshot.price,
            "timestamp": snapshot.ts.isoformat() if snapshot.ts else None,
            "source": snapshot.source,
            "is_live": snapshot.is_live,
            "stale": snapshot.stale,
            "currency": snapshot.currency,
            "previous_close": snapshot.previous_close,
            "error": snapshot.error,
        }

    def get_price_statistics(
        self, symbol: str, lookback_days: int = 365
    ) -> dict[str, Any]:
        canonical = self.ticker_map.resolve(symbol) or symbol.strip().upper()
        if not canonical:
            return {"ok": False, "error": "symbol is required"}
        days = max(7, min(int(lookback_days), 1000))
        # Trading calendars have fewer bars than calendar days. Request enough
        # history, then apply an exact calendar-day cutoff below.
        requested_bars = min(max(days + 30, int(days * 1.6)), 1000)
        result = self.market.get_klines(
            canonical,
            interval="1d",
            limit=requested_bars,
            force=True,
        )
        all_bars = result.bars
        if all_bars:
            cutoff = all_bars[-1].ts - timedelta(days=days)
            bars = [bar for bar in all_bars if bar.ts >= cutoff]
        else:
            bars = []
        if not bars:
            return {
                "ok": False,
                "symbol": canonical,
                "error": result.error or "no daily bars returned",
                "source": result.source,
            }
        high_bar = max(bars, key=lambda item: item.high)
        low_bar = min(bars, key=lambda item: item.low)
        snapshot = self.market.get_current_snapshot(
            canonical,
            allow_fallback=False,
            force=True,
        )
        current = snapshot.price if snapshot.price is not None else bars[-1].close
        drawdown = ((current / high_bar.high) - 1) * 100 if high_bar.high else None
        rebound = ((current / low_bar.low) - 1) * 100 if low_bar.low else None
        self._verified_symbols.add(canonical.upper())
        return {
            "ok": True,
            "symbol": canonical,
            "lookback_days": days,
            "bars": len(bars),
            "period_start_at": bars[0].ts.isoformat(),
            "period_end_at": bars[-1].ts.isoformat(),
            "current_price": current,
            "current_timestamp": (
                snapshot.ts.isoformat() if snapshot.ts else bars[-1].ts.isoformat()
            ),
            "period_high": high_bar.high,
            "period_high_at": high_bar.ts.isoformat(),
            "period_low": low_bar.low,
            "period_low_at": low_bar.ts.isoformat(),
            "drawdown_from_high_pct": drawdown,
            "rebound_from_low_pct": rebound,
            "source": result.source,
            "provider_symbol": result.provider_symbol,
            "is_live": result.is_live,
            "current_source": snapshot.source,
            "current_currency": snapshot.currency,
        }

    def register_instrument(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested = str(arguments.get("symbol") or "").strip().upper()
        yf_symbol = str(arguments.get("yfinance_symbol") or requested).strip().upper()
        asset_class = str(arguments.get("asset_class") or "other").strip().lower()
        canonical = self.ticker_map.canonicalize_symbol(requested, asset_class)
        candidate = (
            self._discovered.get(canonical.upper())
            or self._discovered.get(yf_symbol.upper())
            or self._discovered.get(requested.upper())
        )
        verified = bool(candidate) or canonical.upper() in self._verified_symbols
        if not verified:
            return {
                "ok": False,
                "error": (
                    "instrument is not verified; call search_instruments or a price "
                    "lookup before registration"
                ),
                "symbol": canonical,
            }
        if candidate:
            asset_class = str(candidate["asset_class"])
            canonical = str(candidate["symbol"])
            yf_symbol = str(candidate["provider_symbol"])
        aliases = [
            str(value).strip()
            for value in (arguments.get("aliases") or [])
            if str(value).strip()
        ]
        name = str(arguments.get("name") or "").strip() or str(
            (candidate or {}).get("name") or canonical
        )
        meta = self.ticker_map.register_instrument(
            canonical,
            name=name,
            asset_class=asset_class,
            aliases=aliases,
            yfinance_symbol=yf_symbol,
            reason="agent_tool_verified",
            persist=True,
        )
        self.market.register_instrument(
            meta.symbol,
            asset_class=meta.asset_class,
            yfinance_symbol=meta.yfinance_symbol or meta.symbol,
        )
        return {
            "ok": True,
            "symbol": meta.symbol,
            "name": meta.name,
            "asset_class": meta.asset_class,
            "aliases": meta.aliases,
            "yfinance_symbol": meta.yfinance_symbol,
            "registered_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
