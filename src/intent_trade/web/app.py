"""FastAPI dashboard — posts, signals, notes, paper trades, KOL stats."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from intent_trade.config import load_settings
from intent_trade.execution.timing import evaluate_signal
from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    PositionState,
    SocialPost,
    TradingSignal,
)
from intent_trade.pipeline.runner import Pipeline
from intent_trade.storage.db import Storage
from intent_trade.web import poller as bg_poller

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("intent_trade.web")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start/stop the background KOL poller with the web process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bg_poller.start()
    try:
        yield
    finally:
        bg_poller.stop()


app = FastAPI(title="IntentTrade", version="0.1.0", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_PIPELINE: Pipeline | None = None
_PIPELINE_LOCK = threading.Lock()


def _pipe() -> Pipeline:
    """Reuse the process pipeline so market caches span API requests."""
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = Pipeline(load_settings())
    return _PIPELINE


def _storage() -> Storage:
    return Storage(load_settings().db_path)


def _display_payload(value: Any, timezone_name: str) -> Any:
    """Serialize datetimes with an explicit display offset for API clients."""

    if isinstance(value, datetime):
        aware = value.replace(tzinfo=ZoneInfo("UTC")) if value.tzinfo is None else value
        return aware.astimezone(ZoneInfo(timezone_name)).isoformat()
    if isinstance(value, list):
        return [_display_payload(item, timezone_name) for item in value]
    if isinstance(value, dict):
        return {
            key: _display_payload(item, timezone_name)
            for key, item in value.items()
        }
    return value


def _model_payload(model: Any, timezone_name: str) -> dict[str, Any]:
    return _display_payload(model.model_dump(mode="python"), timezone_name)


def _refresh_signal_decisions(pipe: Pipeline, signals: list) -> list:
    """Refresh waiting signals so the dashboard reflects the latest quote."""

    for signal in signals:
        if not signal.executed:
            pipe.broker.evaluate_signal(signal)
    return pipe.storage.list_signals()


def _spa_index() -> FileResponse:
    """Serve the dashboard shell for client-side feature routes."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    """Root serves the shell; client normalizes URL to /overview."""
    return _spa_index()


@app.get("/overview", response_class=HTMLResponse)
@app.get("/overview/{rest:path}", response_class=HTMLResponse)
def spa_overview(rest: str = "") -> FileResponse:
    """SPA shell for the overview / dashboard page."""
    return _spa_index()


@app.get("/timeline", response_class=HTMLResponse)
@app.get("/timeline/{rest:path}", response_class=HTMLResponse)
def spa_timeline(rest: str = "") -> FileResponse:
    """SPA shell for the symbol-timeline page (/timeline or /timeline/SNDK)."""
    return _spa_index()


@app.get("/tools", response_class=HTMLResponse)
@app.get("/tools/{rest:path}", response_class=HTMLResponse)
def spa_tools(rest: str = "") -> FileResponse:
    """SPA shell for the tools page."""
    return _spa_index()


@app.get("/symbol/{symbol}", response_class=HTMLResponse)
def spa_symbol(symbol: str) -> FileResponse:
    """Legacy deep-link → same shell; client maps to /timeline/{symbol}."""
    return _spa_index()


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = load_settings()
    try:
        feed_error = _pipe().feed_error
    except Exception as exc:
        feed_error = str(exc)
    poll = bg_poller.status()
    return {
        "ok": True,
        "source": s.twitter.source,
        "poll_interval_seconds": s.twitter.poll_interval_seconds,
        "auto_poll": bool(getattr(s.twitter, "auto_poll", True)),
        "auto_poll_max_analyze": int(
            getattr(s.twitter, "auto_poll_max_analyze", 1) or 1
        ),
        "auto_poll_agent_tools": bool(
            getattr(s.twitter, "auto_poll_agent_tools", False)
        ),
        "poller": poll,
        "analysis_mode": s.analysis.mode,
        "llm_model": s.analysis.llm_model,
        "classifier_model": (
            s.analysis.classifier_model or s.analysis.llm_model
        ),
        "classifier_min_confidence": s.analysis.classifier_min_confidence,
        "memory_enabled": s.analysis.memory_enabled,
        "memory_lookback_hours": s.analysis.memory_lookback_hours,
        "memory_max_items": s.analysis.memory_max_items,
        "agent_tools_enabled": s.analysis.agent_tools_enabled,
        "agent_max_rounds": s.analysis.agent_max_rounds,
        "timezone": s.app.timezone,
        "market_quote_ttl_seconds": s.market.quote_ttl_seconds,
        "market_require_live_for_execution": s.market.require_live_for_execution,
        "feed_ready": not bool(feed_error),
        "feed_error": feed_error,
        "kols": [k.username for k in s.kols if k.enabled],
        "db": str(s.db_path),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    pipe = _pipe()
    st = pipe.storage
    timezone_name = pipe.settings.app.timezone
    posts = st.list_posts(limit=50)
    signals = _refresh_signal_decisions(pipe, st.list_signals())
    notes = st.list_notes(limit=50)
    trades = st.list_trades()
    kol_stats = [x.as_dict() for x in pipe.reporter.kol_stats()]
    overall = pipe.reporter.overall()
    signal_states: dict[str, int] = {}
    for signal in signals:
        signal_states[signal.state.value] = signal_states.get(signal.state.value, 0) + 1
    symbols = sorted(
        {
            signal.symbol
            for signal in signals
            if signal.symbol not in {"UNKNOWN", "N/A", ""}
        }
    )
    quotes = {
        symbol: _model_payload(
            pipe.market.get_current_snapshot(symbol), timezone_name
        )
        for symbol in symbols
    }
    return {
        "counts": {
            **st.counts(),
            "open_trades": sum(1 for t in trades if t.status.value == "open"),
            "signal_states": signal_states,
            "waiting_signals": signal_states.get("waiting_entry", 0),
            "ready_signals": signal_states.get("ready", 0),
        },
        "posts": [_model_payload(p, timezone_name) for p in posts[:30]],
        "signals": [_model_payload(s, timezone_name) for s in signals[:40]],
        "notes": [_model_payload(n, timezone_name) for n in notes[:40]],
        "trades": [_model_payload(t, timezone_name) for t in trades[:40]],
        "kol_stats": kol_stats,
        "overall": overall,
        "quotes": quotes,
    }


@app.get("/api/symbol/{symbol}")
def symbol_view(symbol: str) -> dict[str, Any]:
    pipe = _pipe()
    resolved = pipe.ticker_map.resolve(symbol) or symbol
    timezone_name = pipe.settings.app.timezone
    snapshot = _display_payload(
        pipe.storage.symbol_snapshot(resolved, timezone_name), timezone_name
    )
    snapshot["quote"] = _model_payload(
        pipe.market.get_current_snapshot(resolved), timezone_name
    )
    return snapshot


@app.get("/api/market/{symbol}")
def market_view(symbol: str) -> dict[str, Any]:
    """Read the latest quote and recent bars for one canonical symbol."""

    pipe = _pipe()
    resolved = pipe.ticker_map.resolve(symbol) or symbol
    quote = pipe.market.get_current_snapshot(resolved)
    bars = pipe.market.get_history(resolved, days=30)
    timezone_name = pipe.settings.app.timezone
    return {
        "symbol": resolved,
        "quote": _model_payload(quote, timezone_name),
        "bars": [_model_payload(bar, timezone_name) for bar in bars[-60:]],
    }


@app.get("/api/klines/{symbol}")
def klines_view(
    symbol: str,
    interval: str = Query("1d", description="1m|5m|15m|1h|4h|1d|1w"),
    limit: int = Query(300, ge=1, le=1000),
    markers: bool = Query(True, description="Include signal/trade overlay markers"),
    include_quote: bool = Query(True, description="Include the latest quote payload"),
) -> dict[str, Any]:
    """OHLCV candles for TradingView-style chart + optional intent markers."""

    from intent_trade.market.klines import SUPPORTED_INTERVALS, normalize_interval

    pipe = _pipe()
    # Accept aliases (e.g. 闪迪 → SNDK) the same way as /api/symbol
    resolved = pipe.ticker_map.resolve(symbol) or symbol.upper()
    if resolved == symbol and symbol != symbol.upper():
        resolved = pipe.ticker_map.resolve(symbol.upper()) or resolved
    timezone_name = pipe.settings.app.timezone
    try:
        interval_n = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = pipe.market.get_klines(resolved, interval=interval_n, limit=limit)
    quote = pipe.market.get_current_snapshot(resolved) if include_quote else None
    payload: dict[str, Any] = {
        "symbol": resolved,
        "asset_class": pipe.ticker_map.asset_class_of(resolved),
        "interval": interval_n,
        "intervals": list(SUPPORTED_INTERVALS),
        "limit": limit,
        "source": result.source,
        "provider_symbol": result.provider_symbol,
        "is_live": result.is_live,
        "error": result.error,
        "count": len(result.bars),
        "bars": [
            {
                "ts": _display_payload(b.ts, timezone_name),
                "time": int(b.ts.timestamp()),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in result.bars
        ],
        "quote": _model_payload(quote, timezone_name) if quote is not None else None,
    }

    if markers:
        snap = pipe.storage.symbol_snapshot(resolved, timezone_name)
        signals_raw = snap.get("structured_signals") or []
        notes_raw = snap.get("notes") or []
        trades_raw = snap.get("trades") or []

        def _as_dict(item: Any) -> dict[str, Any]:
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="python")
            return item if isinstance(item, dict) else {}

        def _dir_token(value: Any) -> str:
            """Normalize Direction enum / 'Direction.LONG' / 'long' → long|short|…"""
            if value is None:
                return "unknown"
            if hasattr(value, "value"):
                value = value.value
            s = str(value).strip()
            if s.startswith("Direction."):
                s = s.split(".", 1)[1]
            s = s.lower()
            if s in {"long", "short", "flat", "unknown"}:
                return s
            return "unknown"

        def _ts_unix(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, datetime):
                dt = value.replace(tzinfo=ZoneInfo("UTC")) if value.tzinfo is None else value
                return int(dt.timestamp())
            try:
                raw = str(value).replace("Z", "+00:00")
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                return int(dt.timestamp())
            except ValueError:
                return None

        post_ids: list[str] = []
        for s in signals_raw:
            s = _as_dict(s)
            if s.get("post_id"):
                post_ids.append(str(s["post_id"]))
        for n in notes_raw:
            n = _as_dict(n)
            if n.get("post_id"):
                post_ids.append(str(n["post_id"]))
        get_posts = getattr(pipe.storage, "get_posts_by_ids", None)
        posts_by_id = get_posts(post_ids) if callable(get_posts) else {}

        def _post_payload(post_id: str | None) -> dict[str, Any]:
            if not post_id:
                return {}
            post = posts_by_id.get(str(post_id))
            if not post:
                return {"post_id": post_id}
            return {
                "post_id": post.id,
                "post_text": post.text,
                "post_url": post.url,
                "post_author": post.author_username,
                "post_time": _display_payload(post.created_at, timezone_name),
            }

        event_markers: list[dict[str, Any]] = []
        for s in signals_raw:
            s = _as_dict(s)
            ts_unix = _ts_unix(s.get("signal_time") or s.get("created_at"))
            if ts_unix is None:
                continue
            direction = _dir_token(s.get("direction"))
            is_long = direction == "long"
            is_short = direction == "short"
            # long=green↑, short=red↓, unknown=gray●
            if is_long:
                color, shape, position, label = (
                    "#3dd68c",
                    "arrowUp",
                    "belowBar",
                    "看多",
                )
            elif is_short:
                color, shape, position, label = (
                    "#ff6b7a",
                    "arrowDown",
                    "aboveBar",
                    "看空",
                )
            else:
                color, shape, position, label = (
                    "#9a9a9a",
                    "circle",
                    "belowBar",
                    "观察",
                )
            post_bits = _post_payload(s.get("post_id"))
            event_markers.append(
                {
                    "time": ts_unix,
                    "position": position,
                    "shape": shape,
                    "color": color,
                    "text": label,
                    "kind": "signal",
                    "direction": direction,
                    "price": s.get("entry_price") or s.get("current_price"),
                    "entry_price": s.get("entry_price"),
                    "stop_loss": s.get("stop_loss"),
                    "take_profit": s.get("take_profit"),
                    "state": s.get("state"),
                    "kol": s.get("kol_username"),
                    "summary": s.get("summary") or s.get("decision_reason") or "",
                    "source_text": (s.get("source_text") or "")[:500],
                    **post_bits,
                }
            )

        for n in notes_raw:
            n = _as_dict(n)
            ts_unix = _ts_unix(n.get("note_time") or n.get("created_at"))
            if ts_unix is None:
                continue
            hint = _dir_token(n.get("direction_hint"))
            is_long = hint == "long"
            is_short = hint == "short"
            # notes: gray by default; green/red only with explicit long/short hint
            if is_long:
                color, position = "#3dd68c", "belowBar"
            elif is_short:
                color, position = "#ff6b7a", "aboveBar"
            else:
                color, position = "#9a9a9a", "belowBar"
            post_bits = _post_payload(n.get("post_id"))
            event_markers.append(
                {
                    "time": ts_unix,
                    "position": position,
                    "shape": "circle",
                    "color": color,
                    "text": "笔记",
                    "kind": "note",
                    "direction": hint,
                    "kol": n.get("kol_username"),
                    "summary": (n.get("content") or "")[:240],
                    "source_text": (n.get("content") or "")[:500],
                    **post_bits,
                }
            )

        trade_markers: list[dict[str, Any]] = []
        for tr in trades_raw:
            tr = _as_dict(tr)
            ts_unix = _ts_unix(tr.get("entry_time") or tr.get("created_at"))
            if ts_unix is None:
                continue
            trade_markers.append(
                {
                    "time": ts_unix,
                    "position": "belowBar",
                    "shape": "square",
                    "color": "#e8b84a",
                    "text": "成交",
                    "kind": "trade",
                    "price": tr.get("entry_price"),
                    "status": tr.get("status"),
                    "kol": tr.get("kol_username"),
                    "summary": f"模拟成交 {tr.get('entry_price')}",
                }
            )

        price_lines: list[dict[str, Any]] = []
        latest = _as_dict(signals_raw[0]) if signals_raw else None
        if latest:
            for key, title, color in (
                ("entry_price", "入场", "#c8c8c8"),
                ("stop_loss", "止损", "#ff6b7a"),
                ("take_profit", "止盈", "#3dd68c"),
                ("trigger_price", "触发", "#e8b84a"),
            ):
                val = latest.get(key)
                if val is not None:
                    try:
                        price_lines.append(
                            {
                                "price": float(val),
                                "title": title,
                                "color": color,
                                "lineWidth": 1,
                                "lineStyle": 2,
                            }
                        )
                    except (TypeError, ValueError):
                        pass
        payload["markers"] = event_markers + trade_markers
        payload["price_lines"] = price_lines
        payload["events"] = event_markers  # richer payload for hover tooltips
    return payload


@app.post("/api/fetch")
def api_fetch(
    username: str = Query("xtony1314"),
    limit: int = Query(10, ge=1, le=50),
    analyze: bool = Query(True),
    vision: bool = Query(
        False,
        description="Deprecated; original images are analyzed automatically",
    ),
) -> dict[str, Any]:
    """Pull latest posts for one KOL; optionally LLM-analyze new ones."""
    pipe = _pipe()
    if pipe.feed_error:
        raise HTTPException(status_code=503, detail=pipe.feed_error)
    try:
        posts = pipe.feed.fetch_user_posts(username.lstrip("@"), limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    new_ids: list[str] = []
    stored_posts = []
    timezone_name = pipe.settings.app.timezone
    for p in posts:
        inserted = pipe.storage.insert_post(p)
        stored_posts.append(p)
        if inserted:
            new_ids.append(p.id)

    analyses_out: list[dict[str, Any]] = []
    if analyze:
        # analyze only newly inserted posts; if none new, still allow re-display
        pending = [p for p in stored_posts if p.id in new_ids] if new_ids else []
        for p in pending:
            # skip if already has signal/note
            if any(s.post_id == p.id for s in pipe.storage.list_signals()):
                continue
            if any(n.post_id == p.id for n in pipe.storage.list_notes(limit=500)):
                continue
            a = pipe.analyze_post(p)
            pipe.storage.save_analysis(
                p.id, p.author_username, _model_payload(a, timezone_name)
            )
            sigs, notes = pipe.persist_analyses([a])
            pipe.execute_pending_signals()
            persisted = [s for s in pipe.storage.list_signals() if s.post_id == p.id]
            analyses_out.append(
                {
                    "post_id": p.id,
                    "text": p.text[:120],
                    "signal_type": a.signal_type.value,
                    "direction": a.direction.value,
                    "symbols": a.canonical_symbols,
                    "entry": a.entry_price,
                    "sl": a.stop_loss,
                    "tp": a.take_profit,
                    "action": a.action.value,
                    "position_state": a.position_state.value,
                    "entry_mode": a.entry_mode.value,
                    "confidence": a.confidence,
                    "summary": a.summary,
                    "persisted_signals": len(sigs),
                    "persisted_notes": len(notes),
                    "signals": [_model_payload(s, timezone_name) for s in persisted],
                }
            )

    return {
        "fetched": len(posts),
        "new_posts": len(new_ids),
        "analyses": analyses_out,
        "posts": [_model_payload(p, timezone_name) for p in posts],
    }


@app.post("/api/run")
def api_run(
    skip_fetch: bool = Query(False),
    max_analyze: int = Query(10, ge=0, le=50),
    settle: bool = Query(True),
    vision: bool = Query(False),
) -> dict[str, Any]:
    pipe = _pipe()
    summary = pipe.run(
        settle=settle,
        fetch=not skip_fetch,
        vision=vision,
        max_analyze=max_analyze if max_analyze > 0 else None,
    )
    return summary


@app.post("/api/analyze-text")
def analyze_text(payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        return {"error": "text required"}
    pipe = _pipe()
    post = SocialPost(
        id="web_dryrun",
        author_username=str(payload.get("kol") or "dryrun"),
        text=text,
        created_at=datetime.utcnow(),
    )
    a = pipe.analyzer.analyze(post)
    timezone_name = pipe.settings.app.timezone
    result = _model_payload(a, timezone_name)
    if a.canonical_symbols:
        signal = TradingSignal(
            post_id=post.id,
            kol_username=post.author_username,
            symbol=a.canonical_symbols[0],
            direction=a.direction,
            action=a.action,
            position_state=a.position_state,
            entry_mode=a.entry_mode,
            entry_price=a.entry_price,
            entry_price_low=a.entry_price_low,
            entry_price_high=a.entry_price_high,
            trigger_price=a.trigger_price,
            stop_loss=a.stop_loss,
            take_profit=a.take_profit,
            take_profit_levels=a.take_profit_levels,
            entry_condition=a.entry_condition,
            time_horizon=a.time_horizon,
            confidence=a.confidence,
            summary=a.summary,
            source_text=a.analysis_text,
            signal_time=post.created_at,
        )
        quote = pipe.market.get_current_snapshot(signal.symbol)
        decision = evaluate_signal(
            signal,
            quote,
            require_live=(
                pipe.settings.market.require_live_for_execution
                and (pipe.settings.twitter.source or "mock").lower() != "mock"
            ),
            max_age_hours=0,
            tolerance_pct=pipe.settings.execution.entry_tolerance_pct,
        )
        result["decision"] = _model_payload(decision, timezone_name)
        result["market"] = _model_payload(quote, timezone_name)
    return result
