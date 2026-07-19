"""Background KOL poller for the web dashboard process.

The browser "auto refresh 60s" only re-reads SQLite. Without this worker,
tweets are only pulled when someone clicks 拉帖/全流水线 or runs
`intent-trade poll` in a terminal. The poller keeps fetch → analyze →
paper-settle running while `intent-trade serve` is up (e.g. under PM2).
"""

from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from intent_trade.config import load_settings
from intent_trade.pipeline.runner import Pipeline

log = logging.getLogger("intent_trade.poller")

_state_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None

_state: dict[str, Any] = {
    "enabled": False,
    "running": False,
    "interval_seconds": 0,
    "jitter_seconds": 0,
    "next_delay_seconds": 0,
    "max_analyze": 0,
    "cycles": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "last_summary": None,
}


def status() -> dict[str, Any]:
    """Snapshot of poller state for /api/health."""
    with _state_lock:
        return dict(_state)


def _update(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_once(max_analyze: int) -> dict[str, Any]:
    settings = load_settings()
    pipe = Pipeline(settings)
    summary = pipe.run(
        settle=True,
        fetch=True,
        vision=False,
        max_analyze=max_analyze,
    )
    return {
        "new_posts": summary.get("new_posts"),
        "analyses": summary.get("analyses"),
        "structured_signals": summary.get("structured_signals"),
        "notes": summary.get("notes"),
        "new_trades": summary.get("new_trades"),
        "settled_trades": summary.get("settled_trades"),
        "waiting_signals": summary.get("waiting_signals"),
        "ready_signals": summary.get("ready_signals"),
        "feed_error": summary.get("feed_error") or "",
    }


def _loop(interval: int, jitter: int, max_analyze: int) -> None:
    log.info(
        "background poller started interval=%ss jitter=%ss max_analyze=%s",
        interval,
        jitter,
        max_analyze,
    )
    # First cycle soon after boot; subsequent cycles wait full interval.
    first = True
    next_delay = interval
    error_streak = 0
    while not _stop.is_set():
        if not first and _stop.wait(timeout=next_delay):
            break
        first = False

        _update(running=True, last_started_at=_utc_now_iso(), last_error=None)
        try:
            summary = _run_once(max_analyze=max_analyze)
            with _state_lock:
                _state["last_summary"] = summary
                _state["last_finished_at"] = _utc_now_iso()
                _state["cycles"] = int(_state.get("cycles") or 0) + 1
                _state["last_error"] = None
            log.info(
                "poller cycle ok new=%s analyzed=%s signals=%s trades=%s",
                summary.get("new_posts"),
                summary.get("analyses"),
                summary.get("structured_signals"),
                summary.get("new_trades"),
            )
            if summary.get("feed_error"):
                log.warning("poller feed_error: %s", summary["feed_error"])
                error_streak += 1
            else:
                error_streak = 0
            base_delay = (
                min(interval * (2 ** min(error_streak, 3)), 3600)
                if error_streak
                else interval
            )
            next_delay = base_delay + random.randint(0, jitter) if jitter else base_delay
            _update(next_delay_seconds=next_delay)
        except Exception as exc:  # noqa: BLE001 — keep loop alive
            _update(last_error=str(exc), last_finished_at=_utc_now_iso())
            log.exception("poller cycle failed: %s", exc)
            error_streak += 1
            base_delay = min(interval * (2 ** min(error_streak, 3)), 3600)
            next_delay = base_delay + random.randint(0, jitter) if jitter else base_delay
            _update(next_delay_seconds=next_delay)
        finally:
            _update(running=False)

    log.info("background poller stopped")


def start() -> None:
    """Start background thread if auto_poll is enabled in settings."""
    global _thread
    settings = load_settings()
    tw = settings.twitter
    enabled = bool(getattr(tw, "auto_poll", True))
    interval = max(15, int(tw.poll_interval_seconds or 60))
    jitter = max(0, int(getattr(tw, "poll_jitter_seconds", 0) or 0))
    max_analyze = int(getattr(tw, "auto_poll_max_analyze", 10) or 10)

    _stop.clear()
    _update(
        enabled=enabled,
        interval_seconds=interval,
        jitter_seconds=jitter,
        next_delay_seconds=interval,
        max_analyze=max_analyze,
        running=False,
    )

    if not enabled:
        log.info("background poller disabled (twitter.auto_poll=false)")
        return

    if _thread and _thread.is_alive():
        log.info("background poller already running")
        return

    _thread = threading.Thread(
        target=_loop,
        args=(interval, jitter, max_analyze),
        name="intent-trade-poller",
        daemon=True,
    )
    _thread.start()


def stop(timeout: float = 5.0) -> None:
    """Signal the poller to stop (best-effort on process shutdown)."""
    global _thread
    _stop.set()
    t = _thread
    if t and t.is_alive():
        t.join(timeout=timeout)
    _thread = None
    _update(running=False)
