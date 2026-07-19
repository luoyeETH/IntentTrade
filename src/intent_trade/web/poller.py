"""Independent background fetch and analysis workers for the web process.

Fetching must stay frequent even when an LLM request is slow. The fetch worker
only archives timeline snapshots; a separate analysis worker drains pending
posts and runs paper-trade maintenance. Both workers live with `serve`, so no
browser or manual refresh is required.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from intent_trade.config import load_settings
from intent_trade.pipeline.runner import Pipeline

log = logging.getLogger("intent_trade.poller")

_state_lock = threading.Lock()
_stop = threading.Event()
_analysis_wakeup = threading.Event()
_fetch_thread: Optional[threading.Thread] = None
_analysis_thread: Optional[threading.Thread] = None

_state: dict[str, Any] = {
    "enabled": False,
    "running": False,
    "fetch_running": False,
    "analysis_running": False,
    "interval_seconds": 0,
    "jitter_seconds": 0,
    "next_delay_seconds": 0,
    "max_analyze": 0,
    "agent_tools_enabled": False,
    "cycles": 0,
    "fetch_cycles": 0,
    "analysis_cycles": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "last_summary": None,
    "last_fetch_started_at": None,
    "last_fetch_finished_at": None,
    "last_fetch_error": None,
    "last_fetch_summary": None,
    "last_analysis_started_at": None,
    "last_analysis_finished_at": None,
    "last_analysis_error": None,
    "last_analysis_summary": None,
    "analysis_current_post_id": None,
}


def status() -> dict[str, Any]:
    """Snapshot worker state for `/api/health`."""

    with _state_lock:
        return dict(_state)


def _update(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)
        _state["running"] = bool(
            _state.get("fetch_running") or _state.get("analysis_running")
        )
        _state["last_error"] = (
            _state.get("last_fetch_error") or _state.get("last_analysis_error")
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fetch_once(pipe: Pipeline) -> dict[str, Any]:
    new_posts = pipe.ingest()
    return {
        "new_posts": len(new_posts),
        "feed_error": pipe.feed_error or "",
    }


def _analysis_once(pipe: Pipeline, max_analyze: int) -> dict[str, Any]:
    summary = pipe.run(
        settle=True,
        fetch=False,
        vision=False,
        max_analyze=max_analyze,
    )
    return {
        "analyses": summary.get("analyses"),
        "structured_signals": summary.get("structured_signals"),
        "notes": summary.get("notes"),
        "new_trades": summary.get("new_trades"),
        "settled_trades": summary.get("settled_trades"),
        "waiting_signals": summary.get("waiting_signals"),
        "ready_signals": summary.get("ready_signals"),
        "pending_analyses": len(pipe.pending_posts()),
    }


def _fetch_loop(pipe: Pipeline, interval: int, jitter: int) -> None:
    log.info("fetch worker started interval=%ss jitter=%ss", interval, jitter)
    next_delay = 0.0
    error_streak = 0
    while not _stop.is_set():
        if next_delay > 0 and _stop.wait(timeout=next_delay):
            break

        started_monotonic = time.monotonic()
        started_at = _utc_now_iso()
        _update(
            fetch_running=True,
            last_started_at=started_at,
            last_fetch_started_at=started_at,
            last_fetch_error=None,
        )
        try:
            summary = _fetch_once(pipe)
            finished_at = _utc_now_iso()
            feed_error = str(summary.get("feed_error") or "")
            with _state_lock:
                cycles = int(_state.get("fetch_cycles") or 0) + 1
                _state.update(
                    cycles=cycles,
                    fetch_cycles=cycles,
                    last_finished_at=finished_at,
                    last_fetch_finished_at=finished_at,
                    last_fetch_summary=summary,
                    last_summary={**(_state.get("last_summary") or {}), **summary},
                    last_fetch_error=feed_error or None,
                )
            if feed_error:
                error_streak += 1
                log.warning("fetch worker feed_error: %s", feed_error)
            else:
                error_streak = 0
                log.info("fetch cycle ok new=%s", summary.get("new_posts"))
            _analysis_wakeup.set()
        except Exception as exc:  # noqa: BLE001 - keep worker alive
            finished_at = _utc_now_iso()
            _update(
                last_finished_at=finished_at,
                last_fetch_finished_at=finished_at,
                last_fetch_error=str(exc),
            )
            log.exception("fetch cycle failed: %s", exc)
            error_streak += 1
        finally:
            _update(fetch_running=False)

        base_delay = (
            min(interval * (2 ** min(error_streak, 2)), max(interval, 120))
            if error_streak
            else interval
        )
        elapsed = time.monotonic() - started_monotonic
        jitter_delay = random.randint(0, jitter) if jitter else 0
        next_delay = max(0.0, base_delay - elapsed) + jitter_delay
        _update(next_delay_seconds=max(0, round(next_delay)))

    log.info("fetch worker stopped")


def _analysis_loop(pipe: Pipeline, max_analyze: int) -> None:
    log.info("analysis worker started max_analyze=%s", max_analyze)
    run_immediately = True
    while not _stop.is_set():
        if not run_immediately:
            _analysis_wakeup.wait()
            if _stop.is_set():
                break
        _analysis_wakeup.clear()
        run_immediately = False

        pending = pipe.pending_posts()
        current_post_id = pending[0].id if pending else None
        started_at = _utc_now_iso()
        _update(
            analysis_running=True,
            analysis_current_post_id=current_post_id,
            last_analysis_started_at=started_at,
            last_analysis_error=None,
        )
        retry_without_progress = False
        try:
            summary = _analysis_once(pipe, max_analyze=max_analyze)
            finished_at = _utc_now_iso()
            with _state_lock:
                _state["analysis_cycles"] = int(
                    _state.get("analysis_cycles") or 0
                ) + 1
                _state["last_analysis_finished_at"] = finished_at
                _state["last_analysis_summary"] = summary
                _state["last_summary"] = {
                    **(_state.get("last_summary") or {}),
                    **summary,
                }
                _state["last_analysis_error"] = None
            log.info(
                "analysis cycle ok analyzed=%s signals=%s notes=%s pending=%s",
                summary.get("analyses"),
                summary.get("structured_signals"),
                summary.get("notes"),
                summary.get("pending_analyses"),
            )
            has_pending = bool(summary.get("pending_analyses"))
            made_progress = bool(summary.get("analyses"))
            run_immediately = has_pending and made_progress
            retry_without_progress = has_pending and not made_progress
        except Exception as exc:  # noqa: BLE001 - keep worker alive
            _update(
                last_analysis_finished_at=_utc_now_iso(),
                last_analysis_error=str(exc),
            )
            log.exception("analysis cycle failed: %s", exc)
            if not _stop.wait(timeout=30):
                run_immediately = True
        finally:
            _update(analysis_running=False, analysis_current_post_id=None)

        if retry_without_progress and not _stop.is_set():
            _analysis_wakeup.wait(timeout=30)
            _analysis_wakeup.clear()
            run_immediately = True

    log.info("analysis worker stopped")


def start() -> None:
    """Start independent workers when automatic polling is enabled."""

    global _fetch_thread, _analysis_thread
    settings = load_settings()
    tw = settings.twitter
    enabled = bool(getattr(tw, "auto_poll", True))
    interval = max(15, int(tw.poll_interval_seconds or 60))
    jitter = max(0, int(getattr(tw, "poll_jitter_seconds", 0) or 0))
    max_analyze = max(1, int(getattr(tw, "auto_poll_max_analyze", 1) or 1))
    agent_tools_enabled = bool(getattr(tw, "auto_poll_agent_tools", False))

    if (_fetch_thread and _fetch_thread.is_alive()) or (
        _analysis_thread and _analysis_thread.is_alive()
    ):
        log.info("background workers already running")
        return

    _stop.clear()
    _analysis_wakeup.clear()
    _update(
        enabled=enabled,
        interval_seconds=interval,
        jitter_seconds=jitter,
        next_delay_seconds=0,
        max_analyze=max_analyze,
        agent_tools_enabled=agent_tools_enabled,
        fetch_running=False,
        analysis_running=False,
        last_fetch_error=None,
        last_analysis_error=None,
    )
    if not enabled:
        log.info("background workers disabled (twitter.auto_poll=false)")
        return

    # Construct sequentially so additive SQLite migrations cannot race on boot.
    fetch_pipe = Pipeline(settings, agent_tools_enabled=False)
    analysis_pipe = Pipeline(
        settings,
        agent_tools_enabled=agent_tools_enabled,
    )
    _analysis_thread = threading.Thread(
        target=_analysis_loop,
        args=(analysis_pipe, max_analyze),
        name="intent-trade-analysis",
        daemon=True,
    )
    _fetch_thread = threading.Thread(
        target=_fetch_loop,
        args=(fetch_pipe, interval, jitter),
        name="intent-trade-fetch",
        daemon=True,
    )
    _analysis_thread.start()
    _fetch_thread.start()


def stop(timeout: float = 5.0) -> None:
    """Signal both workers to stop (best effort for an in-flight LLM call)."""

    global _fetch_thread, _analysis_thread
    _stop.set()
    _analysis_wakeup.set()
    for thread in (_fetch_thread, _analysis_thread):
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
    if not (_fetch_thread and _fetch_thread.is_alive()):
        _fetch_thread = None
    if not (_analysis_thread and _analysis_thread.is_alive()):
        _analysis_thread = None
    _update(fetch_running=False, analysis_running=False)
