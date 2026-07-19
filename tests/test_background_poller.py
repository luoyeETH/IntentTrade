from __future__ import annotations

import threading
from types import SimpleNamespace

from intent_trade.web import poller


def test_fetch_worker_keeps_running_while_analysis_is_blocked(monkeypatch) -> None:
    analysis_started = threading.Event()
    release_analysis = threading.Event()
    second_fetch_finished = threading.Event()
    fetch_count = 0

    class FakePipe:
        def pending_posts(self):
            return [SimpleNamespace(id="pending-post")]

    def fake_analysis_once(pipe, max_analyze):
        analysis_started.set()
        assert release_analysis.wait(timeout=2)
        return {
            "analyses": 1,
            "structured_signals": 0,
            "notes": 1,
            "new_trades": 0,
            "settled_trades": 0,
            "waiting_signals": 0,
            "ready_signals": 0,
            "pending_analyses": 0,
        }

    def fake_fetch_once(pipe):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 1:
            assert analysis_started.wait(timeout=2)
        if fetch_count == 2:
            second_fetch_finished.set()
            poller._stop.set()
            release_analysis.set()
        return {"new_posts": 0, "feed_error": ""}

    monkeypatch.setattr(poller, "_analysis_once", fake_analysis_once)
    monkeypatch.setattr(poller, "_fetch_once", fake_fetch_once)
    poller._stop.clear()
    poller._analysis_wakeup.clear()

    analysis_thread = threading.Thread(
        target=poller._analysis_loop,
        args=(FakePipe(), 1),
    )
    fetch_thread = threading.Thread(
        target=poller._fetch_loop,
        args=(FakePipe(), 0, 0),
    )
    analysis_thread.start()
    fetch_thread.start()
    try:
        assert second_fetch_finished.wait(timeout=2)
    finally:
        poller._stop.set()
        poller._analysis_wakeup.set()
        release_analysis.set()
        fetch_thread.join(timeout=2)
        analysis_thread.join(timeout=2)

    assert fetch_count >= 2
    assert not fetch_thread.is_alive()
    assert not analysis_thread.is_alive()
    poller._stop.clear()
