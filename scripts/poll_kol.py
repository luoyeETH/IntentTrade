#!/usr/bin/env python3
"""Poll configured KOLs every N seconds (default from settings: 60s).

Usage:
  source .venv/bin/activate
  python scripts/poll_kol.py
  python scripts/poll_kol.py --once
  python scripts/poll_kol.py --interval 60 --limit 20
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console

from intent_trade.config import load_settings
from intent_trade.pipeline.runner import Pipeline
from intent_trade.time_utils import format_display_time

console = Console()


def one_cycle(pipe: Pipeline, limit: int) -> dict:
    # temporarily honor limit for this poll
    old = pipe.settings.twitter.max_posts_per_kol
    pipe.settings.twitter.max_posts_per_kol = limit
    try:
        summary = pipe.run(settle=True)
    finally:
        pipe.settings.twitter.max_posts_per_kol = old
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll KOL tweets into IntentTrade")
    parser.add_argument("--once", action="store_true", help="Single fetch then exit")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between polls (default: settings.twitter.poll_interval_seconds)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max posts per KOL per poll",
    )
    args = parser.parse_args()

    settings = load_settings()
    interval = args.interval or settings.twitter.poll_interval_seconds or 60
    limit = args.limit or settings.twitter.max_posts_per_kol or 20
    pipe = Pipeline(settings)

    console.print(
        f"[bold cyan]IntentTrade poller[/bold cyan] "
        f"source={settings.twitter.source} interval={interval}s limit={limit}"
    )
    console.print(
        "KOLs: "
        + ", ".join(
            f"@{k.username}" for k in settings.kols if k.enabled
        )
    )

    cycle = 0
    while True:
        cycle += 1
        console.rule(
            f"cycle {cycle} · "
            f"{format_display_time(datetime.utcnow(), settings.app.timezone, include_seconds=True)}"
        )
        try:
            summary = one_cycle(pipe, limit=limit)
            console.print(
                f"new_posts={summary.get('new_posts')} "
                f"analyses={summary.get('analyses')} "
                f"signals={summary.get('structured_signals')} "
                f"notes={summary.get('notes')} "
                f"trades={summary.get('new_trades')} "
                f"settled={summary.get('settled_trades')}"
            )
            if summary.get("new_posts") or summary.get("analyses"):
                # show brief latest signals/notes
                for s in pipe.storage.list_signals()[:3]:
                    console.print(
                        f"  [green]SIG[/green] {s.kol_username} {s.symbol} "
                        f"{s.action.value}/{s.direction.value} mode={s.entry_mode.value} "
                        f"entry={s.entry_price} current={s.current_price} "
                        f"state={s.state.value} SL={s.stop_loss} TP={s.take_profit}"
                    )
                for n in pipe.storage.list_notes(limit=3):
                    console.print(f"  [yellow]NOTE[/yellow] {n.symbol} {n.content[:100]}")
        except Exception as e:
            console.print(f"[red]poll error:[/red] {e}")

        if args.once:
            pipe.print_report()
            break
        console.print(f"[dim]sleep {interval}s ...[/dim]")
        time.sleep(interval)


if __name__ == "__main__":
    main()
