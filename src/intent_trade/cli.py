"""CLI entrypoint: intent-trade run | report | symbol | reset-db."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from intent_trade.config import load_settings
from intent_trade.pipeline.runner import Pipeline

app = typer.Typer(help="IntentTrade — KOL social signal copy-trading (Phase 1)")
console = Console()


@app.command()
def run(
    no_settle: bool = typer.Option(False, help="Skip SL/TP settlement pass"),
    skip_fetch: bool = typer.Option(
        False, help="Do not call Twitter API; only analyze posts already in DB"
    ),
    vision: bool = typer.Option(
        False, help="Enable multimodal vision on images (slow/costly)"
    ),
    max_analyze: Optional[int] = typer.Option(
        None, help="Max posts to LLM-analyze this run"
    ),
    config: Optional[Path] = typer.Option(None, help="Path to settings.yaml"),
) -> None:
    """Fetch KOL posts → AI analyze intents → paper trade → report."""
    settings = load_settings(config)
    pipe = Pipeline(settings)
    summary = pipe.run(
        settle=not no_settle,
        fetch=not skip_fetch,
        vision=vision,
        max_analyze=max_analyze,
    )
    pipe.print_report(summary)
    console.print("[green]Done.[/green]")


@app.command()
def report(
    config: Optional[Path] = typer.Option(None, help="Path to settings.yaml"),
) -> None:
    """Print current DB report without re-ingesting."""
    settings = load_settings(config)
    pipe = Pipeline(settings)
    pipe.print_report()


@app.command("symbol")
def symbol_view(
    symbol: str = typer.Argument(..., help="Canonical symbol e.g. BTC-USD or SNDK"),
    config: Optional[Path] = typer.Option(None),
) -> None:
    """Show accumulated structured signals + descriptive notes for a symbol."""
    settings = load_settings(config)
    pipe = Pipeline(settings)
    # try resolve alias
    resolved = pipe.ticker_map.resolve(symbol) or symbol
    snap = pipe.storage.symbol_snapshot(resolved, settings.app.timezone)
    console.print_json(json.dumps(snap, ensure_ascii=False, default=str))


@app.command("price")
def price(
    symbol: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None),
) -> None:
    """Fetch latest market price for a symbol (yfinance / fallback)."""
    settings = load_settings(config)
    pipe = Pipeline(settings)
    resolved = pipe.ticker_map.resolve(symbol) or symbol
    snapshot = pipe.market.get_current_snapshot(resolved)
    px = snapshot.price
    yf_sym = pipe.ticker_map.yfinance_symbol(resolved)
    if px is None:
        console.print(
            f"[red]No price for {resolved} (yf={yf_sym}, source={snapshot.source}, "
            f"error={snapshot.error})[/red]"
        )
        raise typer.Exit(1)
    live = "live" if snapshot.is_live and not snapshot.stale else "delayed/stale"
    age = f" age={snapshot.age_seconds:.0f}s" if snapshot.age_seconds is not None else ""
    console.print(
        f"{resolved} (yf={yf_sym}) last={px} source={snapshot.source} "
        f"quality={live}{age}"
    )


@app.command("reset-db")
def reset_db(
    config: Optional[Path] = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete the SQLite database (re-run pipeline from clean state)."""
    settings = load_settings(config)
    path = settings.db_path
    if not yes:
        confirm = typer.confirm(f"Delete {path}?")
        if not confirm:
            raise typer.Abort()
    if path.exists():
        path.unlink()
        console.print(f"Deleted {path}")
    else:
        console.print("DB does not exist")


@app.command("analyze-text")
def analyze_text(
    text: str = typer.Argument(..., help="Raw KOL text to analyze"),
    config: Optional[Path] = typer.Option(None),
) -> None:
    """Quick dry-run intent analysis on free text (no DB write)."""
    from datetime import datetime

    from intent_trade.models.domain import SocialPost

    settings = load_settings(config)
    pipe = Pipeline(settings)
    post = SocialPost(
        id="dryrun",
        author_username="dryrun",
        text=text,
        created_at=datetime.utcnow(),
    )
    result = pipe.analyzer.analyze(post)
    console.print_json(result.model_dump_json())


@app.command("fetch-kol")
def fetch_kol(
    username: str = typer.Argument(
        "xtony1314", help="X username without @ (default: xtony1314)"
    ),
    limit: int = typer.Option(10, help="Max posts to fetch"),
    config: Optional[Path] = typer.Option(None),
    analyze: bool = typer.Option(True, help="Run intent analysis on each post"),
    save: bool = typer.Option(False, help="Persist posts into SQLite"),
) -> None:
    """Fetch one KOL's recent posts via configured twitter.source (rapidapi / etc)."""
    settings = load_settings(config)
    pipe = Pipeline(settings)
    console.print(
        f"[cyan]source={settings.twitter.source}[/cyan] "
        f"fetching @{username.lstrip('@')} limit={limit}"
    )
    posts = pipe.feed.fetch_user_posts(username.lstrip("@"), limit=limit)
    if not posts:
        console.print("[yellow]No posts returned. Check API key / source / username.[/yellow]")
        raise typer.Exit(1)

    for p in posts:
        if save:
            pipe.storage.upsert_post(p)
        console.rule(f"@{p.author_username} · {p.created_at} · {p.id}")
        console.print(p.text)
        if p.media_urls:
            console.print(f"[dim]media: {p.media_urls}[/dim]")
        if analyze:
            a = pipe.analyzer.analyze(p)
            console.print(
                f"→ type={a.signal_type.value} dir={a.direction.value} "
                f"symbols={a.canonical_symbols} "
                f"entry={a.entry_price} SL={a.stop_loss} TP={a.take_profit} "
                f"conf={a.confidence}"
            )
            if a.descriptive_note:
                console.print(f"  note: {a.descriptive_note[:160]}")
    console.print(f"[green]Fetched {len(posts)} posts.[/green]")


@app.command("serve")
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        help="Bind host (use 0.0.0.0 so LAN/public IP can open the page)",
    ),
    port: int = typer.Option(8787, help="Bind port"),
    reload: bool = typer.Option(False, help="Dev reload"),
) -> None:
    """Start web dashboard. Default binds 0.0.0.0:8787 for IP access."""
    import socket

    import uvicorn

    console.print(f"[green]IntentTrade dashboard → http://{host}:{port}[/green]")
    if host in ("0.0.0.0", "::"):
        try:
            # best-effort local IPs for copy-paste
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]
            for ip in ips:
                if not ip.startswith("127."):
                    console.print(f"  open [cyan]http://{ip}:{port}[/cyan]")
        except Exception:
            pass
        console.print(f"  or [cyan]http://<server-public-ip>:{port}[/cyan]")
    uvicorn.run(
        "intent_trade.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command("poll")
def poll(
    once: bool = typer.Option(False, help="Single cycle then exit"),
    interval: Optional[int] = typer.Option(None, help="Seconds between polls"),
    limit: Optional[int] = typer.Option(None, help="Max posts per KOL"),
    max_analyze: Optional[int] = typer.Option(
        10, help="Max new posts to LLM-analyze per cycle"
    ),
    config: Optional[Path] = typer.Option(None),
) -> None:
    """Poll KOLs on an interval (default 60s) and run AI analyze + paper trade."""
    import time

    settings = load_settings(config)
    pipe = Pipeline(settings)
    sec = interval or settings.twitter.poll_interval_seconds or 60
    lim = limit or settings.twitter.max_posts_per_kol or 20
    console.print(
        f"[bold]poll[/bold] source={settings.twitter.source} every {sec}s, "
        f"limit={lim}, max_analyze={max_analyze}, kols="
        + ",".join(k.username for k in settings.kols if k.enabled)
    )
    n = 0
    while True:
        n += 1
        console.rule(f"poll #{n}")
        old = pipe.settings.twitter.max_posts_per_kol
        pipe.settings.twitter.max_posts_per_kol = lim
        try:
            summary = pipe.run(
                settle=True,
                fetch=True,
                vision=False,
                max_analyze=max_analyze,
            )
        finally:
            pipe.settings.twitter.max_posts_per_kol = old
        console.print(
            f"new={summary.get('new_posts')} analyzed={summary.get('analyses')} "
            f"signals={summary.get('structured_signals')} notes={summary.get('notes')} "
            f"trades={summary.get('new_trades')} waiting={summary.get('waiting_signals')} "
            f"ready={summary.get('ready_signals')}"
        )
        if once:
            pipe.print_report(summary)
            break
        time.sleep(sec)


if __name__ == "__main__":
    app()
