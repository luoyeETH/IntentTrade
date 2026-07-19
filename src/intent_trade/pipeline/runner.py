"""End-to-end Phase 1 pipeline: fetch → analyze → record → paper trade → stats."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from intent_trade.analysis.intent import IntentAnalyzer
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.backtest.stats import PerformanceReporter
from intent_trade.config import Settings, load_settings
from intent_trade.execution.paper import PaperBroker
from intent_trade.market.prices import MarketDataService
from intent_trade.models.domain import (
    Direction,
    IntentAction,
    IntentAnalysis,
    InstrumentNote,
    PositionState,
    SignalState,
    SignalType,
    SocialPost,
    TradingSignal,
)
from intent_trade.storage.db import Storage
from intent_trade.time_utils import format_display_time
from intent_trade.twitter.client import (
    SocialFeed,
    UnavailableSocialFeed,
    create_social_feed,
)

console = Console()


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.storage = Storage(self.settings.db_path)
        self.ticker_map = TickerMap(self.settings.ticker_aliases_path)
        self.feed_error = ""
        try:
            self.feed: SocialFeed = create_social_feed(self.settings)
        except RuntimeError as exc:
            self.feed_error = str(exc)
            self.feed = UnavailableSocialFeed(self.feed_error)
        self.market = MarketDataService(
            self.settings,
            yf_symbol_map=self.ticker_map.yfinance_map(),
            asset_class_map=self.ticker_map.asset_class_map(),
        )
        self.analyzer = IntentAnalyzer(self.ticker_map, self.settings.analysis)
        self.broker = PaperBroker(
            self.storage,
            self.market,
            self.settings.execution,
            require_live_market=(
                self.settings.market.require_live_for_execution
                and (self.settings.twitter.source or "mock").lower() != "mock"
            ),
            max_signal_age_hours=self.settings.analysis.pending_signal_ttl_hours,
        )
        self.reporter = PerformanceReporter(self.storage)

    def run(
        self,
        settle: bool = True,
        *,
        fetch: bool = True,
        vision: bool = False,
        max_analyze: Optional[int] = None,
    ) -> dict[str, Any]:
        posts_new = self.ingest() if fetch else []
        if fetch:
            console.print(f"[dim]ingest done: {len(posts_new)} new posts[/dim]")
        analyses, signals, notes = self.analyze_new_posts(
            vision=vision,
            max_analyze=max_analyze,
        )
        trades = self.execute_pending_signals()
        settled: list = []
        if settle:
            settled = self.broker.settle_open_trades(
                path_resolution=self.settings.backtest.path_resolution
            )
        stats = {
            "kol": [s.as_dict() for s in self.reporter.kol_stats()],
            "overall": self.reporter.overall(),
        }
        signal_states = {}
        for signal in self.storage.list_signals():
            signal_states[signal.state.value] = signal_states.get(signal.state.value, 0) + 1
        summary = {
            "new_posts": len(posts_new),
            "analyses": len(analyses),
            "structured_signals": len(signals),
            "notes": len(notes),
            "new_trades": len([t for t in trades if t]),
            "settled_trades": len(settled),
            "signal_states": signal_states,
            "waiting_signals": signal_states.get(SignalState.WAITING_ENTRY.value, 0),
            "ready_signals": signal_states.get(SignalState.READY.value, 0),
            "feed_error": self.feed_error,
            "stats": stats,
        }
        return summary

    def ingest(self) -> list:
        usernames = [
            k.username for k in self.settings.kols if k.enabled
        ]
        console.print(f"[dim]fetching {usernames} ...[/dim]")
        try:
            posts = self.feed.fetch_kols(
                usernames, limit_per_user=self.settings.twitter.max_posts_per_kol
            )
        except RuntimeError as exc:
            self.feed_error = str(exc)
            console.print(f"[red]feed unavailable: {self.feed_error}[/red]")
            return []
        new = []
        for p in posts:
            if not self.storage.post_exists(p.id):
                self.storage.upsert_post(p)
                new.append(p)
            else:
                # Refresh text/media metadata for posts already seen.
                self.storage.upsert_post(p)
        return new

    def analyze_new_posts(
        self,
        *,
        vision: bool = False,
        max_analyze: Optional[int] = None,
    ) -> tuple[list[IntentAnalysis], list[TradingSignal], list[InstrumentNote]]:
        """Analyze and persist posts oldest-first so later posts see new history."""
        existing_posts = self.storage.list_posts(limit=500)
        done_post_ids = set()
        for s in self.storage.list_signals():
            done_post_ids.add(s.post_id)
        for n in self.storage.list_notes(limit=1000):
            done_post_ids.add(n.post_id)

        pending = [p for p in existing_posts if p.id not in done_post_ids]
        pending.sort(key=lambda item: item.created_at)
        if max_analyze is not None:
            pending = pending[: max(0, max_analyze)]
        console.print(
            f"[dim]LLM analyze pending={len(pending)} "
            f"(skip already recorded; original images are analyzed automatically)[/dim]"
        )

        results: list[IntentAnalysis] = []
        signals: list[TradingSignal] = []
        notes: list[InstrumentNote] = []
        for i, post in enumerate(pending, 1):
            console.print(
                f"[cyan][{i}/{len(pending)}][/cyan] @{post.author_username} "
                f"{post.id[:12]}… {post.text[:48].replace(chr(10), ' ')}"
            )
            try:
                analysis = self.analyze_post(post)
            except Exception as e:
                console.print(f"[red]analyze failed {post.id}: {e}[/red]")
                continue
            self.storage.save_analysis(
                post.id, post.author_username, analysis.model_dump(mode="json")
            )
            new_signals, new_notes = self.persist_analyses([analysis])
            signals.extend(new_signals)
            notes.extend(new_notes)
            console.print(
                f"  → {analysis.analyzer} {analysis.signal_type.value} "
                f"{analysis.action.value} {analysis.direction.value} "
                f"mode={analysis.entry_mode.value} {analysis.canonical_symbols} "
                f"E={analysis.entry_price} SL={analysis.stop_loss} "
                f"TP={analysis.take_profit} trigger={analysis.trigger_price} "
                f"conf={analysis.confidence}"
            )
            results.append(analysis)
        return results, signals, notes

    def analyze_post(self, post: SocialPost) -> IntentAnalysis:
        """Analyze one post and reconcile it with recent same-KOL history."""

        history = self._recent_kol_history(post)
        analysis = self.analyzer.analyze(post, history=history)
        self._apply_memory_actions(analysis, history)
        return analysis

    def _recent_kol_history(self, post: SocialPost) -> list[dict[str, Any]]:
        config = self.settings.analysis
        if not config.memory_enabled or config.memory_max_items <= 0:
            return []
        cutoff = post.created_at - timedelta(hours=config.memory_lookback_hours)
        username = post.author_username.lstrip("@").lower()
        candidates: list[tuple[Any, dict[str, Any]]] = []
        eligible_states = {
            SignalState.WAITING_MARKET_DATA,
            SignalState.WAITING_ENTRY,
            SignalState.READY,
            SignalState.WAITING_RISK_LIMIT,
        }

        for signal in self.storage.list_signals():
            if signal.post_id == post.id:
                continue
            if signal.kol_username.lstrip("@").lower() != username:
                continue
            if not (cutoff <= signal.signal_time < post.created_at):
                continue
            candidates.append(
                (
                    signal.signal_time,
                    {
                        "kind": "signal",
                        "signal_id": signal.id,
                        "post_id": signal.post_id,
                        "time": signal.signal_time.isoformat(),
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "action": signal.action.value,
                        "position_state": signal.position_state.value,
                        "entry_mode": signal.entry_mode.value,
                        "entry_price": signal.entry_price,
                        "entry_price_low": signal.entry_price_low,
                        "entry_price_high": signal.entry_price_high,
                        "trigger_price": signal.trigger_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "state": signal.state.value,
                        "executed": signal.executed,
                        "eligible_for_supersede": (
                            not signal.executed and signal.state in eligible_states
                        ),
                        "summary": signal.summary,
                        "source_text": signal.source_text,
                    },
                )
            )

        for note in self.storage.list_notes(limit=500):
            if note.post_id == post.id or note.symbol in {"N/A", "UNKNOWN", ""}:
                continue
            if note.kol_username.lstrip("@").lower() != username:
                continue
            if not (cutoff <= note.note_time < post.created_at):
                continue
            candidates.append(
                (
                    note.note_time,
                    {
                        "kind": "note",
                        "post_id": note.post_id,
                        "time": note.note_time.isoformat(),
                        "symbol": note.symbol,
                        "direction": note.direction_hint.value,
                        "content": note.content,
                    },
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        symbols: set[str] = set()
        for _, item in candidates:
            symbol = str(item.get("symbol") or "")
            if symbol not in symbols and len(symbols) >= 3:
                continue
            symbols.add(symbol)
            selected.append(item)
            if len(selected) >= config.memory_max_items:
                break
        return selected

    def _apply_memory_actions(
        self,
        analysis: IntentAnalysis,
        history: list[dict[str, Any]],
    ) -> list[str]:
        memory = analysis.extracted_fields.get("memory") or {}
        requested_ids = memory.get("supersede_signal_ids") or []
        relation = str(memory.get("relation") or "")
        related_symbol = str(memory.get("related_symbol") or "")
        current_symbols = set(analysis.canonical_symbols)
        if related_symbol:
            current_symbols.add(related_symbol)
        related_pending = [
            item
            for item in history
            if item.get("kind") == "signal"
            and item.get("eligible_for_supersede")
            and item.get("symbol") in current_symbols
        ]
        if relation == "continues" and related_pending:
            analysis.signal_type = SignalType.DESCRIPTIVE
            analysis.descriptive_note = (
                str(memory.get("summary") or "")
                or analysis.summary
                or analysis.raw_text
            )
            memory["suppressed_duplicate_signal"] = True
            memory["applied_signal_ids"] = []
            return []
        if not requested_ids:
            return []

        target_directions = {
            str(item.get("direction") or "")
            for item in related_pending
            if item.get("direction")
        }
        reverses_direction = (
            relation == "reverses"
            and analysis.direction in (Direction.LONG, Direction.SHORT)
            and (
                Direction.SHORT.value
                if analysis.direction == Direction.LONG
                else Direction.LONG.value
            )
            in target_directions
        )
        relation_matches_intent = (
            relation == "confirms_entry"
            and analysis.position_state == PositionState.ENTERED
        ) or (
            relation == "adjusts"
            and analysis.action in (IntentAction.OPEN, IntentAction.ADD)
            and analysis.position_state == PositionState.PLANNED
        ) or (
            relation == "exits"
            and analysis.action in (IntentAction.CLOSE, IntentAction.REDUCE)
        ) or relation == "cancels" or reverses_direction
        if not relation_matches_intent:
            memory["supersede_signal_ids"] = []
            memory["applied_signal_ids"] = []
            return []

        allowed_symbols = set(analysis.canonical_symbols)
        if related_symbol:
            allowed_symbols.add(related_symbol)
        eligible_by_id = {
            str(item["signal_id"]): item
            for item in history
            if item.get("kind") == "signal"
            and item.get("signal_id")
            and item.get("eligible_for_supersede")
            and item.get("symbol") in allowed_symbols
        }
        validated_ids = [
            str(value) for value in requested_ids if str(value) in eligible_by_id
        ]
        applied = self.storage.mark_signals_superseded(
            validated_ids,
            replacement_post_id=analysis.post_id,
            reason=str(memory.get("summary") or analysis.summary),
        )
        memory["supersede_signal_ids"] = validated_ids
        memory["applied_signal_ids"] = applied
        return applied

    @staticmethod
    def _levels_consistent(
        entry: float | None,
        sl: float | None,
        tp: float | None,
        direction: Direction,
        entry_low: float | None = None,
        entry_high: float | None = None,
        take_profit_levels: list[float] | None = None,
        trigger_price: float | None = None,
    ) -> bool:
        """Reject cross-contaminated or directionally impossible price sets."""
        levels = [
            x
            for x in (
                entry,
                entry_low,
                entry_high,
                trigger_price,
                sl,
                tp,
                *(take_profit_levels or []),
            )
            if x is not None and x > 0
        ]
        if not levels:
            return False
        # This is only an extraction sanity check, not a volatility rule.
        # A fixed small ratio would incorrectly reject legitimate long-term
        # setups, so use a broad bound and rely on direction checks below.
        if max(levels) / min(levels) > 100:
            return False
        if entry_low is not None and entry_high is not None and entry_low > entry_high:
            return False
        if entry and sl:
            if direction == Direction.LONG and sl >= entry:
                return False
            if direction == Direction.SHORT and sl <= entry:
                return False
        targets = [x for x in ([tp] + (take_profit_levels or [])) if x is not None]
        if entry and targets:
            if direction == Direction.LONG and any(x <= entry for x in targets):
                return False
            if direction == Direction.SHORT and any(x >= entry for x in targets):
                return False
        return True

    def persist_analyses(self, analyses: list) -> tuple[list[TradingSignal], list[InstrumentNote]]:
        signals: list[TradingSignal] = []
        notes: list[InstrumentNote] = []
        min_conf = self.settings.analysis.structured_min_confidence

        for a in analyses:
            symbols = a.canonical_symbols or []
            if not symbols:
                # Non-trading chatter: no instrument — store as N/A (not "UNKNOWN")
                if a.descriptive_note or a.raw_text:
                    note = InstrumentNote(
                        symbol="N/A",
                        kol_username=a.kol_username,
                        post_id=a.post_id,
                        note_time=a.analyzed_at,
                        content=a.descriptive_note or a.raw_text,
                        direction_hint=a.direction,
                        confidence=a.confidence,
                        tags=["non_trade", "unmapped"],
                    )
                    self.storage.insert_note(note)
                    notes.append(note)
                continue

            has_explicit_level = any(
                value is not None
                for value in (
                    a.entry_price,
                    a.entry_price_low,
                    a.entry_price_high,
                    a.trigger_price,
                    a.stop_loss,
                    a.take_profit,
                )
            ) or bool(a.take_profit_levels)
            structured_ok = (
                a.signal_type == SignalType.STRUCTURED
                and a.confidence >= min_conf
                and symbols
                and (
                    self._levels_consistent(
                        a.entry_price,
                        a.stop_loss,
                        a.take_profit,
                        a.direction,
                        a.entry_price_low,
                        a.entry_price_high,
                        a.take_profit_levels,
                        a.trigger_price,
                    )
                    if has_explicit_level
                    else a.action in (
                        IntentAction.OPEN,
                        IntentAction.ADD,
                        IntentAction.CLOSE,
                        IntentAction.REDUCE,
                    )
                )
                and (
                    a.action in (
                        IntentAction.OPEN,
                        IntentAction.ADD,
                        IntentAction.CLOSE,
                        IntentAction.REDUCE,
                    )
                )
                and (
                    a.action in (IntentAction.CLOSE, IntentAction.REDUCE)
                    or a.direction in (Direction.LONG, Direction.SHORT)
                )
            )
            # One structured primary symbol per post (avoid multi-symbol level reuse)
            primary_symbols = symbols[:1] if structured_ok else []

            for sym in symbols:
                if structured_ok and sym in primary_symbols:
                    signal_time = self._post_time(a.post_id) or a.analyzed_at
                    ttl_hours = a.validity_hours
                    if ttl_hours is None:
                        ttl_hours = self.settings.analysis.pending_signal_ttl_hours
                    expires_at = (
                        signal_time + timedelta(hours=ttl_hours)
                        if ttl_hours and ttl_hours > 0
                        else None
                    )
                    sig = TradingSignal(
                        post_id=a.post_id,
                        kol_username=a.kol_username,
                        symbol=sym,
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
                        expires_at=expires_at,
                        confidence=a.confidence,
                        summary=a.summary,
                        source_text=a.analysis_text,
                        analyzer=a.analyzer,
                        reasoning=a.reasoning,
                        field_confidence=a.field_confidence,
                        evidence=a.evidence,
                        signal_time=signal_time,
                    )
                    self.storage.insert_signal(sig)
                    signals.append(sig)
                else:
                    content = a.descriptive_note or a.summary or a.raw_text
                    # format like: 【2026-7-18 23:00 KOL_NAME: ...】
                    ts = self._post_time(a.post_id) or a.analyzed_at
                    stamp = (
                        format_display_time(ts, self.settings.app.timezone)
                        if hasattr(ts, "year")
                        else str(ts)
                    )
                    formatted = f"【{stamp} {a.kol_username}: {content}】"
                    note = InstrumentNote(
                        symbol=sym,
                        kol_username=a.kol_username,
                        post_id=a.post_id,
                        note_time=ts,
                        content=formatted,
                        direction_hint=a.direction,
                        confidence=a.confidence,
                        tags=[
                            a.signal_type.value,
                            a.action.value,
                            *(
                                ["unmapped", "non_trade"]
                                if sym in {"UNKNOWN", "N/A", ""}
                                else []
                            ),
                        ],
                    )
                    self.storage.insert_note(note)
                    notes.append(note)
        return signals, notes

    def _post_time(self, post_id: str):
        for p in self.storage.list_posts(limit=500):
            if p.id == post_id:
                return p.created_at
        return None

    def execute_pending_signals(self) -> list:
        pending = self.storage.list_signals(unexecuted_only=True)
        trades = []
        for sig in pending:
            t = self.broker.execute_signal(sig)
            if t:
                trades.append(t)
        return trades

    def print_report(self, summary: dict[str, Any] | None = None) -> None:
        if summary is None:
            summary = {
                "stats": {
                    "kol": [s.as_dict() for s in self.reporter.kol_stats()],
                    "overall": self.reporter.overall(),
                }
            }

        console.rule("[bold cyan]IntentTrade Phase 1 Report")
        if "new_posts" in summary:
            console.print(
                f"新帖 {summary.get('new_posts')} | 分析 {summary.get('analyses')} | "
                f"结构化信号 {summary.get('structured_signals')} | "
                f"描述笔记 {summary.get('notes')} | "
                f"新开仓 {summary.get('new_trades')} | 平仓结算 {summary.get('settled_trades')} | "
                f"等待入场 {summary.get('waiting_signals')} | 可执行 {summary.get('ready_signals')}"
            )

        # Signals table
        sigs = self.storage.list_signals()
        t1 = Table(title="结构化信号 (entry / SL / TP)")
        t1.add_column("Time")
        t1.add_column("KOL")
        t1.add_column("Symbol")
        t1.add_column("Dir")
        t1.add_column("Entry")
        t1.add_column("SL")
        t1.add_column("TP")
        t1.add_column("Mode")
        t1.add_column("State")
        t1.add_column("Current")
        t1.add_column("Conf")
        for s in sigs:
            t1.add_row(
                format_display_time(s.signal_time, self.settings.app.timezone),
                s.kol_username,
                s.symbol,
                s.direction.value,
                f"{s.entry_price:g}" if s.entry_price else "-",
                f"{s.stop_loss:g}" if s.stop_loss else "-",
                f"{s.take_profit:g}" if s.take_profit else "-",
                s.entry_mode.value,
                s.state.value,
                f"{s.current_price:g}" if s.current_price else "-",
                f"{s.confidence:.2f}",
            )
        console.print(t1)

        # Notes
        notes = self.storage.list_notes(limit=50)
        t2 = Table(title="标的描述 / 长期观点积累")
        t2.add_column("Symbol", style="cyan")
        t2.add_column("Note")
        for n in notes:
            t2.add_row(n.symbol, n.content[:120])
        console.print(t2)

        # Trades
        trades = self.storage.list_trades()
        t3 = Table(title="模拟成交记录")
        t3.add_column("KOL")
        t3.add_column("Symbol")
        t3.add_column("Dir")
        t3.add_column("Entry")
        t3.add_column("Exit")
        t3.add_column("Status")
        t3.add_column("PnL%")
        t3.add_column("PnL$")
        for tr in trades:
            t3.add_row(
                tr.kol_username,
                tr.symbol,
                tr.direction.value,
                f"{tr.entry_price:.4g}",
                f"{tr.exit_price:.4g}" if tr.exit_price else "-",
                tr.status.value,
                f"{tr.pnl_pct:.2f}" if tr.pnl_pct is not None else "-",
                f"{tr.pnl_usd:.2f}" if tr.pnl_usd is not None else "-",
            )
        console.print(t3)

        # KOL stats
        t4 = Table(title="KOL 跟单胜率")
        t4.add_column("KOL")
        t4.add_column("Closed")
        t4.add_column("W/L")
        t4.add_column("WinRate")
        t4.add_column("PnL$")
        t4.add_column("Summary")
        for row in summary.get("stats", {}).get("kol", []):
            t4.add_row(
                row["kol"],
                str(row["closed"]),
                f"{row['wins']}/{row['losses']}",
                f"{row['win_rate']:.1%}",
                f"{row['total_pnl_usd']:.2f}",
                row["summary"],
            )
        overall = summary.get("stats", {}).get("overall")
        if overall:
            t4.add_row(
                overall["kol"],
                str(overall["closed"]),
                f"{overall['wins']}/{overall['losses']}",
                f"{overall['win_rate']:.1%}",
                f"{overall['total_pnl_usd']:.2f}",
                overall["summary"],
            )
        console.print(t4)

        # Symbol snapshots for BTC as demo
        for sym in ("BTC-USD", "SNDK", "NVDA"):
            snap = self.storage.symbol_snapshot(sym, self.settings.app.timezone)
            if snap["structured_signals"] or snap["notes"]:
                console.rule(f"[bold]{sym} 信息积累")
                for line in snap["note_timeline"][:8]:
                    console.print(f"  {line}")
                for s in snap["structured_signals"][:5]:
                    console.print(
                        f"  信号: {s['direction']} entry={s.get('entry_price')} "
                        f"SL={s.get('stop_loss')} TP={s.get('take_profit')} "
                        f"by {s['kol_username']}"
                    )
