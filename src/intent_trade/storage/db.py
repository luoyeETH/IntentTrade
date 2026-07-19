"""SQLite persistence for posts, signals, notes, paper trades."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    inspect,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    InstrumentNote,
    PaperTrade,
    PositionState,
    SignalState,
    SocialPost,
    TradeSide,
    TradeStatus,
    TradingSignal,
)
from intent_trade.time_utils import format_display_time

metadata = MetaData()

posts_t = Table(
    "social_posts",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("platform", String(32)),
    Column("author_username", String(128), index=True),
    Column("author_display_name", String(256)),
    Column("text", Text),
    Column("created_at", DateTime, index=True),
    Column("url", String(512)),
    Column("media_urls_json", Text),
    Column("media_alt_texts_json", Text),
    Column("media_transcripts_json", Text),
    Column("raw_json", Text),
    Column("fetched_at", DateTime),
)

signals_t = Table(
    "trading_signals",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("post_id", String(32), index=True),
    Column("kol_username", String(128), index=True),
    Column("symbol", String(64), index=True),
    Column("direction", String(16)),
    Column("action", String(16)),
    Column("position_state", String(16)),
    Column("entry_mode", String(16)),
    Column("entry_price", Float, nullable=True),
    Column("entry_price_low", Float, nullable=True),
    Column("entry_price_high", Float, nullable=True),
    Column("trigger_price", Float, nullable=True),
    Column("stop_loss", Float, nullable=True),
    Column("take_profit", Float, nullable=True),
    Column("take_profit_levels_json", Text),
    Column("entry_condition", Text),
    Column("time_horizon", String(32)),
    Column("expires_at", DateTime, nullable=True),
    Column("confidence", Float),
    Column("summary", Text),
    Column("source_text", Text),
    Column("analyzer", String(32)),
    Column("reasoning", Text),
    Column("field_confidence_json", Text),
    Column("evidence_json", Text),
    Column("signal_time", DateTime, index=True),
    Column("created_at", DateTime),
    Column("executed", Boolean, default=False),
    Column("state", String(32)),
    Column("current_price", Float, nullable=True),
    Column("market_timestamp", DateTime, nullable=True),
    Column("market_source", String(64)),
    Column("market_is_live", Boolean, nullable=True),
    Column("price_distance_pct", Float, nullable=True),
    Column("decision_reason", Text),
    Column("last_evaluated_at", DateTime, nullable=True),
)

notes_t = Table(
    "instrument_notes",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("symbol", String(64), index=True),
    Column("kol_username", String(128), index=True),
    Column("post_id", String(32)),
    Column("note_time", DateTime, index=True),
    Column("content", Text),
    Column("direction_hint", String(16)),
    Column("confidence", Float),
    Column("tags_json", Text),
    Column("created_at", DateTime),
)

trades_t = Table(
    "paper_trades",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("signal_id", String(32), index=True),
    Column("kol_username", String(128), index=True),
    Column("symbol", String(64), index=True),
    Column("side", String(8)),
    Column("direction", String(16)),
    Column("quantity", Float),
    Column("entry_price", Float),
    Column("stop_loss", Float, nullable=True),
    Column("take_profit", Float, nullable=True),
    Column("status", String(32), index=True),
    Column("entry_time", DateTime, index=True),
    Column("exit_time", DateTime, nullable=True),
    Column("exit_price", Float, nullable=True),
    Column("pnl_usd", Float, nullable=True),
    Column("pnl_pct", Float, nullable=True),
    Column("commission_usd", Float),
    Column("notes", Text),
    Column("created_at", DateTime),
)

analyses_t = Table(
    "intent_analyses",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("post_id", String(32), index=True),
    Column("kol_username", String(128)),
    Column("payload_json", Text),
    Column("analyzed_at", DateTime),
)


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _uj(s: Optional[str], default: Any = None) -> Any:
    if not s:
        return default if default is not None else []
    return json.loads(s)


_NOTE_PREFIX = re.compile(
    r"^【\d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2} ([^:：]+)[:：] (.*)】$",
    re.DOTALL,
)


def _display_note_content(
    content: str, note_time: datetime, timezone_name: str
) -> str:
    """Rewrite legacy embedded UTC prefixes for consistent presentation."""

    match = _NOTE_PREFIX.match(content or "")
    if not match:
        return content
    author, body = match.groups()
    stamp = format_display_time(note_time, timezone_name)
    return f"【{stamp} {author}: {body}】"


_SIGNAL_MIGRATIONS = {
    "action": "VARCHAR(16)",
    "position_state": "VARCHAR(16)",
    "entry_mode": "VARCHAR(16)",
    "entry_price_low": "FLOAT",
    "entry_price_high": "FLOAT",
    "trigger_price": "FLOAT",
    "entry_condition": "TEXT",
    "time_horizon": "VARCHAR(32)",
    "expires_at": "DATETIME",
    "state": "VARCHAR(32)",
    "current_price": "FLOAT",
    "market_timestamp": "DATETIME",
    "market_source": "VARCHAR(64)",
    "market_is_live": "BOOLEAN",
    "price_distance_pct": "FLOAT",
    "decision_reason": "TEXT",
    "last_evaluated_at": "DATETIME",
    "analyzer": "VARCHAR(32)",
    "reasoning": "TEXT",
    "field_confidence_json": "TEXT",
    "evidence_json": "TEXT",
}

_POST_MIGRATIONS = {
    "media_alt_texts_json": "TEXT",
}


class Storage:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            f"sqlite:///{db_path}",
            future=True,
            connect_args={"check_same_thread": False},
        )
        metadata.create_all(self.engine)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Add newer fields without dropping or rewriting the existing journal."""

        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            for table_name, migrations in (
                ("social_posts", _POST_MIGRATIONS),
                ("trading_signals", _SIGNAL_MIGRATIONS),
            ):
                existing = {
                    column["name"]
                    for column in inspector.get_columns(table_name)
                }
                for name, sql_type in migrations.items():
                    if name not in existing:
                        conn.exec_driver_sql(
                            f"ALTER TABLE {table_name} ADD COLUMN {name} {sql_type}"
                        )

    def insert_post(self, post: SocialPost) -> bool:
        """Archive the first fetched snapshot of a post without overwriting it.

        A later timeline response may omit a deleted post or return changed or
        incomplete metadata. The archive is append-only by post id, so the first
        complete snapshot remains available for analysis and review.
        """

        row = {
            "id": post.id,
            "platform": post.platform,
            "author_username": post.author_username,
            "author_display_name": post.author_display_name,
            "text": post.text,
            "created_at": post.created_at,
            "url": post.url,
            "media_urls_json": _j(post.media_urls),
            "media_alt_texts_json": _j(post.media_alt_texts),
            "media_transcripts_json": _j(post.media_transcripts),
            "raw_json": _j(post.raw),
            "fetched_at": post.fetched_at,
        }
        with self.engine.begin() as conn:
            result = conn.execute(
                sqlite_insert(posts_t)
                .values(**row)
                .on_conflict_do_nothing(index_elements=[posts_t.c.id])
            )
        return result.rowcount == 1

    def upsert_post(self, post: SocialPost) -> bool:
        """Backward-compatible alias; existing post snapshots stay immutable."""

        return self.insert_post(post)

    def list_posts(
        self, username: Optional[str] = None, limit: int = 200
    ) -> list[SocialPost]:
        q = select(posts_t).order_by(posts_t.c.created_at.desc()).limit(limit)
        if username:
            q = q.where(posts_t.c.author_username == username)
        with self.engine.begin() as conn:
            rows = conn.execute(q).mappings().all()
        return [
            SocialPost(
                id=r["id"],
                platform=r["platform"],
                author_username=r["author_username"],
                author_display_name=r["author_display_name"] or "",
                text=r["text"],
                created_at=r["created_at"],
                url=r["url"],
                media_urls=_uj(r["media_urls_json"], []),
                media_alt_texts=_uj(r["media_alt_texts_json"], []),
                media_transcripts=_uj(r["media_transcripts_json"], []),
                raw=_uj(r["raw_json"], {}),
                fetched_at=r["fetched_at"] or datetime.utcnow(),
            )
            for r in rows
        ]

    def post_exists(self, post_id: str) -> bool:
        with self.engine.begin() as conn:
            return (
                conn.execute(
                    select(posts_t.c.id).where(posts_t.c.id == post_id)
                ).first()
                is not None
            )

    def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Fetch one post by id, or None if missing."""

        if not post_id:
            return None
        with self.engine.begin() as conn:
            r = (
                conn.execute(select(posts_t).where(posts_t.c.id == post_id))
                .mappings()
                .first()
            )
        if not r:
            return None
        return SocialPost(
            id=r["id"],
            platform=r["platform"],
            author_username=r["author_username"],
            author_display_name=r["author_display_name"] or "",
            text=r["text"],
            created_at=r["created_at"],
            url=r["url"],
            media_urls=_uj(r["media_urls_json"], []),
            media_alt_texts=_uj(r["media_alt_texts_json"], []),
            media_transcripts=_uj(r["media_transcripts_json"], []),
            raw=_uj(r["raw_json"], {}),
            fetched_at=r["fetched_at"] or datetime.utcnow(),
        )

    def get_posts_by_ids(self, post_ids: list[str]) -> dict[str, SocialPost]:
        """Batch-load posts keyed by id (missing ids omitted)."""

        ids = [str(x) for x in post_ids if x]
        if not ids:
            return {}
        with self.engine.begin() as conn:
            rows = (
                conn.execute(select(posts_t).where(posts_t.c.id.in_(ids)))
                .mappings()
                .all()
            )
        out: dict[str, SocialPost] = {}
        for r in rows:
            out[r["id"]] = SocialPost(
                id=r["id"],
                platform=r["platform"],
                author_username=r["author_username"],
                author_display_name=r["author_display_name"] or "",
                text=r["text"],
                created_at=r["created_at"],
                url=r["url"],
                media_urls=_uj(r["media_urls_json"], []),
                media_alt_texts=_uj(r["media_alt_texts_json"], []),
                media_transcripts=_uj(r["media_transcripts_json"], []),
                raw=_uj(r["raw_json"], {}),
                fetched_at=r["fetched_at"] or datetime.utcnow(),
            )
        return out

    def counts(self) -> dict[str, int]:
        """Return journal counts without truncating the dashboard metrics."""

        with self.engine.begin() as conn:
            return {
                "posts": int(conn.scalar(select(func.count()).select_from(posts_t)) or 0),
                "signals": int(conn.scalar(select(func.count()).select_from(signals_t)) or 0),
                "notes": int(conn.scalar(select(func.count()).select_from(notes_t)) or 0),
                "trades": int(conn.scalar(select(func.count()).select_from(trades_t)) or 0),
            }

    def save_analysis(self, post_id: str, kol: str, payload: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                analyses_t.insert().values(
                    post_id=post_id,
                    kol_username=kol,
                    payload_json=_j(payload),
                    analyzed_at=datetime.utcnow(),
                )
            )

    def insert_signal(self, sig: TradingSignal) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                signals_t.insert().values(
                    id=sig.id,
                    post_id=sig.post_id,
                    kol_username=sig.kol_username,
                    symbol=sig.symbol,
                    direction=sig.direction.value,
                    action=sig.action.value,
                    position_state=sig.position_state.value,
                    entry_mode=sig.entry_mode.value,
                    entry_price=sig.entry_price,
                    entry_price_low=sig.entry_price_low,
                    entry_price_high=sig.entry_price_high,
                    trigger_price=sig.trigger_price,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    take_profit_levels_json=_j(sig.take_profit_levels),
                    entry_condition=sig.entry_condition,
                    time_horizon=sig.time_horizon,
                    expires_at=sig.expires_at,
                    confidence=sig.confidence,
                    summary=sig.summary,
                    source_text=sig.source_text,
                    analyzer=sig.analyzer,
                    reasoning=sig.reasoning,
                    field_confidence_json=_j(sig.field_confidence),
                    evidence_json=_j(sig.evidence),
                    signal_time=sig.signal_time,
                    created_at=sig.created_at,
                    executed=sig.executed,
                    state=sig.state.value,
                    current_price=sig.current_price,
                    market_timestamp=sig.market_timestamp,
                    market_source=sig.market_source,
                    market_is_live=sig.market_is_live,
                    price_distance_pct=sig.price_distance_pct,
                    decision_reason=sig.decision_reason,
                    last_evaluated_at=sig.last_evaluated_at,
                )
            )

    def mark_signal_executed(self, signal_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(signals_t)
                .where(signals_t.c.id == signal_id)
                .values(executed=True)
            )

    def update_signal_decision(self, sig: TradingSignal) -> None:
        """Persist the latest market evaluation without changing the intent."""

        with self.engine.begin() as conn:
            conn.execute(
                update(signals_t)
                .where(signals_t.c.id == sig.id)
                .values(
                    state=sig.state.value,
                    current_price=sig.current_price,
                    market_timestamp=sig.market_timestamp,
                    market_source=sig.market_source,
                    market_is_live=sig.market_is_live,
                    price_distance_pct=sig.price_distance_pct,
                    decision_reason=sig.decision_reason,
                    last_evaluated_at=sig.last_evaluated_at,
                )
            )

    def mark_signals_superseded(
        self,
        signal_ids: list[str],
        *,
        replacement_post_id: str,
        reason: str,
    ) -> list[str]:
        """Stop eligible pending signals after a later post revises the plan."""

        ids = list(dict.fromkeys(str(value) for value in signal_ids if value))
        if not ids:
            return []
        terminal_states = {
            SignalState.EXECUTED.value,
            SignalState.SUPERSEDED.value,
            SignalState.EXPIRED.value,
        }
        with self.engine.begin() as conn:
            eligible = list(
                conn.scalars(
                    select(signals_t.c.id).where(
                        signals_t.c.id.in_(ids),
                        signals_t.c.executed.is_(False),
                        signals_t.c.state.not_in(terminal_states),
                    )
                )
            )
            if not eligible:
                return []
            detail = reason.strip() or "后续推文更新了交易计划"
            conn.execute(
                update(signals_t)
                .where(signals_t.c.id.in_(eligible))
                .values(
                    state=SignalState.SUPERSEDED.value,
                    decision_reason=(
                        f"{detail}；replacement_post={replacement_post_id}"
                    ),
                    last_evaluated_at=datetime.utcnow(),
                )
            )
        return eligible

    def list_signals(
        self,
        symbol: Optional[str] = None,
        unexecuted_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[TradingSignal]:
        q = select(signals_t).order_by(signals_t.c.signal_time.desc())
        if symbol:
            q = q.where(signals_t.c.symbol == symbol)
        if unexecuted_only:
            q = q.where(
                signals_t.c.executed.is_(False),
                signals_t.c.state != SignalState.SUPERSEDED.value,
            )
        if limit is not None:
            q = q.limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(q).mappings().all()
        return [
            TradingSignal(
                id=r["id"],
                post_id=r["post_id"],
                kol_username=r["kol_username"],
                symbol=r["symbol"],
                direction=Direction(r["direction"]),
                action=IntentAction(str(r["action"] or "open")),
                position_state=PositionState(str(r["position_state"] or "planned")),
                entry_mode=EntryMode(str(r["entry_mode"] or "unknown")),
                entry_price=r["entry_price"],
                entry_price_low=r["entry_price_low"],
                entry_price_high=r["entry_price_high"],
                trigger_price=r["trigger_price"],
                stop_loss=r["stop_loss"],
                take_profit=r["take_profit"],
                take_profit_levels=_uj(r["take_profit_levels_json"], []),
                entry_condition=r["entry_condition"] or "",
                time_horizon=r["time_horizon"] or "",
                expires_at=r["expires_at"],
                confidence=r["confidence"],
                summary=r["summary"] or "",
                source_text=r["source_text"] or "",
                analyzer=r["analyzer"] or "unknown",
                reasoning=r["reasoning"] or "",
                field_confidence=_uj(r["field_confidence_json"], {}),
                evidence=_uj(r["evidence_json"], {}),
                signal_time=r["signal_time"],
                created_at=r["created_at"],
                executed=bool(r["executed"]),
                state=SignalState(
                    str(
                        r["state"]
                        or ("executed" if r["executed"] else "waiting_market_data")
                    )
                ),
                current_price=r["current_price"],
                market_timestamp=r["market_timestamp"],
                market_source=r["market_source"] or "",
                market_is_live=r["market_is_live"],
                price_distance_pct=r["price_distance_pct"],
                decision_reason=r["decision_reason"] or "",
                last_evaluated_at=r["last_evaluated_at"],
            )
            for r in rows
        ]

    def insert_note(self, note: InstrumentNote) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                notes_t.insert().values(
                    id=note.id,
                    symbol=note.symbol,
                    kol_username=note.kol_username,
                    post_id=note.post_id,
                    note_time=note.note_time,
                    content=note.content,
                    direction_hint=note.direction_hint.value,
                    confidence=note.confidence,
                    tags_json=_j(note.tags),
                    created_at=note.created_at,
                )
            )

    def list_notes(
        self,
        symbol: Optional[str] = None,
        limit: int = 200,
        timezone_name: str = "Asia/Shanghai",
    ) -> list[InstrumentNote]:
        q = select(notes_t).order_by(notes_t.c.note_time.desc()).limit(limit)
        if symbol:
            q = q.where(notes_t.c.symbol == symbol)
        with self.engine.begin() as conn:
            rows = conn.execute(q).mappings().all()
        return [
            InstrumentNote(
                id=r["id"],
                symbol=r["symbol"],
                kol_username=r["kol_username"],
                post_id=r["post_id"],
                note_time=r["note_time"],
                content=_display_note_content(
                    r["content"], r["note_time"], timezone_name
                ),
                direction_hint=Direction(r["direction_hint"]),
                confidence=r["confidence"],
                tags=_uj(r["tags_json"], []),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def insert_trade(self, trade: PaperTrade) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                trades_t.insert().values(
                    id=trade.id,
                    signal_id=trade.signal_id,
                    kol_username=trade.kol_username,
                    symbol=trade.symbol,
                    side=trade.side.value,
                    direction=trade.direction.value,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    stop_loss=trade.stop_loss,
                    take_profit=trade.take_profit,
                    status=trade.status.value,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    exit_price=trade.exit_price,
                    pnl_usd=trade.pnl_usd,
                    pnl_pct=trade.pnl_pct,
                    commission_usd=trade.commission_usd,
                    notes=trade.notes,
                    created_at=trade.created_at,
                )
            )

    def update_trade(self, trade: PaperTrade) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(trades_t)
                .where(trades_t.c.id == trade.id)
                .values(
                    status=trade.status.value,
                    exit_time=trade.exit_time,
                    exit_price=trade.exit_price,
                    pnl_usd=trade.pnl_usd,
                    pnl_pct=trade.pnl_pct,
                    commission_usd=trade.commission_usd,
                    notes=trade.notes,
                )
            )

    def list_trades(
        self,
        kol: Optional[str] = None,
        status: Optional[TradeStatus] = None,
        symbol: Optional[str] = None,
    ) -> list[PaperTrade]:
        q = select(trades_t).order_by(trades_t.c.entry_time.desc())
        if kol:
            q = q.where(trades_t.c.kol_username == kol)
        if status:
            q = q.where(trades_t.c.status == status.value)
        if symbol:
            q = q.where(trades_t.c.symbol == symbol)
        with self.engine.begin() as conn:
            rows = conn.execute(q).mappings().all()
        return [
            PaperTrade(
                id=r["id"],
                signal_id=r["signal_id"],
                kol_username=r["kol_username"],
                symbol=r["symbol"],
                side=TradeSide(r["side"]),
                direction=Direction(r["direction"]),
                quantity=r["quantity"],
                entry_price=r["entry_price"],
                stop_loss=r["stop_loss"],
                take_profit=r["take_profit"],
                status=TradeStatus(r["status"]),
                entry_time=r["entry_time"],
                exit_time=r["exit_time"],
                exit_price=r["exit_price"],
                pnl_usd=r["pnl_usd"],
                pnl_pct=r["pnl_pct"],
                commission_usd=r["commission_usd"] or 0.0,
                notes=r["notes"] or "",
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def symbol_snapshot(
        self, symbol: str, timezone_name: str = "Asia/Shanghai"
    ) -> dict[str, Any]:
        """Build the accumulated view: structured signals + descriptive notes."""
        signals = self.list_signals(symbol=symbol)
        notes = self.list_notes(symbol=symbol, timezone_name=timezone_name)
        trades = self.list_trades(symbol=symbol)
        return {
            "symbol": symbol,
            "structured_signals": [s.model_dump(mode="python") for s in signals],
            "notes": [n.model_dump(mode="python") for n in notes],
            "trades": [t.model_dump(mode="python") for t in trades],
            "note_timeline": [
                f"[{format_display_time(n.note_time, timezone_name)}] "
                f"{n.kol_username}: {n.content}"
                for n in notes
            ],
        }
