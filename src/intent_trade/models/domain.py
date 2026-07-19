"""Domain models for IntentTrade.

Intent extraction is kept separate from execution. A KOL can describe a
plan, an already-open position, or an exit without that text becoming an
immediate paper fill.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _uid() -> str:
    return uuid4().hex[:16]


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    UNKNOWN = "unknown"


class SignalType(str, Enum):
    """Structured actionable signal vs soft descriptive note."""

    STRUCTURED = "structured"
    DESCRIPTIVE = "descriptive"


class IntentAction(str, Enum):
    """What the KOL is trying to do with the instrument."""

    OPEN = "open"
    ADD = "add"
    CLOSE = "close"
    REDUCE = "reduce"
    HOLD = "hold"
    WATCH = "watch"
    UNKNOWN = "unknown"


class PositionState(str, Enum):
    """Whether a post describes a plan or an already-held position."""

    PLANNED = "planned"
    ENTERED = "entered"
    EXITING = "exiting"
    UNKNOWN = "unknown"


class EntryMode(str, Enum):
    """How an open/add instruction should be entered."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    RANGE = "range"
    UNKNOWN = "unknown"


class SignalState(str, Enum):
    """Lifecycle state after combining intent with current market data."""

    WAITING_MARKET_DATA = "waiting_market_data"
    WAITING_ENTRY = "waiting_entry"
    READY = "ready"
    WAITING_RISK_LIMIT = "waiting_risk_limit"
    OBSERVED_POSITION = "observed_position"
    EXIT_INTENT = "exit_intent"
    EXECUTED = "executed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    REJECTED = "rejected"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_MANUAL = "closed_manual"
    EXPIRED = "expired"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    OTHER = "other"


class KOL(BaseModel):
    username: str
    display_name: str = ""
    weight: float = 1.0
    enabled: bool = True


class SocialPost(BaseModel):
    id: str = Field(default_factory=_uid)
    platform: str = "twitter"
    author_username: str
    author_display_name: str = ""
    text: str
    created_at: datetime
    url: Optional[str] = None
    media_urls: list[str] = Field(default_factory=list)
    media_alt_texts: list[str] = Field(default_factory=list)
    # Legacy imported captions. The active analyzer reads media_urls directly.
    media_transcripts: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class IntentAnalysis(BaseModel):
    """Raw analysis output before persistence split."""

    post_id: str
    kol_username: str
    raw_text: str
    # Source text used by the text stage; image evidence stays structured.
    analysis_text: str = ""
    mentioned_tickers: list[str] = Field(default_factory=list)
    canonical_symbols: list[str] = Field(default_factory=list)
    direction: Direction = Direction.UNKNOWN
    action: IntentAction = IntentAction.UNKNOWN
    position_state: PositionState = PositionState.UNKNOWN
    entry_mode: EntryMode = EntryMode.UNKNOWN
    signal_type: SignalType = SignalType.DESCRIPTIVE
    entry_price: Optional[float] = None
    entry_price_low: Optional[float] = None
    entry_price_high: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_levels: list[float] = Field(default_factory=list)
    entry_condition: str = ""
    time_horizon: str = ""
    validity_hours: Optional[float] = None
    confidence: float = 0.0
    field_confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    descriptive_note: str = ""
    plan_text: str = ""
    reasoning: str = ""
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    analyzer: str = "rule_based"  # rule_based | llm | hybrid


class TradingSignal(BaseModel):
    """Persisted intent plus its latest execution decision.

    ``entry_price`` is the KOL's requested/reference price. The actual paper
    fill is stored separately on :class:`PaperTrade` and can be better than a
    long limit or short limit when the market has moved through it.
    """

    id: str = Field(default_factory=_uid)
    post_id: str
    kol_username: str
    symbol: str
    direction: Direction
    action: IntentAction = IntentAction.OPEN
    position_state: PositionState = PositionState.PLANNED
    entry_mode: EntryMode = EntryMode.UNKNOWN
    entry_price: Optional[float] = None
    entry_price_low: Optional[float] = None
    entry_price_high: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_levels: list[float] = Field(default_factory=list)
    entry_condition: str = ""
    time_horizon: str = ""
    expires_at: Optional[datetime] = None
    confidence: float = 0.0
    summary: str = ""
    source_text: str = ""
    analyzer: str = "unknown"
    reasoning: str = ""
    field_confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)
    signal_time: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    executed: bool = False
    state: SignalState = SignalState.WAITING_MARKET_DATA
    current_price: Optional[float] = None
    market_timestamp: Optional[datetime] = None
    market_source: str = ""
    market_is_live: Optional[bool] = None
    price_distance_pct: Optional[float] = None
    decision_reason: str = ""
    last_evaluated_at: Optional[datetime] = None


class InstrumentNote(BaseModel):
    """Uncertain / long-term / analytical text attached to a symbol."""

    id: str = Field(default_factory=_uid)
    symbol: str
    kol_username: str
    post_id: str
    note_time: datetime
    content: str
    direction_hint: Direction = Direction.UNKNOWN
    confidence: float = 0.0
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MarketBar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MarketSnapshot(BaseModel):
    """Latest quote used for a decision, including provenance and freshness."""

    symbol: str
    price: Optional[float] = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    source: str = "unknown"
    is_live: bool = False
    stale: bool = False
    age_seconds: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    previous_close: Optional[float] = None
    currency: Optional[str] = None
    error: Optional[str] = None


class SignalDecision(BaseModel):
    """Pure decision result; it never places an order by itself."""

    state: SignalState
    can_execute: bool = False
    current_price: Optional[float] = None
    market_timestamp: Optional[datetime] = None
    market_source: str = ""
    market_is_live: Optional[bool] = None
    price_distance_pct: Optional[float] = None
    reason: str = ""
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class PaperTrade(BaseModel):
    id: str = Field(default_factory=_uid)
    signal_id: str
    kol_username: str
    symbol: str
    side: TradeSide
    direction: Direction
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission_usd: float = 0.0
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
