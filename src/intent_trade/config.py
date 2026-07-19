"""Load YAML config + environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from intent_trade.models.domain import KOL

ROOT = Path(__file__).resolve().parents[2]


class TwitterConfig(BaseModel):
    source: str = "mock"
    poll_interval_seconds: int = 300
    max_posts_per_kol: int = 50
    include_media: bool = True
    # When True, `intent-trade serve` runs a background fetch→analyze→settle loop
    # so tweets keep updating without a browser open or manual 拉帖.
    auto_poll: bool = True
    auto_poll_max_analyze: int = 10


class MarketConfig(BaseModel):
    default_interval: str = "1d"
    history_days: int = 120
    cache_ttl_seconds: int = 300
    quote_ttl_seconds: int = 30
    max_quote_age_seconds: int = 900
    allow_fallback: bool = True
    require_live_for_execution: bool = True
    # K-line chart sources (independent of paper-fill quote path)
    kline_cache_ttl_seconds: int = 30
    # crypto preferred provider: binance (data-api.binance.vision) | okx
    kline_crypto_provider: str = "binance"
    # Default: public data host (works without proxy). With INTENT_TRADE_HTTP_PROXY
    # you can point this at https://api.binance.com for the main market API.
    binance_data_base_url: str = "https://data-api.binance.vision"
    okx_base_url: str = "https://www.okx.com"
    kline_default_limit: int = 300
    # Optional override; otherwise INTENT_TRADE_HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
    http_proxy: str = ""
    # Route OKX through the same proxy (usually unnecessary)
    kline_proxy_okx: bool = False


class AnalysisConfig(BaseModel):
    mode: str = "llm"
    llm_model: str = "grok-4.5"
    confidence_threshold: float = 0.55
    structured_min_confidence: float = 0.7
    pending_signal_ttl_hours: float = 72.0
    default_entry_mode: str = "limit"
    memory_enabled: bool = True
    memory_lookback_hours: float = 168.0
    memory_max_items: int = 6
    memory_min_confidence: float = 0.75
    agent_tools_enabled: bool = True
    agent_max_rounds: int = 8
    agent_price_lookback_days: int = 365


class ExecutionConfig(BaseModel):
    mode: str = "paper"
    default_position_size_usd: float = 1000
    fill_price: str = "close"
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.08
    max_open_trades_per_symbol: int = 3
    commission_bps: float = 5
    entry_tolerance_pct: float = 0.0


class BacktestConfig(BaseModel):
    mark_to_market: bool = True
    path_resolution: str = "conservative"


class AppConfig(BaseModel):
    name: str = "IntentTrade"
    timezone: str = "Asia/Shanghai"
    data_dir: str = "data"
    db_path: str = "data/db/intent_trade.db"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    kols: list[KOL] = Field(default_factory=list)
    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    root: Path = ROOT
    ticker_aliases_path: Path = ROOT / "config" / "ticker_aliases.yaml"

    @property
    def db_path(self) -> Path:
        p = Path(os.getenv("INTENT_TRADE_DB_PATH", self.app.db_path))
        if not p.is_absolute():
            p = self.root / p
        return p

    @property
    def data_dir(self) -> Path:
        p = Path(self.app.data_dir)
        if not p.is_absolute():
            p = self.root / p
        return p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    path = Path(
        config_path
        or os.getenv("INTENT_TRADE_CONFIG")
        or (ROOT / "config" / "settings.yaml")
    )
    if not path.is_absolute():
        path = ROOT / path
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return Settings(**raw)
