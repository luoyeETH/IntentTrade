"""Canonical instrument registry + alias learning (LLM can add slang → symbol)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class InstrumentMeta:
    symbol: str
    name: str = ""
    asset_class: str = "other"
    aliases: list[str] = field(default_factory=list)
    yfinance_symbol: Optional[str] = None


class TickerMap:
    def __init__(self, path: Path, learned_path: Path | None = None) -> None:
        self.path = path
        self.learned_path = learned_path or (
            path.parent / "ticker_aliases.learned.yaml"
        )
        self.by_symbol: dict[str, InstrumentMeta] = {}
        self._alias_index: dict[str, str] = {}
        self._load()
        self._load_learned()

    def _norm(self, s: object) -> str:
        s = str(s).strip().lower()
        s = s.lstrip("$")
        s = re.sub(r"\s+", "", s)
        return s

    def _register(self, symbol: str, aliases: list[str], *, name: str = "", asset_class: str = "other", yfinance_symbol: Optional[str] = None) -> None:
        symbol = str(symbol)
        clean_aliases: list[str] = []
        for a in aliases:
            a = str(a).strip()
            if a and a not in clean_aliases:
                clean_aliases.append(a)
        if symbol not in clean_aliases:
            clean_aliases.append(symbol)
        if symbol in self.by_symbol:
            im = self.by_symbol[symbol]
            for a in clean_aliases:
                if a not in im.aliases:
                    im.aliases.append(a)
            if name and not im.name:
                im.name = name
        else:
            self.by_symbol[symbol] = InstrumentMeta(
                symbol=symbol,
                name=name or symbol,
                asset_class=asset_class,
                aliases=clean_aliases,
                yfinance_symbol=yfinance_symbol,
            )
        for a in self.by_symbol[symbol].aliases:
            self._alias_index[self._norm(a)] = symbol

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        for symbol, meta in raw.items():
            if not isinstance(meta, dict):
                continue
            aliases = [str(a) for a in (meta.get("aliases") or [])]
            if meta.get("name"):
                aliases.append(str(meta["name"]))
            yf_sym = meta.get("yfinance_symbol")
            self._register(
                str(symbol),
                aliases,
                name=str(meta.get("name") or symbol),
                asset_class=str(meta.get("asset_class") or "other"),
                yfinance_symbol=str(yf_sym) if yf_sym is not None else None,
            )

    def _load_learned(self) -> None:
        if not self.learned_path.exists():
            return
        raw = yaml.safe_load(self.learned_path.read_text(encoding="utf-8")) or {}
        # format: { "大饼": {"symbol": "BTC-USD", "reason": "...", "at": "..."}, ...}
        if not isinstance(raw, dict):
            return
        for alias, meta in raw.items():
            if isinstance(meta, dict):
                symbol = str(meta.get("symbol") or "")
            else:
                symbol = str(meta)
            if alias and symbol:
                self._register(symbol, [str(alias)])

    def learn_alias(
        self,
        alias: str,
        symbol: str,
        *,
        reason: str = "",
        persist: bool = True,
    ) -> None:
        """Attach slang/alias to an existing canonical symbol (AI-driven growth)."""
        alias = str(alias).strip()
        symbol = str(symbol).strip()
        if not alias or not symbol:
            return
        # resolve symbol if alias of another
        symbol = self.resolve(symbol) or symbol
        if symbol not in self.by_symbol:
            # do not invent brand-new instruments here; only attach to known
            return
        self._register(symbol, [alias])
        if persist:
            self._persist_learned(alias, symbol, reason=reason)

    def _persist_learned(self, alias: str, symbol: str, reason: str = "") -> None:
        data: dict = {}
        if self.learned_path.exists():
            data = yaml.safe_load(self.learned_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                data = {}
        data[alias] = {
            "symbol": symbol,
            "reason": reason,
            "at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self.learned_path.parent.mkdir(parents=True, exist_ok=True)
        self.learned_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )

    def resolve(self, token: str) -> Optional[str]:
        key = self._norm(token)
        return self._alias_index.get(key)

    def yfinance_symbol(self, symbol: str) -> str:
        im = self.by_symbol.get(symbol)
        if im and im.yfinance_symbol:
            return im.yfinance_symbol
        return symbol

    def yfinance_map(self) -> dict[str, str]:
        return {
            sym: (im.yfinance_symbol or sym) for sym, im in self.by_symbol.items()
        }

    def asset_class_map(self) -> dict[str, str]:
        return {sym: (im.asset_class or "other") for sym, im in self.by_symbol.items()}

    def asset_class_of(self, symbol: str) -> str:
        resolved = self.resolve(symbol) or symbol
        im = self.by_symbol.get(resolved)
        return (im.asset_class if im else "other") or "other"

    def catalog_for_prompt(self) -> list[dict]:
        return [
            {
                "symbol": s,
                "name": m.name,
                "aliases": m.aliases[:20],
            }
            for s, m in self.by_symbol.items()
        ]

    def find_in_text(self, text: str) -> list[str]:
        """Secondary helper: substring/alias scan. Primary path should be LLM."""
        if not text:
            return []
        pairs = sorted(self._alias_index.items(), key=lambda kv: len(kv[0]), reverse=True)
        found: list[str] = []
        seen: set[str] = set()
        lower = text
        collapsed = re.sub(r"\s+", "", text)
        for alias_n, symbol in pairs:
            if symbol in seen:
                continue
            if re.fullmatch(r"[a-z0-9.\-]+", alias_n):
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(alias_n)}(?![A-Za-z0-9])",
                    re.IGNORECASE,
                )
                if pattern.search(lower) or pattern.search(collapsed):
                    found.append(symbol)
                    seen.add(symbol)
            else:
                if alias_n in collapsed.lower() or alias_n in lower.lower():
                    found.append(symbol)
                    seen.add(symbol)
        return found

