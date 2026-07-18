"""Tests: ticker registry + AI path structure (rules only as fallback unit)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_trade.analysis.intent import IntentAnalyzer
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.config import AnalysisConfig
from intent_trade.models.domain import Direction, SocialPost


def test_daping_maps_to_btc():
    tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml")
    assert tm.resolve("大饼") == "BTC-USD"
    assert tm.resolve("比特币") == "BTC-USD"
    assert tm.resolve("闪迪") == "SNDK"


def test_learn_alias_persists(tmp_path: Path | None = None):
    learned = ROOT / "config" / "ticker_aliases.learned.yaml"
    # use isolated learned file
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "learned.yaml"
        tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml", learned_path=p)
        tm.learn_alias("大饼饼", "BTC-USD", reason="unit-test", persist=True)
        assert tm.resolve("大饼饼") == "BTC-USD"
        tm2 = TickerMap(ROOT / "config" / "ticker_aliases.yaml", learned_path=p)
        assert tm2.resolve("大饼饼") == "BTC-USD"


def test_rule_fallback_does_not_crash():
    tm = TickerMap(ROOT / "config" / "ticker_aliases.yaml")
    az = IntentAnalyzer(tm, AnalysisConfig(mode="rule_based"))
    post = SocialPost(
        id="t1",
        author_username="k",
        text="大饼看到7万",
        created_at=datetime(2026, 7, 18),
    )
    a = az.analyze(post)
    # fallback may be descriptive; mapping via catalog still works for 大饼 if in map
    assert "BTC-USD" in (a.canonical_symbols or tm.find_in_text(post.text))


if __name__ == "__main__":
    test_daping_maps_to_btc()
    test_learn_alias_persists()
    test_rule_fallback_does_not_crash()
    print("all tests passed")
