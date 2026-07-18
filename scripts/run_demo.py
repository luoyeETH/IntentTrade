#!/usr/bin/env python3
"""One-shot demo: clean DB optional, run pipeline, print report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_trade.config import load_settings
from intent_trade.pipeline.runner import Pipeline


def main() -> None:
    settings = load_settings()
    # ensure fresh demo
    if settings.db_path.exists():
        settings.db_path.unlink()
    pipe = Pipeline(settings)
    summary = pipe.run(settle=True)
    pipe.print_report(summary)
    print("\n--- JSON summary ---")
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
