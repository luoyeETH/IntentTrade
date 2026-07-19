"""Frontend shell theme + client routing checks against React SPA assets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from intent_trade.web import app as web_app

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "intent_trade" / "web" / "static"
FRONTEND_CSS = ROOT / "frontend" / "src" / "styles.css"
# Compatibility helpers kept for Node route smoke tests
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"


def _latest_built_css() -> Path:
    assets = STATIC / "assets"
    candidates = sorted(assets.glob("*.css"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert candidates, f"no built css under {assets}"
    return candidates[0]


def _latest_built_js() -> Path:
    assets = STATIC / "assets"
    candidates = sorted(assets.glob("index-*.js"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert candidates, f"no built js under {assets}"
    return candidates[0]


def test_css_near_black_grid_and_no_blue_chrome():
    # Prefer source CSS (stable path); also accept built hashed css content
    css = FRONTEND_CSS.read_text(encoding="utf-8")
    assert "--bg:" in css
    assert re.search(r"--bg:\s*#000(?:000)?\b", css) or re.search(
        r"--bg:\s*#0[0-9a-fA-F]{5}", css
    )
    assert "linear-gradient" in css
    assert "var(--grid-color)" in css or "--grid-color" in css
    assert "background-size" in css
    assert re.search(r"--grid-size:\s*96px", css)
    assert "scrollbar-width" in css
    assert "::-webkit-scrollbar" in css
    assert "#3d6dff" not in css
    assert "#5b8cff" not in css
    assert "linear-gradient(135deg, #3d6dff" not in css
    assert not re.search(r"--bg:\s*#0[0-9a-fA-F]{2}[2-9a-fA-F][0-9a-fA-F]{2}", css)
    assert "--shadow: none" in css or re.search(r"--shadow:\s*none", css)


def test_post_review_workspace_has_bounded_linked_layout():
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = FRONTEND_CSS.read_text(encoding="utf-8")
    app_source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "dash-layout" in html or "id=\"root\"" in html
    assert "PostReviewWorkspace" in app_source
    assert "PostAnalysis" in app_source
    assert "data-post-id" in app_source
    assert "AI ANALYSIS" in app_source
    assert "NotesTable" not in app_source
    assert ".review-grid" in css
    assert ".posts-rail.review-feed" in css
    assert re.search(r"\.posts-rail\.review-feed[^}]+max-height", css)
    assert "grid-template-columns" in css


def test_dashboard_reads_background_results_without_manual_refresh():
    app_source = (ROOT / "frontend" / "src" / "App.jsx").read_text(
        encoding="utf-8"
    )
    assert "const LIVE_REFRESH_MS = 15000" in app_source
    assert "window.setInterval(() => refresh(true), LIVE_REFRESH_MS)" in app_source
    assert "window.setInterval(refreshHealth, LIVE_REFRESH_MS)" in app_source
    assert "health.poller?.analysis_running" in app_source


def test_resolve_route_from_shipped_app_js():
    """Drive resolveRoute exported from static/app.js compatibility helpers."""
    assert APP_JS.exists(), "static/app.js route helpers missing"
    script = r"""
const path = require('path');
const mod = require(path.resolve(process.argv[1]));
const assert = (c, m) => { if (!c) { console.error('FAIL', m); process.exit(1); } };
assert(mod.resolveRoute('/') === 'dash', 'root -> dash');
assert(mod.resolveRoute('/overview') === 'dash', 'overview -> dash');
assert(mod.resolveRoute('/timeline') === 'timeline', 'path timeline');
assert(mod.resolveRoute('/timeline/SNDK') === 'timeline', 'path timeline symbol');
assert(mod.resolveRoute('/tools') === 'tools', 'path tools');
assert(mod.resolveRoute('/', '#/timeline') === 'timeline', 'hash timeline');
assert(mod.resolveRoute('/', '#/tools') === 'tools', 'hash tools');
assert(mod.resolveRoute('/symbol/SNDK') === 'timeline', 'symbol path');
assert(mod.pathForTab('dash') === '/overview', 'pathForTab overview');
assert(mod.pathForTab('timeline', {symbol: 'BTC-USD'}) === '/timeline/BTC-USD', 'pathForTab symbol');
assert(mod.parseSymbolFromLocation('/symbol/ETH-USD') === 'ETH-USD', 'parse symbol legacy');
assert(mod.parseSymbolFromLocation('/timeline/SOL-USD') === 'SOL-USD', 'parse timeline path');
assert(mod.parseSymbolFromLocation('/timeline', '', '?symbol=SOL-USD') === 'SOL-USD', 'query symbol');
console.log('resolveRoute ok', JSON.stringify(mod.ROUTES));
"""
    r = subprocess.run(
        ["node", "-e", script, str(APP_JS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "resolveRoute ok" in r.stdout


def test_spa_routes_serve_shell_and_static(monkeypatch, tmp_path):
    monkeypatch.setenv("INTENT_TRADE_DB_PATH", str(tmp_path / "web-routes.db"))
    monkeypatch.setattr(web_app.bg_poller, "start", lambda: None)
    monkeypatch.setattr(web_app.bg_poller, "stop", lambda: None)
    monkeypatch.setattr(web_app, "_PIPELINE", None)
    responses = [
        web_app.index(),
        web_app.spa_overview(),
        web_app.spa_timeline(),
        web_app.spa_timeline("SNDK"),
        web_app.spa_tools(),
        web_app.spa_symbol("SNDK"),
    ]
    assert all(Path(response.path) == INDEX_HTML for response in responses)

    registered_paths = {route.path for route in web_app.app.routes}
    assert {"/", "/overview", "/timeline", "/tools", "/symbol/{symbol}"}.issubset(
        registered_paths
    )

    body = INDEX_HTML.read_text(encoding="utf-8")
    assert "IntentTrade" in body
    assert 'id="root"' in body
    assert "/static/assets/" in body
    assert re.search(r"/static/assets/index-[^\"']+\.js", body)
    assert re.search(r"/static/assets/index-[^\"']+\.css", body)
    assert 'href="/overview"' in body
    assert 'href="/timeline"' in body
    assert 'href="/tools"' in body

    built_css = _latest_built_css().read_text(encoding="utf-8")
    assert "--bg" in built_css
    assert "--grid-size" in built_css

    built_js = _latest_built_js().read_text(encoding="utf-8")
    assert "/overview" in built_js

    assert web_app.health().get("ok") is True
