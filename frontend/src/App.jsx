import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Check,
  CircleAlert,
  Database,
  ExternalLink,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  LineChart,
  LoaderCircle,
  RefreshCw,
  Search,
  Settings2,
  Sparkles,
  TimerReset,
  Wrench,
} from "lucide-react";
import { getHealth, getOverview, getTimelineKlines, getTimelineSnapshot, postJson } from "./api";
import KlineChart from "./components/KlineChart.jsx";
import {
  MODE_LABELS,
  ROUTES,
  classNames,
  dirLabel,
  dirToken,
  displaySymbol,
  fmt,
  isNonTradeSymbol,
  parseSymbolFromLocation,
  pathForTab,
  postUrl,
  quoteUsable,
  resolveRoute,
  sourceLabel,
  stateLabel,
  timeText,
} from "./utils.js";
import "./styles.css";

const DEFAULT_SYMBOLS = ["SNDK", "BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "TSLA"];
const INTERVAL_OPTIONS = [
  { value: "1m", label: "1分" },
  { value: "5m", label: "5分" },
  { value: "15m", label: "15分" },
  { value: "1h", label: "1时" },
  { value: "4h", label: "4时" },
  { value: "1d", label: "日" },
  { value: "1w", label: "周" },
];
const MARKER_FILTERS = [
  { key: "long", label: "看多", tone: "long" },
  { key: "short", label: "看空", tone: "short" },
  { key: "trade", label: "成交", tone: "trade" },
  { key: "note", label: "笔记", tone: "note" },
];

function markerFilterKey(event) {
  if (event.kind === "trade") return "trade";
  const direction = dirToken(event.direction || event.direction_hint);
  if (direction === "long") return "long";
  if (direction === "short") return "short";
  // notes / unknown / flat → gray "note" bucket
  return "note";
}

function locationState() {
  return { pathname: window.location.pathname, hash: window.location.hash, search: window.location.search };
}

function useBrowserLocation() {
  const [location, setLocation] = useState(locationState);
  useEffect(() => {
    const update = () => setLocation(locationState());
    window.addEventListener("popstate", update);
    window.addEventListener("hashchange", update);
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener("hashchange", update);
    };
  }, []);
  return [location, setLocation];
}

function IconButton({ icon: Icon, label, onClick, disabled = false, className = "" }) {
  return (
    <button className={classNames("icon-button", className)} onClick={onClick} disabled={disabled} title={label} aria-label={label}>
      <Icon size={16} strokeWidth={1.8} />
    </button>
  );
}

function StatusDot({ tone = "neutral" }) {
  return <span className={classNames("status-dot", tone)} aria-hidden="true" />;
}

function healthSummary(health) {
  if (!health) return { label: "连接中", tone: "neutral" };
  if (!health.feed_ready) return { label: "数据源异常", tone: "danger" };
  if (health.auto_poll === false || health.poller?.enabled === false) return { label: "自动拉取已关", tone: "warning" };
  if (health.poller?.last_error) return { label: "拉取异常", tone: "danger" };
  if (health.poller?.running) return { label: "同步中", tone: "success" };
  return { label: "在线", tone: "success" };
}

function Shell({ route, symbol, health, autoRefresh, setAutoRefresh, onNavigate, children }) {
  const status = healthSummary(health);
  useEffect(() => {
    document.title = route === "timeline" && symbol ? `IntentTrade · ${symbol}` : `IntentTrade · ${(ROUTES[route] || ROUTES.dash).title}`;
  }, [route, symbol]);
  const nav = [
    { id: "dash", label: "总览", icon: LayoutDashboard },
    { id: "timeline", label: "时间线", icon: LineChart },
    { id: "tools", label: "工具", icon: Wrench },
  ];
  return (
    <div className={classNames("app-shell", route === "timeline" && "timeline-shell")}>
      <header className="app-header">
        <a className="brand" href="/overview" onClick={(event) => { event.preventDefault(); onNavigate("dash"); }}>
          <span className="brand-mark"><Activity size={17} /></span>
          <span><strong>IntentTrade</strong><small>KOL 意图 · 模拟跟单</small></span>
        </a>
        <div className="header-status">
          <StatusDot tone={status.tone} />
          <span>{status.label}</span>
          {health?.kols?.length ? <span className="status-detail">关注 {health.kols.map((kol) => `@${kol}`).join(" ")}</span> : null}
        </div>
      </header>

      <nav className="primary-nav" aria-label="主导航">
        <div className="nav-links">
          {nav.map(({ id, label, icon: Icon }) => (
            <a
              key={id}
              className={classNames("nav-link", route === id || (id === "dash" && route === "overview") ? "active" : "")}
              href={pathForTab(id)}
              onClick={(event) => { event.preventDefault(); onNavigate(id); }}
            >
              <Icon size={15} />
              <span>{label}</span>
            </a>
          ))}
        </div>
        <label className="refresh-control">
          <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
          <span>自动刷新</span><em>60s</em>
        </label>
      </nav>
      <main className="page-content">{children}</main>
      <footer className="app-footer"><span>IntentTrade</span><span>研究工具 · 默认 paper execution</span></footer>
    </div>
  );
}

function SectionHeading({ eyebrow, title, action }) {
  return (
    <div className="section-heading">
      <div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
      {action}
    </div>
  );
}

function KpiStrip({ counts = {}, quoteValues = [] }) {
  const items = [
    { label: "原始推文", value: counts.posts, icon: Database },
    { label: "结构化信号", value: counts.signals, icon: Sparkles },
    { label: "等待入场", value: counts.waiting_signals, icon: TimerReset },
    { label: "可执行", value: counts.ready_signals, icon: Check, tone: "good" },
    { label: "模拟仓位", value: counts.open_trades, icon: BarChart3 },
    { label: "非实时行情", value: quoteValues.filter((quote) => !quote.is_live || quote.stale).length, icon: CircleAlert, tone: "warning" },
  ];
  return <div className="kpi-strip">{items.map(({ label, value, icon: Icon, tone }) => <div className={classNames("kpi-item", tone)} key={label}><Icon size={15} /><div><strong>{fmt(value)}</strong><span>{label}</span></div></div>)}</div>;
}

function EmptyState({ children = "暂无数据" }) {
  return <div className="empty-state"><span>{children}</span></div>;
}

function DataTable({ columns, children, empty = "暂无数据", className = "" }) {
  const hasRows = Array.isArray(children) ? children.length > 0 : Boolean(children);
  return <div className={classNames("table-frame", className)}><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{hasRows ? children : <tr><td colSpan={columns.length}><EmptyState>{empty}</EmptyState></td></tr>}</tbody></table></div>;
}

function SignalTable({ signals = [], navigate, previousIds }) {
  const ids = new Set((previousIds || []).map(String));
  return <DataTable className="signal-table" columns={["时间", "KOL", "标的", "方向", "方式", "入场 / 触发", "现价", "状态", "SL / TP", "置信"]} empty="暂无结构化信号">
    {(signals || []).map((signal) => {
      const nonTrade = isNonTradeSymbol(signal.symbol);
      const direction = dirToken(signal.direction);
      return <tr className={classNames(nonTrade && "muted-row", !ids.has(String(signal.id)) && previousIds?.length && "new-row")} key={signal.id || `${signal.post_id}-${signal.symbol}`}>
        <td data-label="时间">{timeText(signal.signal_time, "Asia/Shanghai", true)}</td>
        <td data-label="KOL" className="mono">@{signal.kol_username || "—"}</td>
        <td data-label="标的">{nonTrade ? <span className="muted">N/A<small>非交易推文</small></span> : <a href={pathForTab("timeline", { symbol: signal.symbol })} onClick={(event) => { event.preventDefault(); navigate("timeline", { symbol: signal.symbol }); }}>{displaySymbol(signal.symbol)}</a>}</td>
        <td data-label="方向"><span className={classNames("direction", direction)}>{nonTrade ? "非交易" : dirLabel(direction)}</span></td>
        <td data-label="方式">{nonTrade ? "—" : MODE_LABELS[signal.entry_mode] || signal.entry_mode || "—"}</td>
        <td data-label="入场 / 触发">{nonTrade ? "—" : <><strong>{fmt(signal.entry_price_low != null ? `${fmt(signal.entry_price_low)}-${fmt(signal.entry_price_high)}` : signal.entry_price)}</strong>{signal.trigger_price != null ? <small>触发 {fmt(signal.trigger_price)}</small> : null}</>}</td>
        <td data-label="现价">{nonTrade ? "—" : <><strong>{fmt(signal.current_price)}</strong><small>{signal.market_source || ""}</small></>}</td>
        <td data-label="状态"><span className={classNames("state", signal.state)}>{nonTrade ? "非交易推文" : stateLabel(signal.state)}</span><small>{(signal.decision_reason || signal.summary || "").slice(0, 90)}</small></td>
        <td data-label="SL / TP">{nonTrade ? "—" : `${fmt(signal.stop_loss)} / ${fmt(signal.take_profit)}`}</td>
        <td data-label="置信" className="mono">{fmt(signal.confidence)}</td>
      </tr>;
    })}
  </DataTable>;
}

function TradesTable({ trades = [] }) {
  return <DataTable columns={["KOL", "标的", "方向", "开仓", "平仓", "状态", "PnL %", "PnL $"]} empty="暂无模拟成交">
    {trades.map((trade) => <tr key={trade.id || `${trade.kol_username}-${trade.symbol}-${trade.entry_time}`}>
      <td data-label="KOL" className="mono">@{trade.kol_username || "—"}</td><td data-label="标的">{displaySymbol(trade.symbol)}</td><td data-label="方向"><span className={classNames("direction", dirToken(trade.direction))}>{dirLabel(trade.direction)}</span></td><td data-label="开仓">{fmt(trade.entry_price)}</td><td data-label="平仓">{fmt(trade.exit_price)}</td><td data-label="状态"><span className={classNames("trade-state", trade.status)}>{trade.status || "—"}</span></td><td data-label="PnL %" className="mono">{fmt(trade.pnl_pct)}</td><td data-label="PnL $" className="mono">{fmt(trade.pnl_usd)}</td>
    </tr>)}
  </DataTable>;
}

function QuotesTable({ quotes = {} }) {
  return <DataTable columns={["标的", "现价", "来源", "时间", "实时", "新鲜度", "执行"]} empty="暂无行情">
    {Object.entries(quotes).map(([symbol, quote]) => <tr key={symbol}><td data-label="标的" className="mono">{symbol}</td><td data-label="现价" className="mono">{fmt(quote.price)}</td><td data-label="来源">{sourceLabel(quote.source) || "—"}</td><td data-label="时间">{timeText(quote.ts, "Asia/Shanghai", true)}</td><td data-label="实时"><span className={classNames("live-state", quote.is_live ? "live" : "stale")}>{quote.is_live ? "是" : "否"}</span></td><td data-label="新鲜度" className="mono">{quote.age_seconds == null ? "—" : `${Math.round(quote.age_seconds)}s`}</td><td data-label="执行"><span className={classNames("state", quoteUsable(quote) ? "ready" : "waiting_market_data")}>{quoteUsable(quote) ? "可执行" : "观察"}</span></td></tr>)}
  </DataTable>;
}

function StatsTable({ stats = [], overall }) {
  const rows = [...stats, ...(overall ? [overall] : [])];
  return <DataTable columns={["KOL", "已平", "胜 / 负", "胜率", "PnL $", "摘要"]} empty="暂无胜率统计">
    {rows.map((row) => <tr key={row.kol}><td data-label="KOL" className="mono">@{row.kol}</td><td data-label="已平">{fmt(row.closed)}</td><td data-label="胜 / 负">{fmt(row.wins)} / {fmt(row.losses)}</td><td data-label="胜率" className="mono">{((row.win_rate || 0) * 100).toFixed(1)}%</td><td data-label="PnL $" className="mono">{fmt(row.total_pnl_usd)}</td><td data-label="摘要">{row.summary || "—"}</td></tr>)}
  </DataTable>;
}

function PostAnalysis({ post, signals = [], notes = [], trades = [], navigate }) {
  const postSignals = signals.filter((signal) => String(signal.post_id) === String(post?.id));
  const postNotes = notes.filter((note) => String(note.post_id) === String(post?.id));
  const signalIds = new Set(postSignals.map((signal) => String(signal.id)));
  const postTrades = trades.filter((trade) => signalIds.has(String(trade.signal_id)));
  const kind = postSignals.length ? "结构化信号" : postNotes.length ? "描述笔记" : "尚未解析";

  if (!post) return <div className="review-analysis"><EmptyState>请选择一条原帖</EmptyState></div>;
  return <aside className="review-analysis" aria-live="polite">
    <div className="analysis-pane-head">
      <div><span className="eyebrow">AI ANALYSIS</span><h3>{kind}</h3></div>
      <span className="analysis-post-time">{timeText(post.created_at, "Asia/Shanghai", true)}</span>
    </div>
    {postSignals.map((signal) => {
      const direction = dirToken(signal.direction);
      const nonTrade = isNonTradeSymbol(signal.symbol);
      return <div className="linked-analysis signal-analysis" key={signal.id}>
        <div className="analysis-status-row">
          <span className={classNames("direction", direction)}>{dirLabel(direction)}</span>
          <span className={classNames("state", signal.state)}>{stateLabel(signal.state)}</span>
          {!nonTrade ? <a href={pathForTab("timeline", { symbol: signal.symbol })} onClick={(event) => { event.preventDefault(); navigate("timeline", { symbol: signal.symbol }); }}>{displaySymbol(signal.symbol)}</a> : null}
        </div>
        <p className="analysis-summary">{signal.summary || "—"}</p>
        <dl className="analysis-levels">
          <div><dt>动作</dt><dd>{signal.action || "—"}</dd></div>
          <div><dt>入场</dt><dd>{signal.entry_price_low != null ? `${fmt(signal.entry_price_low)}-${fmt(signal.entry_price_high)}` : fmt(signal.entry_price)}</dd></div>
          <div><dt>止损</dt><dd>{fmt(signal.stop_loss)}</dd></div>
          <div><dt>止盈</dt><dd>{fmt(signal.take_profit)}</dd></div>
          <div><dt>置信</dt><dd>{fmt(signal.confidence)}</dd></div>
          <div><dt>方式</dt><dd>{MODE_LABELS[signal.entry_mode] || signal.entry_mode || "—"}</dd></div>
        </dl>
        {signal.reasoning ? <div className="analysis-reasoning"><span>判断依据</span><p>{signal.reasoning}</p></div> : null}
      </div>;
    })}
    {postNotes.map((note) => <div className="linked-analysis note-analysis" key={note.id}>
      <div className="analysis-status-row"><span>描述积累</span>{!isNonTradeSymbol(note.symbol) ? <a href={pathForTab("timeline", { symbol: note.symbol })} onClick={(event) => { event.preventDefault(); navigate("timeline", { symbol: note.symbol }); }}>{displaySymbol(note.symbol)}</a> : null}<span className={classNames("direction", dirToken(note.direction_hint))}>{dirLabel(note.direction_hint)}</span></div>
      <p className="analysis-summary">{note.content || "—"}</p>
    </div>)}
    {postTrades.length ? <div className="linked-analysis trade-analysis"><span className="analysis-block-label">关联模拟成交</span>{postTrades.map((trade) => <div className="linked-trade" key={trade.id}><strong>{fmt(trade.entry_price)}</strong><span className={classNames("trade-state", trade.status)}>{trade.status}</span><span>{fmt(trade.pnl_pct)}%</span></div>)}</div> : null}
    {!postSignals.length && !postNotes.length ? <EmptyState>该原帖暂无 AI 解析记录</EmptyState> : null}
  </aside>;
}

function PostReviewWorkspace({ posts = [], signals = [], notes = [], trades = [], navigate }) {
  const [activePostId, setActivePostId] = useState("");
  useEffect(() => {
    if (!posts.length) setActivePostId("");
    else if (!posts.some((post) => String(post.id) === String(activePostId))) setActivePostId(String(posts[0].id));
  }, [posts, activePostId]);
  const activePost = posts.find((post) => String(post.id) === String(activePostId)) || posts[0];
  const handleScroll = (event) => {
    const container = event.currentTarget;
    const items = [...container.querySelectorAll("[data-post-id]")];
    const threshold = container.scrollTop + 72;
    let current = items[0];
    for (const item of items) {
      if (item.offsetTop <= threshold) current = item;
      else break;
    }
    if (current?.dataset.postId) setActivePostId(current.dataset.postId);
  };

  return <section className="content-section review-section">
    <SectionHeading eyebrow="POST REVIEW" title="原帖与 AI 解析" action={<span className="section-count">{posts.length} 条 · 北京时间</span>} />
    <div className="review-grid dash-layout">
      <div className="posts-rail review-feed post-list" onScroll={handleScroll} aria-label="最近原帖">
        {posts.length ? posts.map((post) => {
          const url = postUrl(post);
          const active = String(post.id) === String(activePost?.id);
          return <article className={classNames("post-item", active && "active")} data-post-id={post.id} key={post.id} tabIndex={0} onClick={() => setActivePostId(String(post.id))} onFocus={() => setActivePostId(String(post.id))}>
            <div className="post-meta"><span>@{post.author_username || "—"}</span><time>{timeText(post.created_at, "Asia/Shanghai", true)}</time>{url ? <a href={url} target="_blank" rel="noreferrer" aria-label="打开原帖" onClick={(event) => event.stopPropagation()}><ExternalLink size={13} /></a> : null}</div>
            <p>{post.text || "—"}</p>
            {(post.media_urls || []).slice(0, 2).map((media) => <img key={media} src={media} alt="" loading="lazy" />)}
          </article>;
        }) : <EmptyState>暂无原帖</EmptyState>}
      </div>
      <PostAnalysis post={activePost} signals={signals} notes={notes} trades={trades} navigate={navigate} />
    </div>
  </section>;
}

function DashboardPage({ health, autoRefresh, onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [lastIds, setLastIds] = useState([]);
  const [refreshedAt, setRefreshedAt] = useState(null);
  const previousSignalIds = useRef([]);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setBusy(true);
    try {
      const next = await getOverview();
      setLastIds(previousSignalIds.current);
      setData(next);
      previousSignalIds.current = (next.signals || []).map((signal) => signal.id);
      setRefreshedAt(new Date());
    } catch (error) {
      // keep quiet on auto refresh failures; surface on manual via empty state
      if (!silent) {
        setData(null);
      }
    } finally {
      setLoading(false);
      setBusy(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!autoRefresh) return undefined;
    const timer = window.setInterval(() => refresh(true), 60000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, refresh]);

  const counts = data?.counts || {};
  const quotes = data?.quotes || {};
  return <>
    <div className="page-intro">
      <div>
        <span className="eyebrow">OPERATIONS / OVERVIEW</span>
        <h1>市场意图总览</h1>
        <p>把 KOL 的自然语言喊单，整理成可审阅的信号、行情和模拟执行状态。</p>
      </div>
      <div className="intro-meta">
        <span><StatusDot tone={healthSummary(health).tone} />{healthSummary(health).label}</span>
        <span>{refreshedAt ? `刷新于 ${timeText(refreshedAt.toISOString(), health?.timezone || "Asia/Shanghai", true)}` : "等待数据"}</span>
        <IconButton
          icon={RefreshCw}
          label="刷新总览"
          className={busy ? "spinning" : ""}
          disabled={busy}
          onClick={() => refresh()}
        />
      </div>
    </div>
    {loading && !data ? <div className="loading-line"><LoaderCircle className="spin" size={17} />加载看板数据…</div> : <>
      <KpiStrip counts={counts} quoteValues={Object.values(quotes)} />
      <div className="dashboard-main">
        <section className="content-section"><SectionHeading eyebrow="SIGNALS" title="结构化信号" action={data?.signals?.length ? <span className="section-count">{data.signals.length} 条</span> : null} /><SignalTable signals={data?.signals} navigate={onNavigate} previousIds={lastIds} /></section>
        <PostReviewWorkspace posts={data?.posts} signals={data?.signals} notes={data?.notes} trades={data?.trades} navigate={onNavigate} />
        <section className="content-section"><SectionHeading eyebrow="PAPER TRADES" title="模拟成交" /><TradesTable trades={data?.trades} /></section>
        <section className="content-section"><SectionHeading eyebrow="MARKET DATA" title="行情状态" /><QuotesTable quotes={quotes} /></section>
        <section className="content-section"><SectionHeading eyebrow="PERFORMANCE" title="KOL 胜率" /><StatsTable stats={data?.kol_stats} overall={data?.overall} /></section>
      </div>
    </>}
  </>;
}

function QuoteStrip({ symbol, quote, chartData }) {
  let price = quote?.price;
  let source = sourceLabel(quote?.source);
  const bars = chartData?.bars || [];
  const last = bars[bars.length - 1];
  if (last?.close != null && (price == null || chartData?.is_live || quote?.stale || !quote?.is_live)) {
    price = last.close;
    source = sourceLabel(chartData.source) || source;
  }
  return <div className="quote-strip"><strong>{symbol}</strong><span className="quote-price">{fmt(price)}</span><span>{source || "暂无来源"}</span>{quote?.ts && !quote.stale ? <time>{timeText(quote.ts, "Asia/Shanghai", true)}</time> : null}<span className={classNames("live-state", quoteUsable(quote) ? "live" : "stale")}>{quoteUsable(quote) ? "实时可执行" : "仅供观察"}</span></div>;
}

function TimelineEvent({ event }) {
  const token = dirToken(event.direction);
  const kind = event.kind === "signal" ? "SIGNAL" : event.kind === "trade" ? "TRADE" : "NOTE";
  return <article className={classNames("timeline-event", event.kind, token)}><div className="event-rail"><span className="event-dot" /></div><div className="event-content"><div className="event-meta"><span className="event-tag">{kind}</span><time>{timeText(event.time, "Asia/Shanghai", true)}</time>{event.kol ? <span className="mono">@{event.kol}</span> : null}</div>{event.kind === "signal" ? <><div className="event-title"><span className={classNames("direction", token)}>{dirLabel(token)}</span><span>{stateLabel(event.state)}</span><span>{MODE_LABELS[event.entry_mode] || event.entry_mode || "未明确"}</span></div><p>{event.decision_reason || event.summary || "—"}</p><div className="event-levels">入场 {fmt(event.entry_price)} · 触发 {fmt(event.trigger_price)} · 当前 {fmt(event.current_price)} · SL {fmt(event.stop_loss)} · TP {fmt(event.take_profit)} · 置信 {fmt(event.confidence)}</div></> : event.kind === "trade" ? <><div className="event-title"><span className={classNames("direction", token)}>{dirLabel(token)}</span><span>{fmt(event.entry_price)} → {fmt(event.exit_price)}</span><span className={classNames("trade-state", event.status)}>{event.status}</span></div><p>PnL {fmt(event.pnl_pct)}% / ${fmt(event.pnl_usd)}</p></> : <p>{event.content || "—"}</p>}</div></article>;
}

function TimelinePage({ initialSymbol, onNavigate }) {
  const [symbol, setSymbol] = useState(initialSymbol || "SNDK");
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [input, setInput] = useState(initialSymbol || "SNDK");
  const [interval, setInterval] = useState("15m");
  const [eventFilters, setEventFilters] = useState({ long: true, short: true, trade: true, note: true });
  const [snapshot, setSnapshot] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);

  useEffect(() => {
    getOverview().then((overview) => {
      const values = new Set(DEFAULT_SYMBOLS);
      Object.keys(overview.quotes || {}).forEach((value) => values.add(value));
      (overview.signals || []).forEach((item) => { if (!isNonTradeSymbol(item.symbol)) values.add(item.symbol); });
      setSymbols([...values].sort());
    }).catch(() => {});
  }, []);

  const load = useCallback(async (nextSymbol, nextInterval) => {
    const clean = String(nextSymbol || "").trim();
    if (!clean || isNonTradeSymbol(clean)) return;
    const currentRequest = ++requestId.current;
    setLoading(true); setError(""); setSymbol(clean); setInput(clean);
    try {
      const [nextSnapshot, nextChart] = await Promise.all([getTimelineSnapshot(clean), getTimelineKlines(clean, nextInterval)]);
      if (currentRequest !== requestId.current) return;
      setSnapshot(nextSnapshot); setChartData(nextChart);
      setSymbols((current) => [...new Set([...current, nextSnapshot.symbol || clean])].sort());
    } catch (loadError) {
      if (currentRequest === requestId.current) { setError(loadError.message); setSnapshot(null); setChartData(null); }
    } finally {
      if (currentRequest === requestId.current) setLoading(false);
    }
  }, []);

  useEffect(() => { load(initialSymbol || "SNDK", interval); }, [initialSymbol, load]);

  const events = useMemo(() => {
    if (!snapshot) return [];
    const signalEvents = (snapshot.structured_signals || []).map((signal) => ({ ...signal, kind: "signal", time: signal.signal_time || signal.created_at }));
    const noteEvents = (snapshot.notes || []).map((note) => ({ ...note, kind: "note", time: note.note_time || note.created_at }));
    const tradeEvents = (snapshot.trades || []).map((trade) => ({ ...trade, kind: "trade", time: trade.entry_time || trade.created_at }));
    return [...signalEvents, ...noteEvents, ...tradeEvents].sort((a, b) => String(b.time).localeCompare(String(a.time)));
  }, [snapshot]);

  const visibleEvents = useMemo(
    () => events.filter((event) => eventFilters[markerFilterKey(event)] !== false),
    [events, eventFilters],
  );

  const visibleChartData = useMemo(() => {
    if (!chartData) return null;
    const visible = (event) => eventFilters[markerFilterKey(event)] !== false;
    return {
      ...chartData,
      events: (chartData.events || []).filter(visible),
      markers: (chartData.markers || []).filter(visible),
    };
  }, [chartData, eventFilters]);

  const submit = (event) => {
    event.preventDefault();
    const clean = input.trim();
    if (!clean) return;
    if (clean === symbol) load(clean, interval);
    else onNavigate("timeline", { symbol: clean, replace: true });
  };
  const resolved = snapshot?.symbol || symbol;
  return <>
    <div className="page-intro"><div><span className="eyebrow">MARKET / TIMELINE</span><h1>标的时间线</h1><p>在 K 线、行情状态和 KOL 事件之间切换，回看信号发生的上下文。</p></div><div className="symbol-badge"><LineChart size={17} /><strong>{resolved}</strong><span>{interval}</span></div></div>
    <section className="timeline-toolbar symbol-toolbar"><form className="symbol-form" onSubmit={submit} aria-label="标的筛选"><span className="symbol-label">标的</span><select value={symbols.includes(symbol) ? symbol : ""} onChange={(event) => { const next = event.target.value; if (next) { setInput(next); if (next === symbol) load(next, interval); else onNavigate("timeline", { symbol: next, replace: true }); } }} aria-label="选择标的"><option value="">快速选择</option>{symbols.map((item) => <option value={item} key={item}>{item}</option>)}</select><label className="search-field"><Search size={15} /><input value={input} onChange={(event) => setInput(event.target.value)} placeholder="代码 / 别名" /></label><button className="button timeline-load" type="submit">加载</button></form></section>
    <QuoteStrip symbol={resolved} quote={snapshot?.quote || chartData?.quote} chartData={chartData} />
    <section className="chart-controls">
      <div className="interval-control">
        <div className="intervals" role="group" aria-label="K线周期">
          {INTERVAL_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              className={classNames("interval-button", interval === value && "active")}
              onClick={() => { setInterval(value); load(symbol, value); }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="event-filter-control" role="group" aria-label="事件筛选">
        {MARKER_FILTERS.map(({ key, label, tone }) => (
          <button
            key={key}
            type="button"
            className={classNames("event-filter-button", eventFilters[key] ? "on" : "off")}
            aria-pressed={!!eventFilters[key]}
            title={label}
            onClick={() => setEventFilters((current) => ({ ...current, [key]: !current[key] }))}
          >
            <span className={classNames("ev-dot", tone)} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </div>
      <span className="chart-meta">
        {loading
          ? <><LoaderCircle size={14} className="spin" />加载中</>
          : `${resolved} · ${interval} · ${chartData?.count || 0} 根 · ${visibleEvents.length} 事件`}
      </span>
    </section>
    {error ? <div className="error-banner"><CircleAlert size={16} />{error}</div> : null}
    <section className="chart-section"><div className="chart-header"><div><span className="eyebrow">PRICE ACTION</span><h2>{resolved} / {interval}</h2></div></div><KlineChart data={visibleChartData} interval={interval} /></section>
    <section className="timeline-section"><SectionHeading eyebrow="EVENT STREAM" title="事件流" action={<span className="section-count">{visibleEvents.length} 条</span>} />{visibleEvents.length ? <div className="timeline-list">{visibleEvents.map((event, index) => <TimelineEvent event={event} key={`${event.kind}-${event.id || event.time}-${index}`} />)}</div> : <EmptyState>当前筛选下暂无事件</EmptyState>}</section>
  </>;
}

function ToolsPage() {
  const [text, setText] = useState("");
  const [kol, setKol] = useState("dryrun");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const submit = async (event) => {
    event.preventDefault();
    if (!text.trim()) return;
    setLoading(true); setResult(null);
    try { setResult(await postJson("/api/analyze-text", { text: text.trim(), kol: kol.trim() || "dryrun" })); } catch (error) { setResult({ error: error.message }); } finally { setLoading(false); }
  };
  return <>
    <div className="page-intro"><div><span className="eyebrow">TOOLS / ANALYSIS</span><h1>意图分析实验台</h1><p>输入一段自然语言喊单，查看模型解析结果和当前行情下的执行判断，不写入数据库。</p></div><div className="tool-mark"><FlaskConical size={18} /><span>DRY RUN</span></div></div>
    <section className="tool-section"><SectionHeading eyebrow="TEXT INPUT" title="AI 干跑" action={<span className="muted">不会写入数据库</span>} /><form className="tool-form" onSubmit={submit}><div className="tool-input-row"><label><span>KOL</span><input value={kol} onChange={(event) => setKol(event.target.value)} /></label><input className="tool-text" value={text} onChange={(event) => setText(event.target.value)} placeholder="例：1345 闪迪上车，止损 1200，目标 1600" /></div><button className="button primary" disabled={loading || !text.trim()}><Sparkles size={15} />{loading ? "分析中…" : "开始分析"}</button></form></section>
    {result ? <section className="result-section"><SectionHeading eyebrow="ANALYSIS RESULT" title="解析结果" action={result.error ? <span className="state rejected">失败</span> : <span className="state ready"><Check size={13} />已完成</span>} />{result.error ? <div className="error-banner"><CircleAlert size={16} />{result.error}</div> : <AnalysisResult result={result} />}</section> : <div className="tool-empty"><Settings2 size={19} /><span>等待一段文本</span><small>结果会显示信号类型、方向、价格级别和执行判断。</small></div>}
  </>;
}

function AnalysisResult({ result }) {
  const fields = [["分析器", result.analyzer], ["信号类型", result.signal_type], ["方向", dirLabel(result.direction)], ["标的", (result.canonical_symbols || []).join(", ") || "N/A"], ["动作", result.action], ["入场方式", MODE_LABELS[result.entry_mode] || result.entry_mode], ["入场", result.entry_price], ["止损", result.stop_loss], ["止盈", result.take_profit], ["置信度", result.confidence]];
  return <div className="analysis-result"><div className="result-grid">{fields.map(([label, value]) => <div key={label}><span>{label}</span><strong>{fmt(value)}</strong></div>)}</div><div className="result-copy"><div><span>摘要</span><p>{result.summary || "—"}</p></div><div><span>推理</span><p>{result.reasoning || result.analysis_text || "—"}</p></div></div>{result.decision ? <div className="decision-row"><Gauge size={17} /><div><span>执行判断</span><strong>{stateLabel(result.decision.state)}</strong><p>{result.decision.reason || "—"}</p></div></div> : null}</div>;
}

export default function App() {
  const [location, setLocation] = useBrowserLocation();
  const [health, setHealth] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const route = resolveRoute(location.pathname, location.hash);
  const symbol = parseSymbolFromLocation(location.pathname, location.hash, location.search) || "SNDK";

  const refreshHealth = useCallback(async () => {
    try { setHealth(await getHealth()); } catch (_) { setHealth((current) => current || { feed_ready: false }); }
  }, []);
  useEffect(() => { refreshHealth(); }, [refreshHealth]);
  useEffect(() => {
    if (!autoRefresh) return undefined;
    const timer = window.setInterval(refreshHealth, 60000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, refreshHealth]);

  const navigate = useCallback((tab, options = {}) => {
    const target = pathForTab(tab, options);
    if (options.replace) window.history.replaceState({}, "", target); else window.history.pushState({}, "", target);
    setLocation(locationState());
  }, [setLocation]);

  return <Shell route={route} symbol={symbol} health={health} autoRefresh={autoRefresh} setAutoRefresh={setAutoRefresh} onNavigate={navigate}>
    {route === "timeline" ? <TimelinePage initialSymbol={symbol} onNavigate={navigate} /> : route === "tools" ? <ToolsPage /> : <DashboardPage health={health} autoRefresh={autoRefresh} onNavigate={navigate} />}
  </Shell>;
}
