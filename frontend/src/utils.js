export const STATE_LABELS = {
  waiting_market_data: "等待行情",
  waiting_entry: "等待入场",
  ready: "可执行",
  waiting_risk_limit: "等待风险额度",
  observed_position: "已入场观察",
  exit_intent: "退出意图",
  executed: "已模拟成交",
  superseded: "已被后续更新取代",
  expired: "已过期",
  rejected: "已拒绝",
};

export const MODE_LABELS = {
  market: "市价",
  limit: "限价",
  stop: "突破/跌破",
  range: "区间",
  unknown: "未明确",
};

export const ROUTES = {
  dash: { path: "/overview", title: "总览" },
  timeline: { path: "/timeline", title: "时间线" },
  tools: { path: "/tools", title: "工具" },
};

export function resolveRoute(pathname, hash = "") {
  const path = String(pathname || "/").replace(/\/+$/, "") || "/";
  const hashPath = String(hash || "")
    .replace(/^#/, "")
    .replace(/\/+$/, "") || "/";
  const normalizedHash = hashPath.startsWith("/") ? hashPath : `/${hashPath}`;
  if (path === "/timeline" || path.startsWith("/timeline/") || normalizedHash === "/timeline" || normalizedHash.startsWith("/timeline/")) return "timeline";
  if (path === "/tools" || path.startsWith("/tools/") || normalizedHash === "/tools" || normalizedHash.startsWith("/tools/")) return "tools";
  if (path.startsWith("/symbol/") || normalizedHash.startsWith("/symbol/")) return "timeline";
  return "dash";
}

export function parseSymbolFromLocation(pathname, hash = "", search = "") {
  const path = String(pathname || "");
  const pathMatch = path.match(/^\/(?:timeline|symbol)\/([^/?#]+)/i);
  if (pathMatch) return decodeURIComponent(pathMatch[1]);
  const hashMatch = String(hash || "").replace(/^#/, "").match(/^\/?(?:timeline|symbol)\/([^/?#]+)/i);
  if (hashMatch) return decodeURIComponent(hashMatch[1]);
  try {
    return new URLSearchParams(String(search || "").replace(/^\?/, "")).get("symbol");
  } catch (_) {
    return null;
  }
}

export function pathForTab(tab, options = {}) {
  if (tab === "timeline" && options.symbol) return `/timeline/${encodeURIComponent(options.symbol)}`;
  return (ROUTES[tab] || ROUTES.dash).path;
}

export function isNonTradeSymbol(symbol) {
  const value = String(symbol || "").trim().toUpperCase();
  return !value || ["UNKNOWN", "N/A", "NONE", "NULL"].includes(value);
}

export function displaySymbol(symbol) {
  return isNonTradeSymbol(symbol) ? "N/A" : String(symbol || "—");
}

export function fmt(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/\.?0+$/, "");
  return String(value);
}

export function dirToken(value) {
  const token = String(value ?? "").replace(/^Direction\./i, "").toLowerCase();
  return ["long", "short", "flat", "unknown"].includes(token) ? token : "unknown";
}

export function dirLabel(value) {
  const token = dirToken(value);
  return { long: "看多", short: "看空", flat: "观望", unknown: "未知" }[token];
}

export function stateLabel(value) {
  return STATE_LABELS[value] || value || "未知";
}

export function sourceLabel(source) {
  const value = String(source || "");
  if (!value || value === "unknown" || value === "unavailable") return "";
  if (value.startsWith("binance_bstock")) return "Binance bStock";
  if (value.startsWith("binance")) return "Binance";
  if (value.startsWith("okx")) return "OKX";
  if (value.startsWith("yfinance")) return "行情";
  if (value === "sample_fallback") return "示例";
  if (value.includes("disk_cache")) return "缓存";
  return value.replace(/_/g, " ");
}

export function timeText(value, timezone = "Asia/Shanghai", short = false) {
  if (!value) return "—";
  const raw = String(value);
  const normalized = /(?:Z|[+-]\d\d:\d\d)$/.test(raw) ? raw : `${raw.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return raw.replace("T", " ").slice(0, 19);
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone: timezone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: short ? undefined : "2-digit",
  }).formatToParts(date).reduce((out, part) => ({ ...out, [part.type]: part.value }), {});
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}${short ? "" : `:${parts.second}`}`;
}

export function postUrl(post) {
  const url = String(post?.url || "").trim();
  return /^https?:\/\//i.test(url) ? url : "";
}

export function quoteUsable(quote) {
  return Boolean(quote?.is_live && !quote?.stale && quote?.price != null);
}

export function classNames(...values) {
  return values.filter(Boolean).join(" ");
}
