/* Compatibility helpers for existing route smoke tests. The runtime is React. */
const ROUTES = {
  dash: { path: "/overview" },
  timeline: { path: "/timeline" },
  tools: { path: "/tools" },
};

function resolveRoute(pathname, hash) {
  const path = String(pathname || "/").replace(/\/+$/, "") || "/";
  const h = String(hash || "").replace(/^#/, "").replace(/\/+$/, "") || "/";
  const hashPath = h.startsWith("/") ? h : `/${h}`;
  if (path === "/timeline" || path.startsWith("/timeline/") || hashPath === "/timeline" || hashPath.startsWith("/timeline/")) return "timeline";
  if (path === "/tools" || path.startsWith("/tools/") || hashPath === "/tools" || hashPath.startsWith("/tools/")) return "tools";
  if (path.startsWith("/symbol/") || hashPath.startsWith("/symbol/")) return "timeline";
  return "dash";
}

function parseSymbolFromLocation(pathname, hash, search) {
  const pathMatch = String(pathname || "").match(/^\/(?:timeline|symbol)\/([^/?#]+)/i);
  if (pathMatch) return decodeURIComponent(pathMatch[1]);
  const hashMatch = String(hash || "").replace(/^#/, "").match(/^\/?(?:timeline|symbol)\/([^/?#]+)/i);
  if (hashMatch) return decodeURIComponent(hashMatch[1]);
  return new URLSearchParams(String(search || "").replace(/^\?/, "")).get("symbol");
}

function pathForTab(tab, opts = {}) {
  if (tab === "timeline" && opts.symbol) return `/timeline/${encodeURIComponent(opts.symbol)}`;
  return (ROUTES[tab] || ROUTES.dash).path;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ROUTES, resolveRoute, parseSymbolFromLocation, pathForTab };
}
