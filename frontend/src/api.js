async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail || body.error || "";
    } catch (_) {
      detail = await response.text();
    }
    throw new Error(`${response.status}${detail ? ` · ${detail}` : ""}`);
  }
  return response.json();
}

const TIMELINE_CACHE_TTL_MS = 30_000;
const timelineCache = new Map();
const timelineInflight = new Map();

function timelineCacheKey(value) {
  return String(value || "").trim().toUpperCase();
}

function cachedTimelineRequest(key, loader) {
  const now = Date.now();
  const cached = timelineCache.get(key);
  if (cached && cached.expiresAt > now) return Promise.resolve(cached.value);

  const pending = timelineInflight.get(key);
  if (pending) return pending;

  const requestPromise = Promise.resolve()
    .then(loader)
    .then((value) => {
      timelineCache.set(key, { value, expiresAt: Date.now() + TIMELINE_CACHE_TTL_MS });
      return value;
    })
    .finally(() => timelineInflight.delete(key));
  timelineInflight.set(key, requestPromise);
  return requestPromise;
}

export const getJson = (url, options) => request(url, options);

export const postJson = (url, body) =>
  request(url, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const getHealth = () => getJson("/api/health");
export const getOverview = () => getJson("/api/overview");
export const getSymbol = (symbol) =>
  getJson(`/api/symbol/${encodeURIComponent(symbol)}`);
export const getKlines = (symbol, interval, { includeQuote = true } = {}) => {
  const limit = interval === "1m" ? 500 : interval === "5m" || interval === "15m" ? 400 : 300;
  return getJson(
    `/api/klines/${encodeURIComponent(symbol)}?interval=${encodeURIComponent(interval)}&limit=${limit}&markers=true&include_quote=${includeQuote}`,
  );
};

export const getTimelineSnapshot = (symbol) => {
  const key = `symbol:${timelineCacheKey(symbol)}`;
  return cachedTimelineRequest(key, () => getSymbol(symbol));
};

export const getTimelineKlines = (symbol, interval) => {
  const key = `klines:${timelineCacheKey(symbol)}:${String(interval || "").trim().toLowerCase()}`;
  return cachedTimelineRequest(key, () => getKlines(symbol, interval, { includeQuote: false }));
};
