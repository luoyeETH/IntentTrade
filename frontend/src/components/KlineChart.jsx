import { useEffect, useRef, useState } from "react";
import { ExternalLink } from "lucide-react";
import { classNames, dirLabel, dirToken, fmt, stateLabel, timeText } from "../utils.js";

function dayValue(unixSeconds) {
  const date = new Date(Number(unixSeconds) * 1000);
  return [date.getUTCFullYear(), String(date.getUTCMonth() + 1).padStart(2, "0"), String(date.getUTCDate()).padStart(2, "0")].join("-");
}

function nearestTime(value, times) {
  if (!times.length) return value;
  let nearest = times[0];
  let distance = Math.abs(Number(value) - nearest);
  for (const candidate of times.slice(1)) {
    const nextDistance = Math.abs(Number(value) - candidate);
    if (nextDistance < distance) {
      nearest = candidate;
      distance = nextDistance;
    }
  }
  return nearest;
}

function chartTimeValue(value, interval) {
  return interval === "1d" || interval === "1w" ? dayValue(value) : Number(value);
}

function timeKey(value) {
  if (value && typeof value === "object") {
    return [value.year, String(value.month).padStart(2, "0"), String(value.day).padStart(2, "0")].join("-");
  }
  return String(value);
}

function eventDate(value) {
  return typeof value === "number" ? new Date(value * 1000).toISOString() : value;
}

function tooltipKind(event) {
  if (event.kind === "note") return "笔记";
  if (event.kind === "trade") return "成交";
  const direction = dirToken(event.direction);
  return direction === "long" ? "看多" : direction === "short" ? "看空" : "观察";
}

function EventTooltip({ hover }) {
  const event = hover.events[0] || {};
  const direction = dirToken(event.direction);
  const text = event.post_text || event.source_text || event.summary || "（无事件详情）";
  const levels = [
    event.entry_price != null ? `入场 ${fmt(event.entry_price)}` : null,
    event.stop_loss != null ? `SL ${fmt(event.stop_loss)}` : null,
    event.take_profit != null ? `TP ${fmt(event.take_profit)}` : null,
  ].filter(Boolean);
  const url = /^https?:\/\//i.test(String(event.post_url || "")) ? event.post_url : "";
  return <div className="chart-event-tooltip" style={{ left: hover.left, top: hover.top }}>
    <div className="tooltip-head"><span className={classNames("direction", direction)}>{dirLabel(direction)}</span><span>{tooltipKind(event)}</span>{event.kol ? <span className="mono">@{event.kol}</span> : null}<time>{timeText(eventDate(event.time), "Asia/Shanghai", true)}</time></div>
    {event.state ? <div className="tooltip-state"><span className={classNames("state", event.state)}>{stateLabel(event.state)}</span></div> : null}
    <p>{String(text).slice(0, 420)}</p>
    {levels.length ? <div className="tooltip-levels">{levels.join(" · ")}</div> : null}
    {hover.events.length > 1 ? <div className="tooltip-more">同一根 K 线另有 {hover.events.length - 1} 条事件</div> : null}
    {url ? <a className="tooltip-link" href={url} target="_blank" rel="noreferrer">查看原帖 <ExternalLink size={12} /></a> : null}
  </div>;
}

export default function KlineChart({ data, interval }) {
  const hostRef = useRef(null);
  const chartRef = useRef(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const host = hostRef.current;
    const chartHost = chartRef.current;
    if (!host || !chartHost) return undefined;
    chartHost.replaceChildren();
    setHover(null);
    if (!data) return undefined;

    const library = window.LightweightCharts;
    if (!library) {
      const empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "图表库未加载，请刷新页面";
      chartHost.appendChild(empty);
      return undefined;
    }

    const bars = data.bars || [];
    if (!bars.length) {
      const empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "暂无 K 线数据";
      chartHost.appendChild(empty);
      return undefined;
    }

    const size = () => ({
      width: Math.max(chartHost.clientWidth || 320, 320),
      height: Math.max(chartHost.clientHeight || 360, 300),
    });
    const chart = library.createChart(chartHost, {
      layout: {
        background: { type: "solid", color: "#0a0d0f" },
        textColor: "#89939a",
        fontSize: 11,
        fontFamily: "Inter, Segoe UI, PingFang SC, sans-serif",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.045)" },
        horzLines: { color: "rgba(255,255,255,0.045)" },
      },
      crosshair: {
        vertLine: { color: "rgba(255,255,255,0.26)", labelBackgroundColor: "#20262a" },
        horzLine: { color: "rgba(255,255,255,0.26)", labelBackgroundColor: "#20262a" },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.09)",
        scaleMargins: { top: 0.08, bottom: 0.22 },
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.09)",
        timeVisible: true,
        secondsVisible: false,
      },
      ...size(),
    });

    const candles = chart.addCandlestickSeries({
      upColor: "#54d39a",
      downColor: "#f47b86",
      borderUpColor: "#54d39a",
      borderDownColor: "#f47b86",
      wickUpColor: "#54d39a",
      wickDownColor: "#f47b86",
    });
    const volume = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    const candleUnix = bars.map((bar) => Number(bar.time));
    const seen = new Set();
    const candleData = bars
      .map((bar) => ({
        time: chartTimeValue(bar.time, interval),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }))
      .filter((bar) => {
        const key = String(bar.time);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    candles.setData(candleData);
    volume.setData(bars.map((bar) => ({
      time: chartTimeValue(bar.time, interval),
      value: bar.volume || 0,
      color: bar.close >= bar.open ? "rgba(84,211,154,0.32)" : "rgba(244,123,134,0.32)",
    })).filter((bar, index, all) => all.findIndex((item) => String(item.time) === String(bar.time)) === index));

    const markerEvents = [...(data.events || []), ...(data.markers || []).filter((marker) => marker.kind === "trade")];
    const eventByTime = new Map();
    for (const event of markerEvents) {
      const rawTime = Number(event.time);
      if (!Number.isFinite(rawTime)) continue;
      const key = timeKey(chartTimeValue(nearestTime(rawTime, candleUnix), interval));
      if (!eventByTime.has(key)) eventByTime.set(key, []);
      eventByTime.get(key).push(event);
    }

    if (typeof candles.setMarkers === "function") {
      const markerSeen = new Set();
      const markers = (data.markers || [])
        .map((marker) => {
          const snapped = nearestTime(Number(marker.time), candleUnix);
          const time = chartTimeValue(snapped, interval);
          const key = `${time}:${marker.shape}:${marker.text || ""}`;
          if (markerSeen.has(key)) return null;
          markerSeen.add(key);
          return { time, position: marker.position || "belowBar", shape: marker.shape || "circle", color: marker.color || "#a8b1b5", text: "" };
        })
        .filter(Boolean)
        .sort((a, b) => String(a.time).localeCompare(String(b.time)));
      candles.setMarkers(markers);
    }

    for (const line of data.price_lines || []) {
      const price = Number(line.price);
      if (!Number.isFinite(price)) continue;
      candles.createPriceLine({ price, color: line.color || "#b8c0c2", lineWidth: line.lineWidth || 1, lineStyle: line.lineStyle ?? 2, axisLabelVisible: true, title: line.title || "" });
    }

    const handleCrosshair = (param) => {
      const key = param?.time == null ? "" : timeKey(param.time);
      const matching = eventByTime.get(key);
      if (!matching?.length) {
        setHover(null);
        return;
      }
      const point = param.point || { x: chartHost.clientWidth * 0.5, y: chartHost.clientHeight * 0.5 };
      const tooltipWidth = Math.min(330, Math.max(210, chartHost.clientWidth * 0.42));
      const tooltipHeight = 156;
      const left = Math.max(10, Math.min(point.x + 18, chartHost.clientWidth - tooltipWidth - 10));
      const top = Math.max(10, Math.min(point.y - tooltipHeight - 18, chartHost.clientHeight - tooltipHeight - 10));
      setHover({ events: matching, left, top, anchor: point });
    };
    chart.subscribeCrosshairMove(handleCrosshair);
    chart.timeScale().fitContent();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => chart.applyOptions(size()));
    observer?.observe(chartHost);
    return () => {
      observer?.disconnect();
      chart.unsubscribeCrosshairMove(handleCrosshair);
      chart.remove();
    };
  }, [data, interval]);

  return <div ref={hostRef} className="kline-chart" aria-label="K线图">
    <div ref={chartRef} className="chart-canvas" />
    {hover ? <><svg className="chart-callout" aria-hidden="true"><line x1={hover.anchor.x} y1={hover.anchor.y} x2={hover.left} y2={hover.top + 24} /></svg><EventTooltip hover={hover} /></> : null}
  </div>;
}
