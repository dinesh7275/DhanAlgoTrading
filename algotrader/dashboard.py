"""Read-only web dashboard over the journal. It has no endpoints that change anything."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify

from .analytics import summarize
from .config import Settings
from .journal import Journal


def create_app(journal_path: Path, settings: Settings, title: str) -> Flask:
    app = Flask(__name__)

    def snapshot() -> dict:
        journal = Journal(journal_path)
        try:
            summary = summarize(journal.closed_trades(), settings)
            signals = journal.recent_signals(15)
            for s in signals:
                s["features"] = json.loads(s["features"])
            return {
                "title": title,
                "summary": summary,
                "open": journal.open_trades(),
                "trades": list(reversed(journal.closed_trades()))[:25],
                "signals": signals,
                "events": journal.recent_events(10),
            }
        finally:
            journal.close()

    @app.get("/api/snapshot")
    def api_snapshot():
        return jsonify(snapshot())

    @app.get("/")
    def index():
        return PAGE.replace("{{TITLE}}", title)

    return app


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e4e4df;
          --up:#1a7f4b; --down:#b42318; --accent:#2f5bd3; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151514; --card:#1f1f1d; --ink:#ececea; --muted:#9b9b95; --line:#33332f;
            --up:#4cc38a; --down:#f97066; --accent:#7c9cff; } }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background:var(--bg); color:var(--ink); }
  main { max-width:1180px; margin:0 auto; padding:24px 20px 48px; }
  header { display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; }
  .badge { font-size:12px; padding:3px 8px; border-radius:99px; background:var(--accent); color:#fff; }
  .muted { color:var(--muted); }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:20px 0; }
  .tile, .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .tile .k { font-size:12px; color:var(--muted); }
  .tile .v { font-size:20px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }
  .grid2 { display:grid; grid-template-columns:2fr 1fr; gap:12px; }
  @media (max-width:860px) { .grid2 { grid-template-columns:1fr; } }
  h2 { font-size:14px; margin:0 0 10px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { font-size:12px; color:var(--muted); font-weight:500; }
  td.num, th.num { text-align:right; }
  .up { color:var(--up); } .down { color:var(--down); }
  .card { margin-top:12px; overflow-x:auto; }
  svg { width:100%; height:220px; display:block; }
  .check { display:flex; gap:8px; padding:5px 0; }
  .pass { color:var(--up); font-weight:600; } .fail { color:var(--down); font-weight:600; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title">{{TITLE}} <span class="badge">PAPER ONLY</span></h1>
    <span class="muted" id="updated"></span>
  </header>
  <div class="tiles" id="tiles"></div>
  <div class="grid2">
    <div class="card"><h2>Equity after each trade</h2><svg id="curve" role="img" aria-label="Equity curve"></svg></div>
    <div class="card"><h2>Go-live checklist</h2><div id="checks"></div>
      <p class="muted">Every check must pass on paper before real money is worth considering.</p></div>
  </div>
  <div class="card"><h2>Open position</h2><div id="open"></div></div>
  <div class="card"><h2>Recent trades</h2><div id="trades"></div></div>
  <div class="card"><h2>Recent signals (taken and skipped)</h2><div id="signals"></div></div>
  <div class="card"><h2>Warnings</h2><div id="events"></div></div>
</main>
<script>
const inr = v => (v < 0 ? "-₹" : "₹") + Math.abs(v).toLocaleString("en-IN", {maximumFractionDigits: 2});
const cls = v => v > 0 ? "up" : v < 0 ? "down" : "";
const t = s => s ? s.replace("T", " ").slice(0, 16) : "";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function table(rows, cols) {
  if (!rows.length) return '<p class="muted">Nothing yet.</p>';
  const head = cols.map(c => `<th class="${c.num ? "num" : ""}">${c.h}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(c => {
    const v = c.f(r); return `<td class="${c.num ? "num" : ""} ${c.c ? c.c(r) : ""}">${v}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function curve(points) {
  const svg = document.getElementById("curve");
  const W = svg.clientWidth || 600, H = 220, P = 36;
  if (points.length < 2) { svg.innerHTML = `<text x="${P}" y="${H/2}" fill="currentColor" opacity=".6">No closed trades yet</text>`; return; }
  const ys = points.map(p => p.equity), lo = Math.min(...ys), hi = Math.max(...ys), span = (hi - lo) || 1;
  const x = i => P + i * (W - P * 1.5) / (points.length - 1);
  const y = v => H - P / 2 - (v - lo) / span * (H - P * 1.2);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join("");
  const base = y(points[0].equity);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `
    <line x1="${P}" x2="${W - P/2}" y1="${base}" y2="${base}" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="3 3"/>
    <text x="2" y="${y(hi) + 4}" font-size="11" fill="currentColor" opacity=".6">${Math.round(hi)}</text>
    <text x="2" y="${y(lo) + 4}" font-size="11" fill="currentColor" opacity=".6">${Math.round(lo)}</text>
    <path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
}

async function refresh() {
  const d = await (await fetch("api/snapshot")).json();
  const s = d.summary;
  document.getElementById("updated").textContent = "Updated " + new Date().toLocaleTimeString();
  document.getElementById("tiles").innerHTML = [
    ["Paper equity", inr(s.equity), ""], ["Net P&L", inr(s.net_pnl), cls(s.net_pnl)],
    ["Trades", s.trades, ""], ["Win rate", s.win_rate_pct + "%", ""],
    ["Expectancy / trade", inr(s.expectancy), cls(s.expectancy)], ["Profit factor", s.profit_factor ?? "n/a", ""],
    ["Max drawdown", s.max_drawdown_pct + "%", ""], ["Charges paid", inr(s.total_charges), ""],
  ].map(([k, v, c]) => `<div class="tile"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");
  curve(s.equity_curve);
  document.getElementById("checks").innerHTML = s.go_live.map(c =>
    `<div class="check"><span class="${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</span>
     <span>${esc(c.check)}<br><span class="muted">${esc(c.detail)}</span></span></div>`).join("");
  document.getElementById("open").innerHTML = table(d.open, [
    {h: "Contract", f: r => esc(r.symbol)}, {h: "Qty", num: 1, f: r => r.quantity},
    {h: "Entry", num: 1, f: r => r.entry_price.toFixed(2)}, {h: "Last", num: 1, f: r => r.last_price.toFixed(2)},
    {h: "Stop", num: 1, f: r => r.stop_price.toFixed(2)}, {h: "Target", num: 1, f: r => r.target_price.toFixed(2)},
    {h: "Open P&L (gross)", num: 1, f: r => inr((r.last_price - r.entry_price) * r.quantity),
     c: r => cls(r.last_price - r.entry_price)}, {h: "Since", f: r => t(r.entry_time)}]);
  document.getElementById("trades").innerHTML = table(d.trades, [
    {h: "Entry", f: r => t(r.entry_time)}, {h: "Contract", f: r => esc(r.symbol)},
    {h: "Buy", num: 1, f: r => r.entry_price.toFixed(2)}, {h: "Sell", num: 1, f: r => r.exit_price.toFixed(2)},
    {h: "Exit", f: r => esc(r.exit_reason)}, {h: "Charges", num: 1, f: r => inr(r.entry_charges + r.exit_charges)},
    {h: "Net P&L", num: 1, f: r => inr(r.net_pnl), c: r => cls(r.net_pnl)}]);
  document.getElementById("signals").innerHTML = table(d.signals, [
    {h: "Time", f: r => t(r.time)}, {h: "Type", f: r => r.option_type},
    {h: "Taken", f: r => r.taken ? "yes" : "no", c: r => r.taken ? "up" : "muted"},
    {h: "Decision", f: r => esc(r.decision)}, {h: "Reason", f: r => esc(r.reason)}]);
  document.getElementById("events").innerHTML = table(d.events, [
    {h: "Time", f: r => t(r.time)}, {h: "Level", f: r => r.level}, {h: "Message", f: r => esc(r.message)}]);
}
refresh(); setInterval(refresh, 15000);
</script>
</body>
</html>
"""
