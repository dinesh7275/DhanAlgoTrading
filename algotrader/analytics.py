"""Performance statistics and the go-live checklist, computed from closed trades."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from .config import Settings


def summarize(trades: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    start = settings.account.starting_capital
    nets = [t["net_pnl"] for t in trades]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    charges = sum(t["entry_charges"] + t["exit_charges"] for t in trades)

    equity, peak, max_dd, max_dd_pct = start, start, 0.0, 0.0
    curve = [{"time": None, "equity": start}]
    for t in trades:
        equity += t["net_pnl"]
        peak = max(peak, equity)
        if peak - equity > max_dd:
            max_dd, max_dd_pct = peak - equity, (peak - equity) / peak * 100
        curve.append({"time": t["exit_time"], "equity": round(equity, 2)})

    # R multiple: net P&L as a multiple of the money at risk to the stop.
    r_multiples = [t["net_pnl"] / ((t["entry_price"] - t["stop_price"]) * t["quantity"])
                   for t in trades if t["entry_price"] > t["stop_price"]]

    days = {t["entry_time"][:10] for t in trades}
    n = len(trades)
    summary = {
        "starting_capital": start,
        "equity": round(equity, 2),
        "trades": n,
        "trading_days": len(days),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0.0,
        "net_pnl": round(sum(nets), 2),
        "return_pct": round(sum(nets) / start * 100, 2),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "expectancy": round(sum(nets) / n, 2) if n else 0.0,
        "avg_r": round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0.0,
        # None means "no losing trades yet" (infinite profit factor).
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else (None if wins else 0.0),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_charges": round(charges, 2),
        "charges_pct_of_gross": round(charges / (gross_profit + gross_loss) * 100, 1)
        if gross_profit + gross_loss else 0.0,
        "by_strategy": _breakdown(trades, lambda t: t["strategy"]),
        "by_exit_reason": _breakdown(trades, lambda t: t["exit_reason"]),
        "by_option_type": _breakdown(trades, lambda t: t["option_type"]),
        "by_weekday": _breakdown(trades, lambda t: datetime.fromisoformat(t["entry_time"]).strftime("%A")),
        "equity_curve": curve,
    }
    summary["go_live"] = go_live_checks(summary, settings)
    return summary


def _breakdown(trades: list[dict[str, Any]], key) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        groups[key(t)].append(t["net_pnl"])
    return {k: {"trades": len(v), "net_pnl": round(sum(v), 2),
                "win_rate_pct": round(sum(1 for x in v if x > 0) / len(v) * 100, 1)}
            for k, v in sorted(groups.items())}


def go_live_checks(summary: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    g = settings.go_live
    pf = summary["profit_factor"]
    checks = [
        ("Enough paper trades", summary["trades"] >= g.min_trades, f"{summary['trades']} / {g.min_trades}"),
        ("Enough trading days", summary["trading_days"] >= g.min_trading_days,
         f"{summary['trading_days']} / {g.min_trading_days}"),
        ("Profitable after all charges", summary["expectancy"] > 0,
         f"₹{summary['expectancy']:,.2f} per trade"),
        ("Profit factor", pf is None or pf >= g.min_profit_factor,
         f"{'no losses yet' if pf is None else pf} (need ≥ {g.min_profit_factor})"),
        ("Drawdown within limit", summary["trades"] > 0 and summary["max_drawdown_pct"] <= g.max_drawdown_pct,
         f"{summary['max_drawdown_pct']}% (limit {g.max_drawdown_pct}%)"),
    ]
    return [{"check": name, "passed": bool(ok), "detail": detail} for name, ok, detail in checks]


def format_comparison(rows: list[tuple[str, dict[str, Any]]], title: str) -> str:
    lines = [title, "=" * len(title),
             f"{'':<14}{'trades':>7}{'win %':>7}{'net P&L':>12}{'per trade':>11}{'PF':>6}{'max DD':>8}",]
    for label, s in rows:
        pf = s["profit_factor"]
        lines.append(f"{label:<14}{s['trades']:>7}{s['win_rate_pct']:>7}{s['net_pnl']:>12,.0f}"
                     f"{s['expectancy']:>11,.0f}{'n/a' if pf is None else pf:>6}{s['max_drawdown_pct']:>7}%")
    lines.append("\nPF = profit factor (gross wins / gross losses). Max DD = largest fall from a peak, % of equity.")
    return "\n".join(lines)


def format_report(summary: dict[str, Any], title: str) -> str:
    s = summary
    lines = [
        title,
        "=" * len(title),
        f"Capital      ₹{s['starting_capital']:,.0f} → ₹{s['equity']:,.2f}  ({s['return_pct']:+.2f}%)",
        f"Trades       {s['trades']} over {s['trading_days']} days | {s['wins']} wins, {s['losses']} losses"
        f" | win rate {s['win_rate_pct']}%",
        f"Per trade    expectancy ₹{s['expectancy']:,.2f} | avg win ₹{s['avg_win']:,.2f}"
        f" | avg loss ₹{s['avg_loss']:,.2f} | avg R {s['avg_r']}",
        f"Quality      profit factor {s['profit_factor'] if s['profit_factor'] is not None else 'n/a'} | max drawdown ₹{s['max_drawdown']:,.2f}"
        f" ({s['max_drawdown_pct']}%)",
        f"Costs        ₹{s['total_charges']:,.2f} in charges = {s['charges_pct_of_gross']}% of gross P&L moved",
    ]
    for label, key in (("Strategy", "by_strategy"), ("Exit reason", "by_exit_reason"), ("Option type", "by_option_type"),
                       ("Weekday", "by_weekday")):
        if s[key]:
            lines.append(f"\nBy {label.lower()}:")
            for name, b in s[key].items():
                lines.append(f"  {name:<18} {b['trades']:>4} trades  ₹{b['net_pnl']:>11,.2f}"
                             f"  win {b['win_rate_pct']}%")
    lines.append("\nGo-live checklist (all must pass before considering real money):")
    for c in s["go_live"]:
        lines.append(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return "\n".join(lines)
