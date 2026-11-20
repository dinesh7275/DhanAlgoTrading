import pandas as pd
import pytest

from algotrader.analytics import summarize
from algotrader.backtest import run_backtest
from algotrader.config import Settings
from algotrader.journal import Journal

from helpers import ist, make_day


def vix(day, value=15.0):
    return pd.Series([value], index=[(day - pd.Timedelta(days=1)).date()])


def backtest(closes, settings=None, only=None):
    settings = settings or Settings()
    day = ist(2026, 11, 19, 0, 0)
    journal = Journal(":memory:")
    run_backtest(settings, make_day(day, closes, spread=2.0), vix(day), journal, only)
    return journal, settings


def test_breakout_then_rally_hits_target():
    closes = [25500, 25520, 25480, 25530] + [25530 + 25 * i for i in range(1, 30)]
    journal, s = backtest(closes)
    trades = journal.closed_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["option_type"] == "CE" and t["exit_reason"] == "TARGET" and t["quantity"] == 65
    assert t["net_pnl"] > 0
    assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["entry_charges"] - t["exit_charges"])
    assert journal.open_position() is None


def test_breakout_then_reversal_hits_stop():
    closes = [25500, 25520, 25480, 25530] + [25530 - 30 * i for i in range(1, 30)]
    journal, _ = backtest(closes)
    reasons = [t["exit_reason"] for t in journal.closed_trades()]
    assert reasons[0] == "STOP_LOSS"
    assert all(t["net_pnl"] < 0 for t in journal.closed_trades() if t["exit_reason"] == "STOP_LOSS")


def test_flat_day_is_squared_off_by_1515():
    closes = [25500, 25520, 25480, 25530] + [25531] * 70
    journal, _ = backtest(closes)
    t = journal.closed_trades()[0]
    assert t["exit_reason"] == "SQUARE_OFF"
    assert t["exit_time"][11:16] == "15:15"


def test_signal_skipped_when_capital_too_small():
    s = Settings()
    from dataclasses import replace
    s = replace(s, account=replace(s.account, starting_capital=1000))
    closes = [25500, 25520, 25480, 25530] + [25600] * 10
    journal, _ = backtest(closes, s)
    assert journal.closed_trades() == []
    sig = journal.recent_signals()[0]
    assert sig["taken"] == 0 and "capital" in sig["decision"]


def test_summary_and_go_live_checks():
    closes = [25500, 25520, 25480, 25530] + [25530 + 25 * i for i in range(1, 30)]
    journal, s = backtest(closes)
    summary = summarize(journal.closed_trades(), s)
    assert summary["trades"] == 1 and summary["win_rate_pct"] == 100.0
    assert summary["profit_factor"] is None
    checks = {c["check"]: c["passed"] for c in summary["go_live"]}
    assert checks["Enough paper trades"] is False
    assert checks["Profitable after all charges"] is True


def test_dashboard_snapshot(tmp_path):
    from algotrader.dashboard import create_app

    closes = [25500, 25520, 25480, 25530] + [25530 + 25 * i for i in range(1, 30)]
    s = Settings()
    path = tmp_path / "j.db"
    day = ist(2026, 11, 19, 0, 0)
    run_backtest(s, make_day(day, closes, spread=2.0), vix(day), Journal(path))
    client = create_app(path, s, "Test").test_client()
    data = client.get("/api/snapshot").get_json()
    assert data["summary"]["trades"] == 1 and data["signals"][0]["taken"] == 1
    assert client.get("/").status_code == 200
    assert client.post("/api/snapshot").status_code == 405  # nothing writable


def test_reversal_trade_closed_by_time_limit():
    # sideways until a rejection at 10:20, then flat: neither stop nor target, so the 45-minute limit exits
    closes = [25490, 25530] * 6 + [25490] * 30
    journal, _ = backtest(closes, only=("reversal",))
    t = journal.closed_trades()[0]
    assert t["strategy"] == "REVERSAL" and t["option_type"] == "PE"
    assert t["exit_reason"] == "TIME_LIMIT"
    assert t["entry_time"][11:16] == "10:20" and t["exit_time"][11:16] == "11:05"


def test_signal_during_open_trade_is_recorded_as_skipped():
    # ORB buys a call at 09:35; the rejection candle at 10:20 arrives while it is still open
    closes = [25500, 25520, 25480, 25530] + [25510, 25560] * 10
    journal, _ = backtest(closes, only=("orb", "reversal"))
    skipped = [s for s in journal.recent_signals(50) if not s["taken"]]
    assert any(s["strategy"] == "REVERSAL" and "already open" in s["decision"] for s in skipped)
