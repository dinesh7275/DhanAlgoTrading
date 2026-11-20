"""SQLite trade journal: every signal (taken or rejected, and why), every trade, every warning.

This is the learning record. The report and dashboard are built from it, and it survives
restarts so an open paper position is picked up again.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import Fill, OptionContract, OptionType, Position, Signal
from .risk import DayStats

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    strategy TEXT NOT NULL,
    option_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    features TEXT NOT NULL,
    taken INTEGER NOT NULL,
    decision TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    security_id TEXT,
    underlying TEXT NOT NULL,
    expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_charges REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    max_price REAL NOT NULL,
    min_price REAL NOT NULL,
    last_price REAL NOT NULL,
    last_update TEXT NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    exit_charges REAL,
    exit_reason TEXT,
    gross_pnl REAL,
    net_pnl REAL,
    status TEXT NOT NULL DEFAULT 'OPEN'
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


class Journal:
    def __init__(self, path: str | Path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- writes -------------------------------------------------------------------------------

    def record_signal(self, signal: Signal, features: dict[str, Any], taken: bool, decision: str) -> int:
        cur = self.db.execute(
            "INSERT INTO signals (time, strategy, option_type, reason, features, taken, decision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (signal.time.isoformat(), signal.strategy, signal.option_type.value, signal.reason,
             json.dumps(features, default=str), int(taken), decision))
        self.db.commit()
        return int(cur.lastrowid)

    def open_trade(self, signal_id: int, strategy: str, fill: Fill, stop: float, target: float) -> Position:
        c = fill.contract
        cur = self.db.execute(
            "INSERT INTO trades (signal_id, strategy, symbol, security_id, underlying, expiry, strike,"
            " option_type, lot_size, quantity, entry_time, entry_price, entry_charges, stop_price,"
            " target_price, max_price, min_price, last_price, last_update)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (signal_id, strategy, c.symbol, c.security_id, c.underlying, c.expiry.isoformat(), c.strike,
             c.option_type.value, c.lot_size, fill.quantity, fill.time.isoformat(), fill.price,
             fill.charges.total, stop, target, fill.price, fill.price, fill.price, fill.time.isoformat()))
        self.db.commit()
        return Position(int(cur.lastrowid), c, fill.quantity, fill.price, fill.time, fill.charges.total,
                        stop, target, strategy, fill.price, fill.price, fill.price)

    def update_mark(self, pos: Position, now: datetime) -> None:
        self.db.execute("UPDATE trades SET max_price=?, min_price=?, last_price=?, last_update=? WHERE id=?",
                        (pos.max_price, pos.min_price, pos.last_price, now.isoformat(), pos.trade_id))
        self.db.commit()

    def close_trade(self, pos: Position, fill: Fill, reason: str) -> float:
        gross = (fill.price - pos.entry_price) * pos.quantity
        net = gross - pos.entry_charges - fill.charges.total
        self.db.execute(
            "UPDATE trades SET exit_time=?, exit_price=?, exit_charges=?, exit_reason=?, gross_pnl=?,"
            " net_pnl=?, max_price=?, min_price=?, last_price=?, last_update=?, status='CLOSED' WHERE id=?",
            (fill.time.isoformat(), fill.price, fill.charges.total, reason, gross, net, pos.max_price,
             pos.min_price, fill.price, fill.time.isoformat(), pos.trade_id))
        self.db.commit()
        return net

    def log_event(self, now: datetime, level: str, message: str) -> None:
        self.db.execute("INSERT INTO events (time, level, message) VALUES (?, ?, ?)",
                        (now.isoformat(), level, message))
        self.db.commit()

    # --- reads --------------------------------------------------------------------------------

    def open_position(self) -> Position | None:
        row = self.db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        contract = OptionContract(row["underlying"], date.fromisoformat(row["expiry"]), row["strike"],
                                  OptionType(row["option_type"]), row["lot_size"], row["security_id"])
        return Position(row["id"], contract, row["quantity"], row["entry_price"],
                        datetime.fromisoformat(row["entry_time"]), row["entry_charges"], row["stop_price"],
                        row["target_price"], row["strategy"], row["max_price"], row["min_price"],
                        row["last_price"])

    def closed_trades(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY exit_time, id")
        return [dict(r) for r in rows]

    def open_trades(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id")]

    def recent_signals(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def day_stats(self, day: date) -> DayStats:
        rows = self.db.execute(
            "SELECT net_pnl, status FROM trades WHERE substr(entry_time, 1, 10) = ? ORDER BY id",
            (day.isoformat(),)).fetchall()
        closed = [r["net_pnl"] for r in rows if r["status"] == "CLOSED"]
        streak = 0
        for pnl in reversed(closed):
            if pnl >= 0:
                break
            streak += 1
        return DayStats(trades=len(rows), net_pnl=sum(closed), consecutive_losses=streak)

    def realized_net(self) -> float:
        value = self.db.execute("SELECT COALESCE(SUM(net_pnl), 0) FROM trades WHERE status='CLOSED'").fetchone()
        return float(value[0])
