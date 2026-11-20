"""Pre-trade and in-trade risk rules. Every entry must pass `can_enter`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import RiskConfig, StrategyConfig


@dataclass(frozen=True)
class DayStats:
    trades: int = 0
    net_pnl: float = 0.0
    consecutive_losses: int = 0


class RiskManager:
    def __init__(self, cfg: RiskConfig, strategy_cfg: StrategyConfig):
        self.cfg = cfg
        self.no_entry_after = strategy_cfg.no_entry_after

    def can_enter(self, now: datetime, day: DayStats, has_position: bool,
                  cash: float, required: float) -> tuple[bool, str]:
        cfg = self.cfg
        if has_position:
            return False, "a position is already open"
        if now.time() >= self.no_entry_after:
            return False, f"no new entries after {self.no_entry_after:%H:%M}"
        if day.trades >= cfg.max_trades_per_day:
            return False, f"daily trade limit reached ({cfg.max_trades_per_day})"
        if day.net_pnl <= -cfg.max_daily_loss:
            return False, f"daily loss limit reached (₹{cfg.max_daily_loss:,.0f})"
        if day.consecutive_losses >= cfg.max_consecutive_losses:
            return False, f"{day.consecutive_losses} losses in a row today"
        if required > cash:
            return False, f"not enough paper capital: need ₹{required:,.0f}, have ₹{cash:,.0f}"
        return True, "ok"

    def must_square_off(self, now: datetime) -> bool:
        return now.time() >= self.cfg.square_off_time

    def daily_loss_breached(self, day: DayStats, open_pnl: float) -> bool:
        return day.net_pnl + open_pnl <= -self.cfg.max_daily_loss
