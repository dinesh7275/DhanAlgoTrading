"""Trading strategies. A strategy looks at completed candles and returns a Signal or None.

Strategies never see the future: they only get candles that have fully closed by `now`.
Each strategy carries its own exits (stop, target, optional time limit) on the option premium.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta

import pandas as pd

from .config import StrategyConfig
from .models import IST, MARKET_OPEN, OptionType, Signal


class Strategy(ABC):
    name = ""

    def __init__(self, cfg: StrategyConfig, own):
        self.cfg = cfg
        self.stop_loss_pct: float = own.stop_loss_pct
        self.target_pct: float = own.target_pct
        self.max_hold_minutes: int = own.max_hold_minutes
        self._day: date | None = None
        self._fired: set[OptionType] = set()

    def on_candles(self, candles: pd.DataFrame, now: datetime) -> Signal | None:
        if now.date() != self._day:
            self._day, self._fired = now.date(), set()
        today = candles[candles.index.date == now.date()] if not candles.empty else candles
        if today.empty:
            return None
        candle = pd.Timedelta(minutes=self.cfg.candle_minutes)
        if now - (today.index[-1] + candle) > candle:
            return None  # the newest candle is old (e.g. after a restart): don't act on history

        found = self.check(candles, today, now)
        if found is None:
            return None
        option_type, reason, features = found
        if option_type in self._fired:
            return None
        self._fired.add(option_type)
        open_at = datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST)
        features.update(minutes_since_open=int((now - open_at).total_seconds() // 60),
                        weekday=now.strftime("%A"))
        return Signal(time=now, strategy=self.name, option_type=option_type, reason=reason, features=features)

    @abstractmethod
    def check(self, candles: pd.DataFrame, today: pd.DataFrame,
              now: datetime) -> tuple[OptionType, str, dict] | None:
        """Look at the newest completed candle (today.iloc[-1]); return (option type, reason, features)."""


class OpeningRangeBreakout(Strategy):
    """15-minute opening range breakout. Best on trend days.

    1. Record the high and low of the first `range_minutes` (09:15-09:30 by default).
    2. Skip the day if that range is unusually narrow or wide.
    3. The first candle that *closes* above the high buys an ATM call; the first that closes below
       the low buys an ATM put. Each direction fires at most once per day.
    """

    name = "ORB"

    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg, cfg.orb)

    def check(self, candles, today, now):
        o = self.cfg.orb
        open_at = datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST)
        range_end = open_at + timedelta(minutes=o.range_minutes)
        opening = today[today.index < range_end]
        if len(opening) < o.range_minutes // self.cfg.candle_minutes:
            return None
        after = today[today.index >= range_end]
        if after.empty:
            return None

        high, low = float(opening.high.max()), float(opening.low.min())
        width_pct = (high - low) / low * 100
        if not o.min_range_pct <= width_pct <= o.max_range_pct:
            return None

        close = float(after.close.iloc[-1])
        earlier = after.close.iloc[:-1]
        if close > high and not (earlier > high).any():
            option_type, side = OptionType.CE, "above"
        elif close < low and not (earlier < low).any():
            option_type, side = OptionType.PE, "below"
        else:
            return None
        return option_type, (f"{self.cfg.candle_minutes}m candle closed {side} opening range "
                             f"({low:.2f}-{high:.2f}) at {close:.2f}"), {
            "range_high": high,
            "range_low": low,
            "range_width_pct": round(width_pct, 3),
            "breakout_close": close,
            "breakout_candle": after.index[-1].isoformat(),
        }


class PreviousDayBreakout(Strategy):
    """Breakout of the previous session's high or low. Best on trend days.

    After the first `start_after_minutes`, a candle that closes above yesterday's high, when the
    candle before it closed at or below it, buys an ATM call (mirror for puts below yesterday's
    low). A gap open beyond the level doesn't count: price must cross it during the day.
    """

    name = "PREV_DAY"

    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg, cfg.prev_day)

    def check(self, candles, today, now):
        before = candles[candles.index.date < now.date()]
        if before.empty or len(today) < 2:
            return None
        prev_day = before[before.index.date == before.index[-1].date()]
        pdh, pdl = float(prev_day.high.max()), float(prev_day.low.min())

        start = datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST) + timedelta(
            minutes=self.cfg.prev_day.start_after_minutes)
        if today.index[-1] < start:
            return None
        close, prev_close = float(today.close.iloc[-1]), float(today.close.iloc[-2])
        if close > pdh >= prev_close:
            option_type, side, level = OptionType.CE, "above previous day high", pdh
        elif close < pdl <= prev_close:
            option_type, side, level = OptionType.PE, "below previous day low", pdl
        else:
            return None
        return option_type, f"{self.cfg.candle_minutes}m candle closed {side} {level:.2f} at {close:.2f}", {
            "prev_day": prev_day.index[-1].date().isoformat(),
            "prev_high": pdh,
            "prev_low": pdl,
            "prev_range_pct": round((pdh - pdl) / pdl * 100, 3),
            "gap_pct": round((float(today.open.iloc[0]) - float(prev_day.close.iloc[-1]))
                             / float(prev_day.close.iloc[-1]) * 100, 3),
            "breakout_close": close,
        }


class RangeReversal(Strategy):
    """Fade the edge of a sideways market. Best on range days, when the breakout strategies sit idle.

    From `earliest`, look at the previous `lookback_candles` candles. If their high-low range is
    tight (between min_range_pct and max_range_pct), and the newest candle reaches the top of that
    range but closes back inside as a red candle in its lower half, buy an ATM put (mirror at the
    bottom for calls). Targets are small and there is a time limit, since the idea is a quick move
    back across the range.
    """

    name = "REVERSAL"

    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg, cfg.reversal)

    def check(self, candles, today, now):
        r = self.cfg.reversal
        if now.time() < r.earliest or len(today) < r.lookback_candles + 1:
            return None
        window, last = today.iloc[-r.lookback_candles - 1:-1], today.iloc[-1]
        top, bottom = float(window.high.max()), float(window.low.min())
        width_pct = (top - bottom) / bottom * 100
        if not r.min_range_pct <= width_pct <= r.max_range_pct:
            return None

        o, h, l, c = (float(last[k]) for k in ("open", "high", "low", "close"))
        mid_candle = (h + l) / 2
        if h >= top and c < top and c < o and c < mid_candle:
            option_type, edge, level = OptionType.PE, "top", top
        elif l <= bottom and c > bottom and c > o and c > mid_candle:
            option_type, edge, level = OptionType.CE, "bottom", bottom
        else:
            return None
        return option_type, (f"rejected at {edge} of {r.lookback_candles}-candle range "
                             f"({bottom:.2f}-{top:.2f}), closed {c:.2f}"), {
            "range_high": top,
            "range_low": bottom,
            "range_width_pct": round(width_pct, 3),
            "rejected_level": level,
            "candle": {"open": o, "high": h, "low": l, "close": c},
        }


STRATEGIES = {"orb": OpeningRangeBreakout, "prev_day": PreviousDayBreakout, "reversal": RangeReversal}


def build_strategies(cfg: StrategyConfig, only: tuple[str, ...] | None = None) -> list[Strategy]:
    """Strategies in the order listed; when two signal on the same candle, the first one wins."""
    return [STRATEGIES[name](cfg) for name in (only or cfg.enabled)]
