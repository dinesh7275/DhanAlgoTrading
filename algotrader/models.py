"""Core value types shared by every part of the app."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    lot_size: int
    security_id: str | None = None

    @property
    def symbol(self) -> str:
        return f"{self.underlying} {self.expiry:%d%b%y} {int(self.strike)} {self.option_type.value}".upper()


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    ltp: float
    time: datetime
    iv: float | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    def moved_to(self, mid: float) -> "Quote":
        """Same spread, centred on a new mid. Used when a stop/target level is touched inside a candle."""
        half = (self.ask - self.bid) / 2
        return replace(self, bid=max(mid - half, 0.05), ask=mid + half, ltp=mid)


@dataclass(frozen=True)
class Signal:
    time: datetime
    strategy: str
    option_type: OptionType
    reason: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Charges:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


@dataclass(frozen=True)
class Fill:
    contract: OptionContract
    side: Side
    quantity: int
    price: float
    time: datetime
    charges: Charges


@dataclass
class Position:
    trade_id: int
    contract: OptionContract
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_charges: float
    stop_price: float
    target_price: float
    strategy: str
    max_price: float
    min_price: float
    last_price: float


def is_trading_day(d: date, holidays: tuple[date, ...] = ()) -> bool:
    return d.weekday() < 5 and d not in holidays


def is_market_open(now: datetime, holidays: tuple[date, ...] = ()) -> bool:
    now = now.astimezone(IST)
    return is_trading_day(now.date(), holidays) and MARKET_OPEN <= now.time() < MARKET_CLOSE
