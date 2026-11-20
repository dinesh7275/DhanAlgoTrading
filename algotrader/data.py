"""Market data providers. All of them are read-only: nothing in this module can place an order.

- YahooData:      live NIFTY spot from Yahoo; option premiums modelled with Black-Scholes + India VIX.
- DhanData:       live NIFTY spot and real option-chain bid/ask from Dhan's Data API.
- HistoricalData: replays stored candles for backtests, pricing options the same way as YahooData.
"""

from __future__ import annotations

import logging
import time as time_mod
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd
import requests

from .config import DataConfig
from .models import IST, OptionContract, OptionType, Quote
from .pricing import ModelPricer

log = logging.getLogger(__name__)

NIFTY_YAHOO = "^NSEI"
VIX_YAHOO = "^INDIAVIX"


class StaleDataError(RuntimeError):
    """The newest data point is too old to trade on."""


class MarketData(ABC):
    label = ""

    @abstractmethod
    def candles(self, now: datetime) -> pd.DataFrame:
        """Completed candles for today and the previous few sessions: columns open/high/low/close,
        indexed by candle start (IST). Strategies filter to today themselves."""

    @abstractmethod
    def spot(self, now: datetime) -> float:
        """Latest underlying price."""

    @abstractmethod
    def quote(self, contract: OptionContract, now: datetime) -> Quote:
        """Current bid/ask for an option."""

    def premium_range(self, contract: OptionContract, now: datetime) -> tuple[float, float] | None:
        """Low/high premium during the last candle, if known. Lets backtests see intra-candle stops."""
        return None


def normalise_candles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    index = pd.DatetimeIndex(df.index)
    df.index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    return df.sort_index()


HISTORY_DAYS = 7  # calendar days of candles handed to strategies (covers weekends and a holiday)


def completed_recent(df: pd.DataFrame, now: datetime, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    start = pd.Timestamp(now.date() - timedelta(days=HISTORY_DAYS), tz=IST)
    end = pd.Timestamp(now) - pd.Timedelta(minutes=minutes)
    return df.loc[start:end]


class _TTLCache:
    def __init__(self) -> None:
        self._items: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any, ttl: float, load: Callable[[], Any]) -> Any:
        hit = self._items.get(key)
        if hit and time_mod.monotonic() - hit[0] < ttl:
            return hit[1]
        value = load()
        self._items[key] = (time_mod.monotonic(), value)
        return value


class YahooData(MarketData):
    label = "Yahoo spot + modelled premiums (Black-Scholes, India VIX)"

    def __init__(self, cfg: DataConfig, candle_minutes: int, pricer: ModelPricer):
        import yfinance  # imported lazily so tests and backtests on stored data don't need it

        self._yf = yfinance
        self.cfg = cfg
        self.minutes = candle_minutes
        self.pricer = pricer
        self._cache = _TTLCache()

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raw = self._yf.download(ticker, period=period, interval=interval, progress=False,
                                auto_adjust=False, threads=False)
        return normalise_candles(raw)

    def candles(self, now: datetime) -> pd.DataFrame:
        df = self._cache.get("candles", self.cfg.poll_seconds - 1,
                             lambda: self._download(NIFTY_YAHOO, "5d", f"{self.minutes}m"))
        return completed_recent(df, now, self.minutes)

    def spot(self, now: datetime) -> float:
        df = self._cache.get("spot", self.cfg.poll_seconds - 1,
                             lambda: self._download(NIFTY_YAHOO, "1d", "1m"))
        if df.empty:
            raise StaleDataError("no NIFTY prices from Yahoo")
        age = now - (df.index[-1] + timedelta(minutes=1))
        if age > timedelta(minutes=self.cfg.max_staleness_minutes):
            raise StaleDataError(f"latest NIFTY price is {age} old")
        return float(df.close.iloc[-1])

    def vol(self) -> float:
        df = self._cache.get("vix", 900, lambda: self._download(VIX_YAHOO, "10d", "1d"))
        if df.empty:
            raise StaleDataError("no India VIX from Yahoo")
        return float(df.close.iloc[-1]) / 100

    def quote(self, contract: OptionContract, now: datetime) -> Quote:
        return self.pricer.quote(contract, self.spot(now), self.vol(), now)


class DhanData(MarketData):
    """Read-only Dhan Data API client. It calls only market-data endpoints, never order endpoints."""

    label = "Dhan live option chain"
    BASE_URL = "https://api.dhan.co/v2"
    NIFTY_ID = 13
    CHAIN_MIN_INTERVAL = 3.1  # Dhan allows one option-chain request per 3 seconds

    def __init__(self, cfg: DataConfig, candle_minutes: int, client_id: str, access_token: str):
        self.cfg = cfg
        self.minutes = candle_minutes
        self._session = requests.Session()
        self._session.headers.update({"access-token": access_token, "client-id": client_id,
                                      "Content-Type": "application/json", "Accept": "application/json"})
        self._cache = _TTLCache()
        self._last_chain_call = 0.0

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self._session.post(f"{self.BASE_URL}{path}", json=body, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Dhan {path} returned HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def candles(self, now: datetime) -> pd.DataFrame:
        def load() -> pd.DataFrame:
            first = now.date() - timedelta(days=HISTORY_DAYS)
            data = self._post("/charts/intraday", {
                "securityId": str(self.NIFTY_ID), "exchangeSegment": "IDX_I", "instrument": "INDEX",
                "interval": str(self.minutes), "fromDate": f"{first} 09:00:00", "toDate": f"{now:%Y-%m-%d} 15:30:00",
            })
            if not data.get("timestamp"):
                return normalise_candles(pd.DataFrame())
            index = pd.to_datetime(data["timestamp"], unit="s", utc=True)
            frame = pd.DataFrame({k: data[k] for k in ("open", "high", "low", "close")}, index=index)
            return normalise_candles(frame)

        return completed_recent(self._cache.get("candles", self.cfg.poll_seconds - 1, load), now, self.minutes)

    def spot(self, now: datetime) -> float:
        def load() -> float:
            data = self._post("/marketfeed/ltp", {"IDX_I": [self.NIFTY_ID]})
            return float(data["data"]["IDX_I"][str(self.NIFTY_ID)]["last_price"])

        return self._cache.get("spot", 2, load)

    def _chain(self, expiry: date) -> dict[str, Any]:
        def load() -> dict[str, Any]:
            wait = self.CHAIN_MIN_INTERVAL - (time_mod.monotonic() - self._last_chain_call)
            if wait > 0:
                time_mod.sleep(wait)
            self._last_chain_call = time_mod.monotonic()
            data = self._post("/optionchain", {"UnderlyingScrip": self.NIFTY_ID, "UnderlyingSeg": "IDX_I",
                                               "Expiry": expiry.isoformat()})
            return data["data"]["oc"]

        return self._cache.get(("chain", expiry), 5, load)

    def quote(self, contract: OptionContract, now: datetime) -> Quote:
        chain = self._chain(contract.expiry)
        side = "ce" if contract.option_type is OptionType.CE else "pe"
        for strike, legs in chain.items():
            if float(strike) == contract.strike and side in legs:
                leg = legs[side]
                ltp = float(leg.get("last_price") or 0)
                bid = float(leg.get("top_bid_price") or ltp)
                ask = float(leg.get("top_ask_price") or ltp)
                if ask <= 0:
                    raise StaleDataError(f"no price for {contract.symbol}")
                iv = leg.get("implied_volatility")
                return Quote(bid=bid, ask=ask, ltp=ltp, time=now, iv=float(iv) / 100 if iv else None)
        raise StaleDataError(f"{contract.symbol} not found in Dhan option chain")


class HistoricalData(MarketData):
    """Replays stored candles. Volatility is the previous day's India VIX close, so nothing from the future leaks in."""

    label = "Historical candles + modelled premiums"

    def __init__(self, candles: pd.DataFrame, vix_close: pd.Series, candle_minutes: int, pricer: ModelPricer):
        self._candles = normalise_candles(candles)
        self._vix = vix_close.sort_index()
        self.minutes = candle_minutes
        self.pricer = pricer

    def candles(self, now: datetime) -> pd.DataFrame:
        return completed_recent(self._candles, now, self.minutes)

    def _last_candle(self, now: datetime) -> pd.Series:
        done = self.candles(now)
        done = done[done.index.date == now.date()]
        if done.empty:
            raise StaleDataError(f"no completed candle at {now}")
        return done.iloc[-1]

    def spot(self, now: datetime) -> float:
        return float(self._last_candle(now).close)

    def vol(self, now: datetime) -> float:
        before = self._vix[self._vix.index < now.date()]
        value = before.iloc[-1] if len(before) else self._vix.iloc[0]
        return float(value) / 100

    def quote(self, contract: OptionContract, now: datetime) -> Quote:
        return self.pricer.quote(contract, self.spot(now), self.vol(now), now)

    def premium_range(self, contract: OptionContract, now: datetime) -> tuple[float, float]:
        candle = self._last_candle(now)
        vol = self.vol(now)
        a = self.pricer.premium(contract, float(candle.high), vol, now)
        b = self.pricer.premium(contract, float(candle.low), vol, now)
        return min(a, b), max(a, b)
