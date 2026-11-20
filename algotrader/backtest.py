"""Backtests replay historical candles through the exact same Engine used for paper trading."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .broker import PaperBroker
from .config import Settings
from .costs import CostModel
from .data import NIFTY_YAHOO, VIX_YAHOO, HistoricalData, StaleDataError, normalise_candles
from .engine import Engine
from .instruments import InstrumentResolver
from .journal import Journal
from .pricing import ModelPricer
from .risk import RiskManager
from .strategy import build_strategies

log = logging.getLogger(__name__)

MAX_YAHOO_INTRADAY_DAYS = 59


def download_history(days: int, candle_minutes: int) -> tuple[pd.DataFrame, pd.Series]:
    import yfinance as yf

    days = min(days, MAX_YAHOO_INTRADAY_DAYS)
    candles = normalise_candles(yf.download(NIFTY_YAHOO, period=f"{days}d", interval=f"{candle_minutes}m",
                                            progress=False, auto_adjust=False, threads=False))
    vix = normalise_candles(yf.download(VIX_YAHOO, period=f"{days + 15}d", interval="1d",
                                        progress=False, auto_adjust=False, threads=False))
    if candles.empty or vix.empty:
        raise RuntimeError("Yahoo returned no data; check your internet connection")
    vix_close = pd.Series(vix.close.values, index=[d.date() for d in vix.index])
    return candles, vix_close


def run_backtest(settings: Settings, candles: pd.DataFrame, vix_close: pd.Series, journal: Journal,
                 only: tuple[str, ...] | None = None) -> None:
    s = settings
    pricer = ModelPricer(s.data.risk_free_rate, s.execution.spread_pct, s.execution.tick_size)
    data = HistoricalData(candles, vix_close, s.strategy.candle_minutes, pricer)
    engine = Engine(
        settings=s,
        data=data,
        broker=PaperBroker(s.execution, CostModel(s.costs)),
        journal=journal,
        strategies=build_strategies(s.strategy, only),
        risk=RiskManager(s.risk, s.strategy),
        # The instrument file only lists future expiries, so past dates use the weekly rule.
        instruments=InstrumentResolver(s.instrument, master=None),
    )
    step = pd.Timedelta(minutes=s.strategy.candle_minutes)
    candles = normalise_candles(candles)
    # Yahoo's first day is often partial; it only serves as the 'previous day' for the next one.
    for day, day_candles in candles.groupby(candles.index.date):
        for start in day_candles.index:
            try:
                engine.step((start + step).to_pydatetime())
            except StaleDataError as e:
                log.debug("skip %s: %s", start, e)
        if engine.position:  # data ended before square-off time: close at the last candle
            last = (day_candles.index[-1] + step).to_pydatetime()
            engine.close_position(engine.position, data.quote(engine.position.contract, last), last, "END_OF_DATA")


def fresh_journal(path: Path) -> Journal:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    return Journal(path)
