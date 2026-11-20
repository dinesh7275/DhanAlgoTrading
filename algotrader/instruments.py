"""Which option contract to trade: lot size, expiry, ATM strike and Dhan security ID.

Lot size and expiry dates come from Dhan's public instrument file (no login needed), so exchange
changes (lot size revisions, expiry-day moves, holiday shifts) are picked up automatically. The
security ID it provides is what a future live broker would need to place an order.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import InstrumentConfig
from .models import OptionContract, OptionType

log = logging.getLogger(__name__)

MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_COLUMNS = ["SEM_EXM_EXCH_ID", "SEM_INSTRUMENT_NAME", "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL",
            "SEM_LOT_UNITS", "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE"]


class InstrumentMaster:
    def __init__(self, options: pd.DataFrame):
        self._df = options

    @classmethod
    def load(cls, cache_dir: Path, today: date, underlying: str) -> "InstrumentMaster":
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"scrip_master_{today:%Y%m%d}.csv"
        if not path.exists():
            log.info("Downloading Dhan instrument file...")
            tmp = path.with_suffix(".part")
            with requests.get(MASTER_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(path)
            for old in cache_dir.glob("scrip_master_*.csv"):
                if old != path:
                    old.unlink()
        df = pd.read_csv(path, usecols=_COLUMNS, low_memory=False)
        df = df[(df.SEM_EXM_EXCH_ID == "NSE") & (df.SEM_INSTRUMENT_NAME == "OPTIDX")
                & df.SEM_TRADING_SYMBOL.str.startswith(f"{underlying}-")].copy()
        if df.empty:
            raise ValueError(f"no {underlying} index options in the instrument file")
        df["expiry"] = pd.to_datetime(df.SEM_EXPIRY_DATE).dt.date
        return cls(df)

    def expiries(self) -> list[date]:
        return sorted(set(self._df.expiry))

    def lot_size(self) -> int:
        return int(self._df.SEM_LOT_UNITS.mode().iloc[0])

    def security_id(self, expiry: date, strike: float, option_type: OptionType) -> str | None:
        df = self._df
        row = df[(df.expiry == expiry) & (df.SEM_STRIKE_PRICE == strike)
                 & (df.SEM_OPTION_TYPE == option_type.value)]
        return str(row.SEM_SMST_SECURITY_ID.iloc[0]) if len(row) else None


class InstrumentResolver:
    def __init__(self, cfg: InstrumentConfig, master: InstrumentMaster | None = None):
        self.cfg = cfg
        self.master = master

    @property
    def lot_size(self) -> int:
        return self.master.lot_size() if self.master else self.cfg.fallback_lot_size

    def expiry_for(self, today: date) -> date:
        if self.master:
            candidates = self.master.expiries()
        else:
            ahead = (self.cfg.fallback_expiry_weekday - today.weekday()) % 7
            first = today + timedelta(days=ahead)
            candidates = [first + timedelta(weeks=i) for i in range(3)]
        valid = [e for e in candidates
                 if e > today or (e == today and not self.cfg.avoid_expiry_day)]
        if not valid:
            raise ValueError(f"no usable expiry on or after {today}")
        return min(valid)

    def atm_strike(self, spot: float) -> float:
        step = self.cfg.strike_step
        return float(round(spot / step) * step)

    def contract(self, spot: float, option_type: OptionType, today: date) -> OptionContract:
        expiry = self.expiry_for(today)
        strike = self.atm_strike(spot)
        security_id = self.master.security_id(expiry, strike, option_type) if self.master else None
        return OptionContract(self.cfg.underlying, expiry, strike, option_type, self.lot_size, security_id)
