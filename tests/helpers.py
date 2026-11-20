from datetime import datetime, timedelta

import pandas as pd

from algotrader.models import IST


def ist(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=IST)


def make_day(day: datetime, closes: list[float], spread: float = 5.0) -> pd.DataFrame:
    """5-minute candles from 09:15 with the given closes; each candle's open is the previous close."""
    rows, prev = [], closes[0]
    for i, close in enumerate(closes):
        o = prev
        rows.append({"open": o, "high": max(o, close) + spread, "low": min(o, close) - spread, "close": close})
        prev = close
    index = [day.replace(hour=9, minute=15) + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))
