"""Black-Scholes premiums, used when real option quotes aren't available (Yahoo data and backtests)."""

from __future__ import annotations

import math
from datetime import datetime

from .models import MARKET_CLOSE, OptionContract, OptionType, Quote

SECONDS_PER_YEAR = 365 * 24 * 3600


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes(spot: float, strike: float, years: float, vol: float, rate: float,
                  option_type: OptionType) -> float:
    if years <= 0 or vol <= 0:
        return max(spot - strike, 0.0) if option_type is OptionType.CE else max(strike - spot, 0.0)
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-rate * years)
    if option_type is OptionType.CE:
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def years_to_expiry(now: datetime, contract: OptionContract) -> float:
    expiry_at = datetime.combine(contract.expiry, MARKET_CLOSE, tzinfo=now.tzinfo)
    return max((expiry_at - now).total_seconds(), 0.0) / SECONDS_PER_YEAR


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 2)


class ModelPricer:
    """Turns a spot price and volatility into a bid/ask quote with a simulated spread."""

    def __init__(self, rate_pct: float, spread_pct: float, tick: float):
        self.rate = rate_pct / 100
        self.spread_pct = spread_pct
        self.tick = tick

    def premium(self, contract: OptionContract, spot: float, vol: float, now: datetime) -> float:
        return black_scholes(spot, contract.strike, years_to_expiry(now, contract), vol, self.rate,
                             contract.option_type)

    def quote(self, contract: OptionContract, spot: float, vol: float, now: datetime) -> Quote:
        mid = max(self.premium(contract, spot, vol, now), self.tick)
        half = max(mid * self.spread_pct / 200, self.tick / 2)
        bid = max(round_to_tick(mid - half, self.tick), self.tick)
        ask = max(round_to_tick(mid + half, self.tick), bid + self.tick)
        return Quote(bid=bid, ask=ask, ltp=round_to_tick(mid, self.tick), time=now, iv=vol)
