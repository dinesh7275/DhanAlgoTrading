from datetime import date, time

import pytest

from algotrader.broker import PaperBroker
from algotrader.config import ConfigError, Settings, load_settings, validate
from algotrader.costs import CostModel
from algotrader.instruments import InstrumentResolver
from algotrader.models import OptionContract, OptionType, Quote, Side, is_market_open
from algotrader.pricing import black_scholes
from algotrader.risk import DayStats, RiskManager
from algotrader.strategy import OpeningRangeBreakout, PreviousDayBreakout, RangeReversal
from dataclasses import replace

import pandas as pd

from helpers import ist, make_day


def test_repo_config_loads():
    s = load_settings("config.toml")
    assert s.instrument.lots == 1
    assert s.risk.square_off_time == time(15, 15)


def test_more_than_one_lot_is_rejected():
    s = Settings()
    with pytest.raises(ConfigError):
        validate(replace(s, instrument=replace(s.instrument, lots=2)))


def test_unknown_strategy_is_rejected():
    s = Settings()
    with pytest.raises(ConfigError):
        validate(replace(s, strategy=replace(s.strategy, enabled=("orb", "magic"))))


def test_charges_buy_and_sell():
    c = CostModel(Settings().costs)
    buy = c.charges(Side.BUY, 100.0, 65)   # turnover 6,500
    sell = c.charges(Side.SELL, 100.0, 65)
    assert buy.stt == 0 and buy.stamp == pytest.approx(0.195)
    assert sell.stamp == 0 and sell.stt == pytest.approx(6.5)
    assert buy.exchange == pytest.approx(6500 * 0.0003503)
    assert buy.gst == pytest.approx((20 + buy.exchange + buy.sebi) * 0.18)


def test_put_call_parity():
    s, k, t, v, r = 25000, 25100, 7 / 365, 0.15, 0.065
    call = black_scholes(s, k, t, v, r, OptionType.CE)
    put = black_scholes(s, k, t, v, r, OptionType.PE)
    import math
    assert call - put == pytest.approx(s - k * math.exp(-r * t), abs=1e-6)


def test_paper_fills_cross_the_spread_with_slippage():
    s = Settings()
    broker = PaperBroker(s.execution, CostModel(s.costs))
    c = OptionContract("NIFTY", date(2026, 11, 24), 25500, OptionType.CE, 65)
    q = Quote(bid=100.0, ask=100.5, ltp=100.2, time=ist(2026, 11, 19, 10, 0))
    assert broker.buy(c, 65, q, q.time).price == 100.55
    assert broker.sell(c, 65, q, q.time).price == 99.95
    assert broker.estimate_buy_cost(c, 65, q) == pytest.approx(100.55 * 65 + broker.buy(c, 65, q, q.time).charges.total)


def test_fallback_expiry_skips_expiry_day():
    r = InstrumentResolver(Settings().instrument)
    assert r.expiry_for(date(2026, 11, 19)) == date(2026, 11, 24)   # Thursday -> next Tuesday
    assert r.expiry_for(date(2026, 11, 24)) == date(2026, 12, 1)    # on expiry Tuesday -> following week
    assert r.atm_strike(25524.0) == 25500 and r.atm_strike(25526.0) == 25550


def test_market_hours():
    assert is_market_open(ist(2026, 11, 19, 9, 15))
    assert not is_market_open(ist(2026, 11, 19, 15, 30))
    assert not is_market_open(ist(2026, 11, 21, 11, 0))  # Saturday
    assert not is_market_open(ist(2026, 11, 19, 11, 0), holidays=(date(2026, 11, 19),))


def test_risk_rules():
    s = Settings()
    risk = RiskManager(s.risk, s.strategy)
    now = ist(2026, 11, 19, 10, 0)
    assert risk.can_enter(now, DayStats(), False, 25000, 10000) == (True, "ok")
    assert not risk.can_enter(now, DayStats(), True, 25000, 10000)[0]
    assert not risk.can_enter(ist(2026, 11, 19, 14, 30), DayStats(), False, 25000, 10000)[0]
    assert not risk.can_enter(now, DayStats(trades=2), False, 25000, 10000)[0]
    assert not risk.can_enter(now, DayStats(net_pnl=-2500), False, 25000, 10000)[0]
    assert not risk.can_enter(now, DayStats(trades=2, consecutive_losses=2), False, 25000, 10000)[0]
    ok, why = risk.can_enter(now, DayStats(), False, 5000, 10000)
    assert not ok and "capital" in why


class TestOpeningRangeBreakout:
    day = ist(2026, 11, 19, 0, 0)

    def run(self, closes):
        strategy = OpeningRangeBreakout(Settings().strategy)
        candles = make_day(self.day, closes)
        signals = []
        for i in range(len(candles)):
            now = candles.index[i] + (candles.index[1] - candles.index[0])
            sig = strategy.on_candles(candles.iloc[: i + 1], now.to_pydatetime())
            if sig:
                signals.append((i, sig))
        return signals

    def test_no_signal_inside_range(self):
        assert self.run([25500, 25510, 25490, 25505, 25495, 25500]) == []

    def test_upside_breakout_buys_call_once(self):
        # opening range high = 25510 + 5 spread = 25515
        signals = self.run([25500, 25510, 25490, 25520, 25530, 25540])
        assert len(signals) == 1
        i, sig = signals[0]
        assert i == 3 and sig.option_type is OptionType.CE
        assert sig.features["range_high"] == 25515

    def test_downside_breakout_buys_put(self):
        signals = self.run([25500, 25510, 25490, 25470, 25460])
        assert [s.option_type for _, s in signals] == [OptionType.PE]

    def test_range_too_wide_is_skipped(self):
        assert self.run([25500, 25800, 25300, 25900, 26000]) == []

    def test_no_signal_from_stale_candle(self):
        strategy = OpeningRangeBreakout(Settings().strategy)
        candles = make_day(self.day, [25500, 25510, 25490, 25520])
        late = candles.index[-1].to_pydatetime().replace(hour=11)
        assert strategy.on_candles(candles, late) is None


def run_strategy(strategy, candles, day):
    """Feed candles one at a time, as the engine does; returns [(index within today, signal)]."""
    step = pd.Timedelta(minutes=5)
    first_today = int((candles.index.date < day.date()).sum())
    signals = []
    for i in range(first_today, len(candles)):
        sig = strategy.on_candles(candles.iloc[: i + 1], (candles.index[i] + step).to_pydatetime())
        if sig:
            signals.append((i - first_today, sig))
    return signals


class TestPreviousDayBreakout:
    prev, day = ist(2026, 11, 18, 0, 0), ist(2026, 11, 19, 0, 0)

    def run(self, today_closes):
        # previous day: high 25485, low 25415 (closes +/- 5 spread)
        candles = pd.concat([make_day(self.prev, [25450, 25480, 25420]), make_day(self.day, today_closes)])
        return run_strategy(PreviousDayBreakout(Settings().strategy), candles, self.day)

    def test_cross_above_previous_high_buys_call(self):
        signals = self.run([25460, 25470, 25475, 25480, 25495, 25500])
        assert [(i, s.option_type) for i, s in signals] == [(4, OptionType.CE)]
        assert signals[0][1].features["prev_high"] == 25485

    def test_cross_below_previous_low_buys_put(self):
        signals = self.run([25440, 25430, 25425, 25420, 25410, 25400])
        assert [(i, s.option_type) for i, s in signals] == [(4, OptionType.PE)]

    def test_gap_open_beyond_level_is_not_a_breakout(self):
        assert self.run([25520, 25530, 25540, 25550, 25560]) == []

    def test_needs_a_previous_day(self):
        candles = make_day(self.day, [25460, 25470, 25475, 25480, 25495, 25500])
        assert run_strategy(PreviousDayBreakout(Settings().strategy), candles, self.day) == []


class TestRangeReversal:
    day = ist(2026, 11, 19, 0, 0)

    def run(self, closes):
        return run_strategy(RangeReversal(Settings().strategy), make_day(self.day, closes), self.day)

    def test_rejections_at_range_edges(self):
        # 25485-25535 range (0.2% wide): falls from the top are put signals, bounces off the bottom
        # are call signals; each direction fires once a day
        signals = self.run([25490, 25530] * 10)
        assert [(i, s.option_type) for i, s in signals] == [(12, OptionType.PE), (13, OptionType.CE)]
        assert signals[0][1].time.time() == time(10, 20)

    def test_trending_day_is_ignored(self):
        assert self.run([25500 + 20 * i for i in range(30)]) == []

    def test_nothing_before_earliest_time(self):
        assert self.run([25490, 25530] * 6) == []
