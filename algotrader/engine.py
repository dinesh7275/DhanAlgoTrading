"""The trading loop. `step(now)` runs one tick; the same code drives live paper trading and backtests."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from .broker import Broker
from .config import Settings
from .data import MarketData, StaleDataError
from .instruments import InstrumentResolver
from .journal import Journal
from .models import IST, Position, Quote, Signal, is_market_open
from .risk import RiskManager
from .strategy import Strategy

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Settings, data: MarketData, broker: Broker, journal: Journal,
                 strategies: list[Strategy], risk: RiskManager, instruments: InstrumentResolver):
        self.s = settings
        self.data = data
        self.broker = broker
        self.journal = journal
        self.strategies = {st.name: st for st in strategies}
        self.risk = risk
        self.instruments = instruments
        self.position: Position | None = journal.open_position()
        if self.position:
            log.info("Resumed open paper position: %s x%d @ %.2f", self.position.contract.symbol,
                     self.position.quantity, self.position.entry_price)

    def cash(self) -> float:
        return self.s.account.starting_capital + self.journal.realized_net()

    # --- one tick -----------------------------------------------------------------------------

    def step(self, now: datetime) -> None:
        if not is_market_open(now, self.s.data.holidays):
            return
        if self.position:
            self._manage_position(now)
        candles = self.data.candles(now)
        for strategy in self.strategies.values():
            # Signals are recorded even while a trade is open, so the journal shows what was missed.
            signal = strategy.on_candles(candles, now)
            if signal is not None:
                self._consider(signal, strategy, now)

    def _consider(self, signal: Signal, strategy: Strategy, now: datetime) -> None:
        spot = self.data.spot(now)
        contract = self.instruments.contract(spot, signal.option_type, now.date())
        quote = self.data.quote(contract, now)
        quantity = contract.lot_size * self.s.instrument.lots
        required = self.broker.estimate_buy_cost(contract, quantity, quote)

        ok, decision = self.risk.can_enter(now, self.journal.day_stats(now.date()), self.position is not None,
                                           self.cash(), required)
        features = {**signal.features, **self._context(spot, contract, quote, quantity, required)}
        signal_id = self.journal.record_signal(signal, features, ok, decision)
        if not ok:
            log.info("Signal skipped (%s): %s", decision, signal.reason)
            return

        fill = self.broker.buy(contract, quantity, quote, now)
        stop = fill.price * (1 - strategy.stop_loss_pct / 100)
        target = fill.price * (1 + strategy.target_pct / 100)
        self.position = self.journal.open_trade(signal_id, signal.strategy, fill, stop, target)
        log.info("PAPER BUY %s x%d @ %.2f | stop %.2f target %.2f | %s", contract.symbol, quantity,
                 fill.price, stop, target, signal.reason)

    @staticmethod
    def _context(spot, contract, quote: Quote, quantity: int, required: float) -> dict:
        return {
            "spot": round(spot, 2),
            "symbol": contract.symbol,
            "expiry": contract.expiry.isoformat(),
            "strike": contract.strike,
            "lot_size": contract.lot_size,
            "quantity": quantity,
            "bid": quote.bid,
            "ask": quote.ask,
            "iv": round(quote.iv, 4) if quote.iv else None,
            "capital_required": round(required, 2),
        }

    def _manage_position(self, now: datetime) -> None:
        pos = self.position
        assert pos is not None
        try:
            quote = self.data.quote(pos.contract, now)
        except StaleDataError:
            if now.date() > pos.entry_time.astimezone(IST).date():
                # Engine was down at square-off and the contract can't be priced: close at last mark.
                self.close_position(pos, Quote(pos.last_price, pos.last_price, pos.last_price, now), now,
                           "MISSED_SQUARE_OFF")
                return
            raise

        low, high = self.data.premium_range(pos.contract, now) or (quote.mid, quote.mid)
        low, high = min(low, quote.mid), max(high, quote.mid)
        pos.max_price = max(pos.max_price, high)
        pos.min_price = min(pos.min_price, low)
        pos.last_price = quote.mid

        open_pnl = (quote.bid - pos.entry_price) * pos.quantity - pos.entry_charges
        day = self.journal.day_stats(now.date())

        if now.date() > pos.entry_time.astimezone(IST).date():
            self.close_position(pos, quote, now, "MISSED_SQUARE_OFF")
        elif low <= pos.stop_price:
            # If the price is already through the stop, fill at the real quote (gap); otherwise at the stop.
            self.close_position(pos, quote if quote.mid <= pos.stop_price else quote.moved_to(pos.stop_price), now,
                       "STOP_LOSS")
        elif high >= pos.target_price:
            self.close_position(pos, quote if quote.mid >= pos.target_price else quote.moved_to(pos.target_price), now,
                       "TARGET")
        elif self._held_too_long(pos, now):
            self.close_position(pos, quote, now, "TIME_LIMIT")
        elif self.risk.daily_loss_breached(day, open_pnl):
            self.close_position(pos, quote, now, "DAILY_LOSS_LIMIT")
        elif self.risk.must_square_off(now):
            self.close_position(pos, quote, now, "SQUARE_OFF")
        else:
            self.journal.update_mark(pos, now)

    def _held_too_long(self, pos: Position, now: datetime) -> bool:
        strategy = self.strategies.get(pos.strategy)
        limit = strategy.max_hold_minutes if strategy else 0
        return bool(limit) and now - pos.entry_time >= timedelta(minutes=limit)

    def close_position(self, pos: Position, quote: Quote, now: datetime, reason: str) -> None:
        fill = self.broker.sell(pos.contract, pos.quantity, quote, now)
        net = self.journal.close_trade(pos, fill, reason)
        self.position = None
        log.info("PAPER SELL %s x%d @ %.2f | %s | net P&L ₹%.2f", pos.contract.symbol, pos.quantity,
                 fill.price, reason, net)

    # --- live loop ----------------------------------------------------------------------------

    def run_forever(self) -> None:
        log.info("Paper trading started. Strategies: %s. Data: %s. Press Ctrl+C to stop.",
                 ", ".join(self.strategies), self.data.label)
        was_open = None
        while True:
            now = datetime.now(IST)
            market_open = is_market_open(now, self.s.data.holidays)
            if market_open != was_open:
                log.info("Market is %s.", "open" if market_open else "closed - waiting")
                was_open = market_open
            if market_open:
                try:
                    self.step(now)
                except StaleDataError as e:
                    log.warning("Skipping tick: %s", e)
                    self.journal.log_event(now, "WARNING", str(e))
                except Exception as e:  # keep running; the next tick may succeed
                    log.exception("Tick failed")
                    self.journal.log_event(now, "ERROR", f"{type(e).__name__}: {e}")
            time.sleep(self.s.data.poll_seconds if market_open else 60)
