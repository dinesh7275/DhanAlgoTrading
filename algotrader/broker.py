"""Order execution.

`Broker` is the seam between the trading logic and wherever orders actually go. Only
`PaperBroker` exists: it fills simulated orders against the current quote and charges realistic
costs. There is deliberately no live broker in this codebase. Going live later means writing a
separate `Broker` implementation and passing it to the engine, after the paper results clear the
go-live checks in `algotrader report`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from .config import ExecutionConfig
from .costs import CostModel
from .models import Fill, OptionContract, Quote, Side
from .pricing import round_to_tick


class Broker(ABC):
    @abstractmethod
    def estimate_buy_cost(self, contract: OptionContract, quantity: int, quote: Quote) -> float:
        """Cash needed to buy, including charges. Must not place anything."""

    @abstractmethod
    def buy(self, contract: OptionContract, quantity: int, quote: Quote, now: datetime) -> Fill: ...

    @abstractmethod
    def sell(self, contract: OptionContract, quantity: int, quote: Quote, now: datetime) -> Fill: ...


class PaperBroker(Broker):
    """Buys at the ask and sells at the bid, plus slippage ticks: the same prices a market order would get."""

    def __init__(self, cfg: ExecutionConfig, costs: CostModel):
        self.cfg = cfg
        self.costs = costs

    def _slip(self) -> float:
        return self.cfg.slippage_ticks * self.cfg.tick_size

    def estimate_buy_cost(self, contract: OptionContract, quantity: int, quote: Quote) -> float:
        price = round_to_tick(quote.ask + self._slip(), self.cfg.tick_size)
        return price * quantity + self.costs.charges(Side.BUY, price, quantity).total

    def buy(self, contract: OptionContract, quantity: int, quote: Quote, now: datetime) -> Fill:
        price = round_to_tick(quote.ask + self._slip(), self.cfg.tick_size)
        return Fill(contract, Side.BUY, quantity, price, now, self.costs.charges(Side.BUY, price, quantity))

    def sell(self, contract: OptionContract, quantity: int, quote: Quote, now: datetime) -> Fill:
        price = max(round_to_tick(quote.bid - self._slip(), self.cfg.tick_size), self.cfg.tick_size)
        return Fill(contract, Side.SELL, quantity, price, now, self.costs.charges(Side.SELL, price, quantity))
