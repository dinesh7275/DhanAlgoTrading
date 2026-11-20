"""Indian exchange-traded options charges, applied to every simulated fill."""

from __future__ import annotations

from .config import CostConfig
from .models import Charges, Side


class CostModel:
    def __init__(self, cfg: CostConfig):
        self.cfg = cfg

    def charges(self, side: Side, price: float, quantity: int) -> Charges:
        c = self.cfg
        turnover = price * quantity
        brokerage = c.brokerage_per_order
        exchange = turnover * c.exchange_txn_pct / 100
        sebi = turnover * c.sebi_per_crore / 1e7
        return Charges(
            brokerage=brokerage,
            stt=turnover * c.stt_sell_pct / 100 if side is Side.SELL else 0.0,
            exchange=exchange,
            sebi=sebi,
            stamp=turnover * c.stamp_buy_pct / 100 if side is Side.BUY else 0.0,
            gst=(brokerage + exchange + sebi) * c.gst_pct / 100,
        )
