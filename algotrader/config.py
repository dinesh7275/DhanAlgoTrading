"""Typed settings loaded from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AccountConfig:
    starting_capital: float = 25000


@dataclass(frozen=True)
class InstrumentConfig:
    underlying: str = "NIFTY"
    lots: int = 1
    strike_step: int = 50
    fallback_lot_size: int = 65
    fallback_expiry_weekday: int = 1
    avoid_expiry_day: bool = True
    use_instrument_master: bool = True


@dataclass(frozen=True)
class DataConfig:
    provider: str = "yahoo"
    poll_seconds: int = 30
    max_staleness_minutes: int = 10
    risk_free_rate: float = 6.5
    holidays: tuple[date, ...] = ()


@dataclass(frozen=True)
class OrbConfig:
    range_minutes: int = 15
    min_range_pct: float = 0.10
    max_range_pct: float = 0.80
    stop_loss_pct: float = 20
    target_pct: float = 40
    max_hold_minutes: int = 0


@dataclass(frozen=True)
class PrevDayConfig:
    start_after_minutes: int = 15
    stop_loss_pct: float = 20
    target_pct: float = 40
    max_hold_minutes: int = 0


@dataclass(frozen=True)
class ReversalConfig:
    earliest: time = time(10, 15)
    lookback_candles: int = 12
    min_range_pct: float = 0.15
    max_range_pct: float = 0.40
    stop_loss_pct: float = 10
    target_pct: float = 15
    max_hold_minutes: int = 45


STRATEGY_NAMES = ("orb", "prev_day", "reversal")


@dataclass(frozen=True)
class StrategyConfig:
    enabled: tuple[str, ...] = ("orb",)
    candle_minutes: int = 5
    no_entry_after: time = time(14, 30)
    orb: OrbConfig = field(default_factory=OrbConfig)
    prev_day: PrevDayConfig = field(default_factory=PrevDayConfig)
    reversal: ReversalConfig = field(default_factory=ReversalConfig)


@dataclass(frozen=True)
class RiskConfig:
    max_trades_per_day: int = 2
    max_daily_loss: float = 2500
    max_consecutive_losses: int = 2
    square_off_time: time = time(15, 15)


@dataclass(frozen=True)
class ExecutionConfig:
    spread_pct: float = 0.6
    slippage_ticks: int = 1
    tick_size: float = 0.05


@dataclass(frozen=True)
class CostConfig:
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.1
    exchange_txn_pct: float = 0.03503
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.003
    gst_pct: float = 18.0


@dataclass(frozen=True)
class GoLiveConfig:
    min_trades: int = 50
    min_trading_days: int = 30
    min_profit_factor: float = 1.3
    max_drawdown_pct: float = 15


@dataclass(frozen=True)
class Settings:
    account: AccountConfig = field(default_factory=AccountConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    go_live: GoLiveConfig = field(default_factory=GoLiveConfig)


class ConfigError(ValueError):
    pass


def _coerce(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys in [{cls.__name__}]: {', '.join(sorted(unknown))}")
    values = {}
    for name, value in raw.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            value = _coerce(type(default), value)
        elif isinstance(default, tuple) and name != "holidays":
            value = tuple(value)
        elif isinstance(default, time):
            value = time.fromisoformat(value)
        elif name == "holidays":
            value = tuple(date.fromisoformat(v) for v in value)
        values[name] = value
    return cls(**values)


def load_settings(path: str | Path = "config.toml") -> Settings:
    path = Path(path)
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    sections = {f.name: f.default_factory for f in fields(Settings)}  # type: ignore[misc]
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown))}")
    settings = Settings(**{
        name: _coerce(factory().__class__, raw.get(name, {})) for name, factory in sections.items()
    })
    validate(settings)
    return settings


def validate(s: Settings) -> None:
    if s.instrument.lots != 1:
        raise ConfigError("instrument.lots must be 1: this app trades a single lot only")
    if s.data.provider not in ("yahoo", "dhan"):
        raise ConfigError("data.provider must be 'yahoo' or 'dhan'")
    st = s.strategy
    if not st.enabled:
        raise ConfigError("strategy.enabled must list at least one strategy")
    unknown = set(st.enabled) - set(STRATEGY_NAMES)
    if unknown:
        raise ConfigError(f"unknown strategies {sorted(unknown)}; choose from {list(STRATEGY_NAMES)}")
    if st.orb.range_minutes % st.candle_minutes:
        raise ConfigError("strategy.orb.range_minutes must be a multiple of candle_minutes")
    for name in STRATEGY_NAMES:
        sub = getattr(st, name)
        if not 0 < sub.stop_loss_pct < 100:
            raise ConfigError(f"strategy.{name}.stop_loss_pct must be between 0 and 100")
        if sub.target_pct <= 0:
            raise ConfigError(f"strategy.{name}.target_pct must be positive")
        if sub.max_hold_minutes < 0:
            raise ConfigError(f"strategy.{name}.max_hold_minutes must be 0 (off) or positive")
    if s.strategy.no_entry_after >= s.risk.square_off_time:
        raise ConfigError("strategy.no_entry_after must be earlier than risk.square_off_time")
