"""Command line: python -m algotrader {paper,backtest,report,dashboard}."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .analytics import format_comparison, format_report, summarize
from .config import STRATEGY_NAMES, ConfigError, Settings, load_settings
from .models import IST

DATA_DIR = Path("data")
PAPER_DB = DATA_DIR / "paper.db"
BACKTEST_DB = DATA_DIR / "backtest.db"

log = logging.getLogger("algotrader")


def setup_logging(name: str, verbose: bool) -> None:
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(DATA_DIR / "logs" / f"{name}_{datetime.now(IST):%Y%m%d}.log")],
    )
    for noisy in ("yfinance", "urllib3", "peewee", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_paper_engine(settings: Settings):
    from .broker import PaperBroker
    from .costs import CostModel
    from .data import DhanData, YahooData
    from .engine import Engine
    from .instruments import InstrumentMaster, InstrumentResolver
    from .journal import Journal
    from .pricing import ModelPricer
    from .risk import RiskManager
    from .strategy import build_strategies

    s = settings
    master = None
    if s.instrument.use_instrument_master:
        try:
            master = InstrumentMaster.load(DATA_DIR / "cache", datetime.now(IST).date(), s.instrument.underlying)
        except Exception as e:
            log.warning("Instrument file unavailable (%s); using fallback lot size and expiry rule.", e)
    instruments = InstrumentResolver(s.instrument, master)
    log.info("%s lot size %d, trading expiry %s (%s)", s.instrument.underlying, instruments.lot_size,
             instruments.expiry_for(datetime.now(IST).date()),
             "from Dhan instrument file" if master else "fallback rule")

    if s.data.provider == "dhan":
        from dotenv import load_dotenv

        load_dotenv()
        client_id, token = os.getenv("DHAN_CLIENT_ID"), os.getenv("DHAN_ACCESS_TOKEN")
        if not client_id or not token:
            raise ConfigError("data.provider is 'dhan' but DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN are not set in .env")
        data = DhanData(s.data, s.strategy.candle_minutes, client_id, token)
    else:
        pricer = ModelPricer(s.data.risk_free_rate, s.execution.spread_pct, s.execution.tick_size)
        data = YahooData(s.data, s.strategy.candle_minutes, pricer)

    return Engine(s, data, PaperBroker(s.execution, CostModel(s.costs)), Journal(PAPER_DB),
                  build_strategies(s.strategy), RiskManager(s.risk, s.strategy), instruments)


def cmd_paper(args, settings: Settings) -> None:
    engine = build_paper_engine(settings)
    try:
        engine.run_forever()
    except KeyboardInterrupt:
        log.info("Stopped. Open position (if any) is saved and will be resumed on next start.")


def cmd_backtest(args, settings: Settings) -> None:
    from .backtest import download_history, fresh_journal, run_backtest

    from .journal import Journal

    only = tuple(args.strategy) if args.strategy else None
    log.info("Downloading %d days of NIFTY %dm candles from Yahoo...", args.days, settings.strategy.candle_minutes)
    candles, vix = download_history(args.days, settings.strategy.candle_minutes)
    first, last = candles.index[0].date(), candles.index[-1].date()
    if args.compare:
        names = only or STRATEGY_NAMES
        rows = []
        for label, run in [(n, (n,)) for n in names] + [("all together", names)]:
            journal = Journal(":memory:")
            run_backtest(settings, candles, vix, journal, run)
            rows.append((label, summarize(journal.closed_trades(), settings)))
        print()
        print(format_comparison(rows, f"Strategy comparison {first} to {last}"))
        print("\nPremiums in backtests are modelled (Black-Scholes + India VIX), not real quotes.")
        return
    journal = fresh_journal(BACKTEST_DB)
    run_backtest(settings, candles, vix, journal, only)
    print()
    print(format_report(summarize(journal.closed_trades(), settings),
                        f"Backtest {first} to {last} ({', '.join(only or settings.strategy.enabled)})"))
    print("\nPremiums in backtests are modelled (Black-Scholes + India VIX), not real quotes."
          f"\nTrades saved to {BACKTEST_DB}; view with: python -m algotrader dashboard --backtest")


def cmd_report(args, settings: Settings) -> None:
    from .journal import Journal

    path = BACKTEST_DB if args.backtest else PAPER_DB
    if not path.exists():
        sys.exit(f"No journal at {path} yet.")
    journal = Journal(path)
    print(format_report(summarize(journal.closed_trades(), settings),
                        "Backtest report" if args.backtest else "Paper trading report"))


def cmd_dashboard(args, settings: Settings) -> None:
    from .dashboard import create_app
    from .journal import Journal

    path = BACKTEST_DB if args.backtest else PAPER_DB
    Journal(path).close()  # create an empty journal if none exists yet
    app = create_app(path, settings, "Backtest" if args.backtest else "Paper trading")
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="algotrader", description="NIFTY options paper-trading system")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("paper", help="run live paper trading during market hours")
    bt = sub.add_parser("backtest", help="replay recent history through the same engine")
    bt.add_argument("--days", type=int, default=59, help="days of history (Yahoo allows up to 59)")
    bt.add_argument("--strategy", action="append", choices=STRATEGY_NAMES,
                    help="run only this strategy (repeatable); default: strategy.enabled in config")
    bt.add_argument("--compare", action="store_true",
                    help="backtest each strategy alone and all together, and print a comparison")
    rp = sub.add_parser("report", help="performance report and go-live checklist")
    rp.add_argument("--backtest", action="store_true", help="report on the last backtest instead")
    db = sub.add_parser("dashboard", help="read-only web dashboard")
    db.add_argument("--backtest", action="store_true", help="show the last backtest instead")
    db.add_argument("--port", type=int, default=5050)
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        sys.exit(f"Config error: {e}")
    setup_logging(args.command, args.verbose)
    try:
        {"paper": cmd_paper, "backtest": cmd_backtest, "report": cmd_report,
         "dashboard": cmd_dashboard}[args.command](args, settings)
    except ConfigError as e:
        sys.exit(f"Config error: {e}")


if __name__ == "__main__":
    main()
