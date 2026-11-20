# NIFTY Options Paper Trader

A paper-trading system for NIFTY index options. It trades **one lot**, on **paper only**:
nothing in this codebase can place a real order.

The point is to learn whether a strategy actually makes money after costs, using
the same engine you would later run with real money. Every signal, trade and skipped trade is
recorded with its reason. A go-live checklist tells you when (if ever) the results justify risking
real capital.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Commands

```bash
.venv/bin/python -m algotrader backtest            # replay the last ~59 days through the engine
.venv/bin/python -m algotrader backtest --compare  # each strategy alone and all together
.venv/bin/python -m algotrader backtest --strategy prev_day   # one strategy only
.venv/bin/python -m algotrader paper               # paper trade live, 09:15-15:30 IST (Ctrl+C to stop)
.venv/bin/python -m algotrader report              # stats + go-live checklist for paper trades
.venv/bin/python -m algotrader dashboard           # read-only dashboard at http://127.0.0.1:5050
.venv/bin/python -m algotrader report --backtest   # same, for the last backtest
.venv/bin/python -m algotrader dashboard --backtest
```

Leave `paper` running through the trading day (run the dashboard in a second terminal). If it
stops with a position open, the position is saved and picked up on the next start.

## How a trading day works

1. **Instruments.** At startup the app downloads Dhan's public instrument file to get the current
   NIFTY lot size, the expiry dates (including holiday shifts) and each contract's security ID.
   On expiry day it trades the next expiry instead.
2. **Strategies** (pick them with `strategy.enabled`; each direction fires at most once a day):
   - `orb`, opening range breakout: the first 5-minute close above the 09:15-09:30 high buys the
     ATM call, the first close below the low buys the ATM put. Days with an unusually narrow or
     wide opening range are skipped.
   - `prev_day`, previous day breakout: after 09:30, a 5-minute close that crosses above
     yesterday's high buys the ATM call, below yesterday's low the ATM put. A gap open beyond
     the level doesn't count; price has to cross it during the day.
   - `reversal`, range reversal (off by default): from 10:15, if the last hour was a tight
     sideways range, a red candle rejected at the top buys the ATM put and a green candle
     rejected at the bottom buys the ATM call, with a smaller stop/target and a 45-minute limit.
     It lost money in backtests (profit factor about 0.6) and stays off until a better version
     proves itself.

   When several strategies are on, they share one position slot and the risk limits below.
   Signals that arrive while a trade is open are still recorded, marked as skipped.
3. **Risk checks**, all of which must pass before entry: one position at a time, max 2 trades a day,
   a daily loss limit, stop after 2 losses in a row, no entries after 14:30, and enough paper
   capital for the lot plus charges.
4. **Exits.** Each strategy's stop and target on the option premium (-20%/+40% for the breakouts),
   its time limit if it has one, the daily loss limit, or square-off at 15:15.
5. **Fills and costs.** Buys fill at the ask and sells at the bid, plus one tick of slippage.
   Brokerage, STT, exchange charges, SEBI fees, stamp duty and GST are charged on every fill.

All of these numbers live in `config.toml`.

## Data: what's real and what's modelled

| `data.provider` | NIFTY price | Option premiums |
|---|---|---|
| `yahoo` (default) | Yahoo, live | **Modelled** with Black-Scholes, using India VIX as volatility |
| `dhan` | Dhan, live | **Real** bid/ask from Dhan's option chain (needs `.env` + Dhan Data API subscription) |

Backtests always use modelled premiums, because free historical option quotes aren't available.
Modelled premiums ignore volatility smile, intraday IV changes and real spreads, so treat
backtest and `yahoo` paper results as approximations. Switch to `dhan` as soon as you can: paper
results on real quotes are the ones worth trusting.

## The learning loop

- `data/paper.db` (SQLite) records each signal with a snapshot of the market (spot, range,
  strike, bid/ask, IV, capital required) and whether risk rules let it through. For each trade it
  records entry, exit, reason, charges, and the best and worst premium seen while the trade was
  open.
- `report` breaks results down by exit reason, call/put and weekday, and shows how much of the
  gross P&L went to charges. This is where you see what's working and what isn't.
- Change one thing in `config.toml`, backtest it, then paper trade it for weeks. Change one
  variable at a time so you know what made the difference.

### Go-live checklist

`report` checks paper results against `[go_live]` in `config.toml`: at least 50 trades over 30
trading days, positive expectancy after charges, profit factor ≥ 1.3, and max drawdown ≤ 15%.
Passing these is necessary but not sufficient. Also confirm the results came from real Dhan
quotes, and that paper results roughly match the backtest.

## Path to real money (not built yet, on purpose)

Order execution goes through the `Broker` interface in `algotrader/broker.py`. Only
`PaperBroker` exists. Going live would mean writing a separate broker class that places Dhan
orders using the security IDs the app already resolves, and handles rejections, partial fills
and reconciliation with the broker's positions. Build and review that only after the checklist
passes on real quotes.

## Layout

```
algotrader/
  config.py       typed settings from config.toml (refuses anything but 1 lot)
  models.py       contracts, quotes, signals, fills, positions, market hours
  instruments.py  lot size, expiry, ATM strike, security IDs (Dhan instrument file)
  data.py         Yahoo / Dhan (read-only) / historical market data
  pricing.py      Black-Scholes and modelled quotes
  strategy.py     the strategies (orb, prev_day, reversal)
  risk.py         entry and in-trade risk rules
  broker.py       Broker interface + PaperBroker
  costs.py        Indian options charges
  engine.py       the trading loop (shared by paper and backtest)
  journal.py      SQLite record of signals, trades and warnings
  analytics.py    stats and go-live checklist
  backtest.py     historical replay
  dashboard.py    read-only web dashboard
tools/compound_calculator.html   standalone compounding calculator
```
