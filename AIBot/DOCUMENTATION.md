# AIBot - Complete Code Documentation

## 📖 Overview

This document provides comprehensive documentation for every Python file in the AIBot system, explaining methods, data sources, and functionality.

---

## 🏗️ Core Application Files

### 1. `main.py` - Main Application Entry Point

**Purpose**: Primary AIBot application with integrated web interface running on localhost:8080.

#### Classes & Methods:

##### `AIBot` Class
```python
class AIBot:
    def __init__(self):
```
- **Purpose**: Initialize AIBot with AI models and configuration
- **Data Sources**: None (initialization only)
- **Returns**: AIBot instance with loaded models

```python
def setup_logging(self):
```
- **Purpose**: Configure logging system for file and console output
- **Data Sources**: `LOGGING_CONFIG` from config.py
- **Creates**: Log files in `logs/aibot.log`

```python
def start(self):
```
- **Purpose**: Main bot execution loop
- **Data Sources**: Real-time market data via `analyze_market()`
- **Process**: Continuous market analysis → decision making → trade execution
- **Frequency**: Every 5 minutes (configurable via `DATA_CONFIG['update_frequency']`)

```python
def analyze_market(self):
```
- **Purpose**: Comprehensive market analysis using all AI models
- **Data Sources**:
  - **Yahoo Finance**: Via IV predictor for NIFTY data
  - **VIX Data**: For volatility analysis
  - **Options Chain**: For anomaly detection
  - **Technical Indicators**: From price predictor
- **Returns**: Dictionary with IV predictions, price forecasts, anomalies, risk metrics

```python
def make_trading_decision(self, analysis):
```
- **Purpose**: Generate trading decisions based on AI analysis
- **Input**: Market analysis from `analyze_market()`
- **Logic**: 
  - Risk checks first (daily loss limits)
  - High confidence price predictions (>70% confidence)
  - Options anomaly opportunities (>5% expected profit)
- **Returns**: Trading decision with action, confidence, and reasoning

```python
def execute_trade(self, decision):
```
- **Purpose**: Execute trading decisions
- **Current State**: Simulation mode (no real trades)
- **Future Integration**: Dhan API for live trading
- **Logging**: All trades logged for analysis

```python
def stop(self):
```
- **Purpose**: Graceful shutdown and data saving
- **Actions**: Stop main loop, save session data, cleanup resources

#### Web Interface Routes:

```python
@app.route('/')
def dashboard():
```
- **Purpose**: Main web dashboard
- **Returns**: HTML interface with real-time bot status

```python
@app.route('/api/status')
def api_status():
```
- **Purpose**: Get current bot status via REST API
- **Data Sources**: Live bot instance data
- **Returns**: JSON with running status, capital, P&L, AI signals

```python
@app.route('/api/start', methods=['POST'])
def api_start():
```
- **Purpose**: Start bot via web interface
- **Process**: Initialize bot in background thread
- **Safety**: Checks if already running

```python
@app.route('/api/stop', methods=['POST'])
def api_stop():
```
- **Purpose**: Stop bot via web interface
- **Process**: Graceful shutdown of bot operations

---

### 2. `web_server.py` - Standalone Web Server

**Purpose**: Lightweight web server with Flask fallback support.

#### Key Methods:

```python
def main():
```
- **Purpose**: Initialize and start web server
- **Port**: 8080 (localhost)
- **Fallback**: Simple HTTP server if Flask unavailable

```python
def start_simple_server():
```
- **Purpose**: Fallback HTTP server using Python's built-in `HTTPServer`
- **Use Case**: When Flask is not installed
- **Features**: Basic dashboard and API endpoints

```python
class SimpleAIBotHandler(BaseHTTPRequestHandler):
```
- **Purpose**: Handle HTTP requests in fallback mode
- **Methods**: GET (dashboard, status), POST (start/stop commands)

---

### 3. `run_aibot.py` - Startup Script with Dependency Checking

**Purpose**: Safe startup script that validates dependencies before running AIBot.

#### Methods:

```python
def check_dependencies():
```
- **Purpose**: Validate required Python packages
- **Checks**: numpy, pandas, scikit-learn, yfinance, scipy, matplotlib, joblib
- **Returns**: Boolean indicating if all dependencies available

```python
def main():
```
- **Purpose**: Main startup sequence
- **Process**: 
  1. Check dependencies
  2. Change to AIBot directory
  3. Import and run main AIBot
- **Error Handling**: Graceful failure with helpful messages

---

## 🤖 AI Models Directory (`models/`)

### IV Prediction Models (`models/iv_prediction/`)

#### `live_iv_predictor.py` - Live Implied Volatility Prediction

```python
class LiveIVPredictor:
    def __init__(self):
```
- **Purpose**: Initialize IV prediction system
- **Models**: LSTM for volatility forecasting
- **Data Sources**: VIX, NIFTY returns, volume data

```python
def predict_iv(self, symbol):
```
- **Purpose**: Predict future implied volatility
- **Data Sources**:
  - **Yahoo Finance**: Historical VIX data via `yfinance`
  - **NSE Data**: NIFTY returns and volume
  - **Options Data**: Historical IV levels
- **Returns**: Predicted IV with confidence score

```python
def get_market_features(self, symbol):
```
- **Purpose**: Extract features for IV prediction
- **Features**: VIX levels, NIFTY volatility, volume patterns, open interest
- **Data Sources**: Real-time via Yahoo Finance API

#### `lstm_model.py` - LSTM Model Implementation

```python
class IVLSTMModel:
    def build_model(self):
```
- **Purpose**: Build LSTM neural network for IV prediction
- **Architecture**: Multi-layer LSTM with dropout
- **Input Features**: 30-day lookback window

```python
def train(self, X, y):
```
- **Purpose**: Train LSTM model on historical data
- **Data Sources**: Historical IV data from Yahoo Finance
- **Training Data**: 1+ years of daily IV observations

#### `volatility_features.py` - Feature Engineering

```python
def calculate_realized_volatility(prices, window=30):
```
- **Purpose**: Calculate historical volatility from price data
- **Data Sources**: OHLC price data from Yahoo Finance
- **Method**: Standard deviation of log returns

```python
def get_vix_features(start_date, end_date):
```
- **Purpose**: Extract VIX-based features
- **Data Sources**: VIX index data via Yahoo Finance
- **Features**: VIX levels, VIX term structure, volatility risk premium

### Price Movement Models (`models/price_movement/`)

#### `live_price_predictor.py` - Real-time Price Movement Prediction

```python
class LivePricePredictor:
    def predict_next_movement(self):
```
- **Purpose**: Predict next price movement (UP/DOWN/SIDEWAYS)
- **Data Sources**:
  - **Yahoo Finance**: Real-time NIFTY OHLCV data
  - **Technical Indicators**: EMA, MACD, RSI from price data
  - **Volume Analysis**: Trading volume patterns
- **Model**: CNN-LSTM hybrid
- **Timeframe**: 5-minute candles
- **Returns**: Direction prediction with confidence

```python
def get_technical_features(self, symbol):
```
- **Purpose**: Calculate technical indicators for prediction
- **Data Sources**: Real-time price data via Yahoo Finance
- **Indicators**: 
  - EMA 6/15 crossover
  - MACD histogram
  - RSI 14-period
  - Bollinger Bands
  - Volume moving averages

#### `cnn_models.py` - CNN Model Architecture

```python
class PriceCNN:
    def build_cnn_lstm_model(self):
```
- **Purpose**: Build CNN-LSTM hybrid for price prediction
- **Architecture**: 1D CNN for pattern recognition + LSTM for sequence modeling
- **Input**: 50-candle lookback window with OHLCV + indicators

```python
def prepare_price_data(self, data):
```
- **Purpose**: Prepare price data for CNN input
- **Process**: Normalize prices, create candlestick patterns, technical features
- **Data Sources**: Real-time market data

#### `pattern_learner.py` - Chart Pattern Recognition

```python
class PatternLearner:
    def identify_patterns(self, price_data):
```
- **Purpose**: Identify chart patterns for price prediction
- **Patterns**: Head & shoulders, triangles, flags, support/resistance
- **Data Sources**: Historical OHLC data from Yahoo Finance

### Options Anomaly Detection (`models/options_anomaly/`)

#### `live_anomaly_detector.py` - Real-time Anomaly Detection

```python
class LiveAnomalyDetector:
    def find_mispriced_options(self):
```
- **Purpose**: Identify mispriced options for arbitrage
- **Data Sources**:
  - **Options Chain Data**: Real-time options prices
  - **Underlying Price**: NIFTY spot price via Yahoo Finance
  - **Risk-free Rate**: Government bond yields
  - **Dividend Yield**: For dividend adjustments
- **Method**: Compare market price vs Black-Scholes theoretical price
- **Returns**: List of mispriced options with expected profit

```python
def calculate_theoretical_price(self, option_data):
```
- **Purpose**: Calculate fair value using Black-Scholes model
- **Inputs**: Spot price, strike, time to expiry, risk-free rate, volatility
- **Data Sources**: Real-time market data and IV estimates

#### `black_scholes.py` - Options Pricing Model

```python
def black_scholes_call(S, K, T, r, sigma):
```
- **Purpose**: Calculate theoretical call option price
- **Method**: Black-Scholes formula
- **Inputs**: Current price (S), strike (K), time (T), rate (r), volatility (σ)

```python
def black_scholes_put(S, K, T, r, sigma):
```
- **Purpose**: Calculate theoretical put option price
- **Method**: Black-Scholes formula with put-call parity

```python
def calculate_greeks(S, K, T, r, sigma):
```
- **Purpose**: Calculate option Greeks (Delta, Gamma, Theta, Vega)
- **Use**: Risk management and hedging strategies

#### `arbitrage_detector.py` - Arbitrage Opportunity Detection

```python
class ArbitrageDetector:
    def find_arbitrage_opportunities(self):
```
- **Purpose**: Detect risk-free arbitrage opportunities
- **Methods**: 
  - Put-call parity violations
  - Calendar spread arbitrage
  - Strike arbitrage
- **Data Sources**: Complete options chain from market data

### Risk Analysis Models (`models/risk_analysis/`)

#### `risk_monitor.py` - Real-time Risk Monitoring

```python
class RiskMonitor:
    def __init__(self, initial_capital, max_daily_loss, max_portfolio_loss):
```
- **Purpose**: Initialize risk monitoring system
- **Parameters**: Capital limits, loss thresholds
- **Monitoring**: Real-time P&L tracking

```python
def can_place_trade(self):
```
- **Purpose**: Validate if new trade is allowed
- **Checks**: 
  - Daily loss limits
  - Trade frequency limits
  - Portfolio concentration
  - Available capital
- **Returns**: Boolean with reason

```python
def update_trade_result(self, pnl):
```
- **Purpose**: Update risk metrics after trade execution
- **Tracking**: Daily P&L, win rate, drawdown
- **Alerts**: Risk limit breaches

```python
def get_risk_dashboard(self):
```
- **Purpose**: Generate risk metrics summary
- **Metrics**: VaR, max drawdown, Sharpe ratio, win rate
- **Returns**: Dictionary with all risk metrics

#### `portfolio_metrics.py` - Portfolio Performance Calculation

```python
def calculate_sharpe_ratio(returns, risk_free_rate):
```
- **Purpose**: Calculate risk-adjusted returns
- **Data Sources**: Portfolio returns, risk-free rate from config

```python
def calculate_var(returns, confidence=0.95):
```
- **Purpose**: Calculate Value at Risk
- **Method**: Historical simulation or parametric approach
- **Returns**: VaR at specified confidence level

```python
def calculate_max_drawdown(equity_curve):
```
- **Purpose**: Calculate maximum portfolio drawdown
- **Input**: Historical portfolio values
- **Returns**: Peak-to-trough drawdown percentage

#### `position_sizer.py` - Position Sizing Algorithms

```python
class PositionSizer:
    def kelly_criterion_size(self, win_prob, avg_win, avg_loss, capital):
```
- **Purpose**: Calculate optimal position size using Kelly criterion
- **Formula**: f = (bp - q) / b
- **Returns**: Optimal fraction of capital to risk

```python
def volatility_based_sizing(self, volatility, capital, target_vol):
```
- **Purpose**: Size positions based on volatility targeting
- **Data Sources**: Historical volatility from price data
- **Method**: Inverse volatility weighting

---

## 🔧 Configuration (`config/`)

### `config.py` - Main Configuration File

**Purpose**: Central configuration for all AIBot components.

#### Configuration Sections:

```python
API_KEYS = {
    'dhan_api_key': 'your_dhan_api_key_here',
    'dhan_access_token': 'your_dhan_access_token_here',
    'alpha_vantage_key': 'your_alpha_vantage_key_here'
}
```
- **Purpose**: API credentials for data sources
- **Dhan API**: For live Indian market data and order execution
- **Alpha Vantage**: Alternative data source (optional)

```python
TRADING_CONFIG = {
    'initial_capital': 10000,
    'max_daily_loss': 0.10,
    'max_trades_per_day': 10,
    'position_size_method': 'kelly'
}
```
- **Purpose**: Trading parameters and risk management
- **Capital**: Starting amount in INR
- **Risk Limits**: Daily loss and frequency limits

```python
MODEL_CONFIG = {
    'iv_prediction': {
        'lookback_days': 30,
        'prediction_horizon': 5,
        'model_type': 'lstm'
    }
}
```
- **Purpose**: AI model configuration
- **Parameters**: Lookback windows, prediction horizons, model types

```python
DATA_CONFIG = {
    'data_sources': ['yahoo_finance', 'dhan_api'],
    'update_frequency': 300,
    'cache_enabled': True
}
```
- **Purpose**: Data source configuration
- **Sources**: Yahoo Finance (primary), Dhan API (optional)
- **Caching**: Performance optimization

---

## 📊 Utilities (`utils/`)

### Indicators (`utils/indicators/`)

#### `technical_indicators.py` - Technical Analysis

```python
def calculate_ema(prices, period):
```
- **Purpose**: Calculate Exponential Moving Average
- **Data Sources**: Price data from Yahoo Finance
- **Used In**: Price prediction models, signal generation

```python
def calculate_macd(prices, fast=12, slow=26, signal=9):
```
- **Purpose**: Calculate MACD indicator
- **Components**: MACD line, signal line, histogram
- **Use**: Trend and momentum analysis

```python
def calculate_rsi(prices, period=14):
```
- **Purpose**: Calculate Relative Strength Index
- **Range**: 0-100 (overbought/oversold levels)
- **Use**: Mean reversion signals

#### `data_source_manager.py` - Data Source Management

```python
class DataSourceManager:
    def get_nifty_data(self, start_date, end_date):
```
- **Purpose**: Fetch NIFTY index data
- **Data Source**: Yahoo Finance via `yfinance.download('^NSEI')`
- **Returns**: OHLCV data for specified date range

```python
def get_options_chain(self, symbol, expiry):
```
- **Purpose**: Fetch options chain data
- **Data Sources**: 
  - **Primary**: Yahoo Finance options data
  - **Fallback**: Dhan API (when configured)
- **Returns**: Complete options chain with strikes, premiums, IV

```python
def get_vix_data(self, start_date, end_date):
```
- **Purpose**: Fetch India VIX data
- **Data Source**: Yahoo Finance via symbol '^INDIAVIX'
- **Use**: Volatility analysis and IV prediction

#### `feature_pipeline.py` - Feature Engineering Pipeline

```python
class FeaturePipeline:
    def create_market_features(self, data):
```
- **Purpose**: Generate market-based features
- **Features**: 
  - Price-based: returns, volatility, momentum
  - Volume-based: volume ratio, VWAP
  - Technical: RSI, MACD, Bollinger Bands
- **Data Sources**: Real-time market data

```python
def create_sentiment_features(self, data):
```
- **Purpose**: Generate market sentiment features
- **Features**: VIX levels, put-call ratio, advance-decline ratio
- **Data Sources**: Market breadth data from Yahoo Finance

---

## 📁 Data Management (`data/`)

### Structure:
- **`raw/`**: Raw market data from APIs
- **`processed/`**: Cleaned and engineered features
- **`features/`**: Feature sets for ML models

### Data Flow:
1. **Raw Data Ingestion**: Yahoo Finance → `data/raw/`
2. **Data Cleaning**: Remove outliers, handle missing values
3. **Feature Engineering**: Technical indicators, market features
4. **Model Input**: Processed features → AI models

---

## 🔍 Data Sources Summary

### Primary Data Sources:

#### 1. Yahoo Finance (`yfinance`)
- **Usage**: Primary data source for all market data
- **Data Types**:
  - NIFTY index data (symbol: `^NSEI`)
  - India VIX (symbol: `^INDIAVIX`)
  - Options chain data
  - Historical OHLCV data
- **Update Frequency**: Real-time during market hours
- **Reliability**: High, free access
- **Rate Limits**: Reasonable for retail use

#### 2. Dhan API (Optional)
- **Usage**: Live Indian market data and order execution
- **Requirements**: Dhan trading account + API credentials
- **Data Types**:
  - Real-time market data
  - Options chain with Greeks
  - Order placement and management
  - Portfolio tracking
- **Advantages**: 
  - Live order execution
  - Indian market focus
  - Real-time data
- **Setup Required**: API key configuration in `config.py`

#### 3. Market Derived Data
- **Technical Indicators**: Calculated from price data
- **Volatility Metrics**: Derived from price movements
- **Market Breadth**: Calculated from multiple instruments
- **Sentiment Indicators**: Derived from options data

### Data Update Frequency:
- **Real-time Data**: During market hours (9:15 AM - 3:30 PM IST)
- **Historical Data**: Downloaded as needed for model training
- **Cache Duration**: 1 hour for performance optimization
- **Model Updates**: Daily after market close

### Data Quality Measures:
- **Validation**: Price reasonableness checks
- **Cleaning**: Outlier removal, missing value handling
- **Backup Sources**: Fallback to alternative data providers
- **Error Handling**: Graceful degradation when data unavailable

---

## 🚀 Execution Flow

### 1. System Startup
1. **Dependency Check**: Validate required packages
2. **Config Loading**: Load trading and model parameters
3. **Model Initialization**: Load AI models (with graceful fallback)
4. **Web Server Start**: Launch interface on localhost:8080

### 2. Trading Loop (Every 5 minutes)
1. **Data Fetch**: Get latest market data from Yahoo Finance
2. **Feature Calculation**: Compute technical indicators
3. **AI Inference**: Run all available models
4. **Signal Generation**: Aggregate model outputs
5. **Risk Check**: Validate against risk limits
6. **Decision Making**: Generate final trading signal
7. **Trade Execution**: Place orders (simulated by default)

### 3. Risk Monitoring (Continuous)
1. **P&L Tracking**: Update portfolio metrics
2. **Risk Metrics**: Calculate VaR, drawdown
3. **Limit Monitoring**: Check daily loss limits
4. **Alert Generation**: Risk breach notifications

### 4. Web Interface (Real-time)
1. **Status Updates**: Every 10 seconds
2. **User Commands**: Start/stop trading
3. **Dashboard Display**: Live metrics and logs
4. **API Endpoints**: RESTful interface

---

## ⚠️ Important Notes

### Safety Features:
- **Paper Trading**: Default mode (no real money)
- **Risk Limits**: Hard-coded maximum losses
- **Emergency Stop**: Manual override capability
- **Graceful Degradation**: Works with missing models

### Production Considerations:
- **API Rate Limits**: Monitor Yahoo Finance usage
- **Data Quality**: Implement data validation
- **Error Handling**: Robust error recovery
- **Monitoring**: Comprehensive logging and alerting

### Customization:
- **Model Addition**: Easy to add new AI models
- **Strategy Changes**: Modify decision logic
- **Data Sources**: Add alternative data providers
- **Risk Rules**: Customize risk management

This documentation covers all major Python files and their integration with real-time data sources. The system is designed for safety with paper trading as default and comprehensive risk management built-in.