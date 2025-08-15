# 🤖 AI Trading Bot with Dhan API Integration

Complete AI-powered trading system with live market data, ensemble AI models, and automated trading execution through Dhan API.

## 🌟 Features

### 🧠 AI Model Ensemble
- **Volatility Prediction**: LSTM models for implied volatility forecasting
- **Price Movement**: CNN+LSTM for directional predictions
- **Anomaly Detection**: Autoencoder for market anomaly identification
- **Risk Assessment**: Real-time portfolio risk analysis

### 📊 Live Data Integration
- Real-time Nifty 50 data via Dhan API
- Fallback to Yahoo Finance for reliability
- India VIX and Bank Nifty monitoring
- Options chain data integration

### 🛡️ Advanced Risk Management
- Real-time risk monitoring
- Circuit breakers for extreme scenarios
- Position sizing algorithms
- Portfolio stress testing
- Stop loss and target management

### 📈 Monitoring & Dashboard
- Web-based real-time dashboard
- Console monitoring mode
- Performance tracking
- Alert system
- Emergency controls

### 🔐 Security & Configuration
- Encrypted credential storage
- Comprehensive configuration management
- Paper trading mode for testing
- Secure API communication

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_live_trading.txt

# Quick start interface
python quick_start.py
```

### 2. Setup (First Time)

```bash
# Interactive setup
python quick_start.py  # Select option 1

# Or command line setup
python main_trading_bot.py --setup
```

### 3. Start Trading

```bash
# Paper trading (recommended first)
python main_trading_bot.py --paper

# Web dashboard
python main_trading_bot.py --mode dashboard

# Live trading (real money)
python main_trading_bot.py --live
```

## 📋 Prerequisites

### Dhan API Credentials
1. Open Dhan trading account
2. Get API credentials:
   - Client ID
   - Access Token
3. Have sufficient account balance

### System Requirements
- Python 3.8+
- 4GB+ RAM (for AI models)
- Internet connection
- Modern web browser (for dashboard)

## ⚙️ Configuration

### Trading Parameters
```json
{
  "trading": {
    "initial_capital": 1000000,
    "max_daily_loss": 0.05,
    "max_portfolio_loss": 0.10,
    "max_risk_per_trade": 0.02,
    "max_positions": 5,
    "paper_trading": true
  }
}
```

### Risk Management
```json
{
  "risk_management": {
    "stop_loss_percent": 0.02,
    "target_profit_ratio": 2.0,
    "max_drawdown": 0.15,
    "var_limit": 0.08
  }
}
```

### AI Models
```json
{
  "ai_models": {
    "confidence_threshold": 0.6,
    "model_weights": {
      "volatility_prediction": 0.25,
      "price_movement": 0.30,
      "anomaly_detection": 0.20,
      "risk_assessment": 0.25
    }
  }
}
```

## 🎯 Usage Examples

### Paper Trading
```python
from main_trading_bot import start_paper_trading

# Safe testing mode
start_paper_trading()
```

### Web Dashboard
```python
from main_trading_bot import start_web_dashboard

# Browser-based monitoring
start_web_dashboard()
```

### Custom Bot Setup
```python
from live_trading import (
    LiveTradingManager, TradingSignalEnsemble, 
    LiveRiskManager, setup_trading_environment
)

# Setup configuration and credentials
config_manager, credentials = setup_trading_environment()

# Create trading manager
trading_manager = LiveTradingManager(
    client_id=credentials['client_id'],
    access_token=credentials['access_token'],
    initial_capital=1000000
)

# Start trading
trading_manager.start_live_trading()
```

## 🧪 Testing Strategy

### 1. Paper Trading Phase
- Test with simulated trades
- Validate AI signal accuracy
- Check risk management
- Monitor performance metrics

### 2. Small Capital Testing
- Start with minimal real capital
- Validate API integration
- Test order execution
- Monitor slippage and fees

### 3. Gradual Scaling
- Increase position sizes gradually
- Monitor drawdowns
- Adjust risk parameters
- Scale based on performance

## 📊 Monitoring

### Web Dashboard Features
- Real-time portfolio value
- Active positions tracking
- AI signal history
- Performance charts
- Risk alerts
- Emergency controls

### Console Monitoring
- Text-based real-time updates
- Portfolio metrics
- Position details
- Risk status
- Signal tracking

### Alerts & Notifications
- Risk limit breaches
- Large drawdowns
- AI signal confidence
- System errors
- Market anomalies

## 🔧 Components Overview

### Core Components

#### 1. Data Fetcher (`dhan_data_fetcher.py`)
- Real-time market data via Dhan API
- Yahoo Finance fallback
- Data preprocessing for AI models

#### 2. AI Ensemble (`ai_ensemble.py`)
- Combined AI model predictions
- Signal generation and weighting
- Confidence scoring

#### 3. Trading Manager (`live_strategy_manager.py`)
- Order execution via Dhan API
- Position management
- Paper trading simulation

#### 4. Risk Manager (`risk_manager.py`)
- Real-time risk monitoring
- Circuit breakers
- Position sizing

#### 5. Dashboard (`monitoring_dashboard.py`)
- Web-based monitoring
- Real-time updates
- Emergency controls

#### 6. Config Manager (`config_manager.py`)
- Secure credential storage
- Configuration management
- Setup utilities

## 🛡️ Risk Controls

### Circuit Breakers
- Daily loss limit breach
- Portfolio loss limit breach
- Maximum drawdown breach
- High volatility scenarios

### Position Limits
- Maximum number of positions
- Single position size limits
- Sector concentration limits
- Correlation-based limits

### AI Signal Filters
- Minimum confidence thresholds
- Signal cooldown periods
- Ensemble score validation
- Market condition adjustments

## 🚨 Safety Features

### Paper Trading Mode
- No real money at risk
- Full feature testing
- Performance validation
- Strategy optimization

### Emergency Stops
- Immediate trading halt
- Position closure
- Web dashboard controls
- Keyboard interrupts

### Validation Checks
- API connectivity tests
- Account balance verification
- Configuration validation
- Model health checks

## 📈 Performance Tracking

### Key Metrics
- Total return and P&L
- Win rate and average win/loss
- Maximum drawdown
- Sharpe ratio
- Risk-adjusted returns

### AI Model Performance
- Signal accuracy
- Confidence calibration
- Model agreement rates
- Prediction stability

### Risk Metrics
- Value at Risk (VaR)
- Expected shortfall
- Portfolio volatility
- Correlation exposure

## 🔍 Troubleshooting

### Common Issues

#### API Connection Problems
```bash
# Check credentials
python -c "from live_trading import ConfigManager; cm = ConfigManager(); print('Credentials exist:', cm.load_credentials() is not None)"

# Test API connection
python -c "from live_trading import DhanLiveDataFetcher; df = DhanLiveDataFetcher('CLIENT_ID', 'ACCESS_TOKEN'); print('API works:', df.validate_connection())"
```

#### Model Loading Issues
```bash
# Check AI model dependencies
python -c "import tensorflow, sklearn, numpy, pandas; print('AI dependencies OK')"

# Verify model ensemble
python -c "from live_trading import TradingSignalEnsemble; tse = TradingSignalEnsemble(); print('Ensemble OK')"
```

#### Configuration Problems
```bash
# Reset configuration
python main_trading_bot.py --config  # Select reset option

# Re-run setup
python main_trading_bot.py --setup
```

### Log Files
- `live_trading_YYYYMMDD.log`: Daily trading logs
- `trading_log_YYYYMMDD_HHMMSS.json`: Session summaries
- `risk_report_YYYYMMDD_HHMMSS.json`: Risk analysis

## 📚 API Reference

### Main Classes

#### LiveTradingManager
```python
manager = LiveTradingManager(client_id, access_token, initial_capital)
manager.start_live_trading(update_interval=30)
```

#### TradingSignalEnsemble
```python
ensemble = TradingSignalEnsemble()
signal = ensemble.generate_ensemble_signal(market_data, portfolio_value)
```

#### LiveRiskManager
```python
risk_manager = LiveRiskManager(initial_capital, max_daily_loss, max_portfolio_loss)
risk_status = risk_manager.update_portfolio_value(current_value)
```

### Configuration Methods
```python
from live_trading import ConfigManager

config = ConfigManager()
config.setup_credentials_interactive()
config.update_config_section('trading', {'paper_trading': False})
```

## ⚡ Command Line Options

```bash
# Basic usage
python main_trading_bot.py

# Monitoring modes
python main_trading_bot.py --mode console      # Console monitoring
python main_trading_bot.py --mode dashboard    # Web dashboard
python main_trading_bot.py --mode headless     # No UI

# Trading modes
python main_trading_bot.py --paper             # Force paper trading
python main_trading_bot.py --live              # Force live trading

# Setup and configuration
python main_trading_bot.py --setup             # Complete setup
python main_trading_bot.py --config            # Configuration only
```

## 🤝 Contributing

1. Test thoroughly with paper trading
2. Follow existing code structure
3. Add comprehensive error handling
4. Update documentation
5. Add unit tests for new features

## ⚠️ Disclaimers

- **Financial Risk**: Trading involves substantial risk of loss
- **No Guarantees**: Past performance doesn't guarantee future results
- **Test First**: Always test with paper trading before going live
- **Monitor Closely**: Supervise automated trading systems
- **Regulatory Compliance**: Ensure compliance with local regulations

## 📞 Support

- Check logs for detailed error information
- Use paper trading mode for testing
- Review configuration settings
- Verify API credentials and connectivity
- Monitor system resources and internet connection

---

**Happy Trading! 🚀**

*Remember: The best traders manage risk first, profit second.*