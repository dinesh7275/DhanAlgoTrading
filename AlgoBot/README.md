# AlgoBot - Web-Based Algorithmic Trading Platform

## Overview
AlgoBot is a professional web-based algorithmic trading platform designed for Indian options market. It provides a complete trading infrastructure with real-time dashboard, live trading capabilities, and comprehensive risk management.

## Features

### 🌐 Web Dashboard
- **Real-time Trading Dashboard**: Live portfolio tracking and market data
- **User Authentication**: Secure login and session management
- **Responsive Design**: Mobile-friendly interface
- **Interactive Charts**: Real-time price charts and technical indicators

### 📊 Trading Features
- **Live Trading**: Integration with Dhan API for real options trading
- **Paper Trading**: Risk-free simulation mode
- **Portfolio Management**: Real-time P&L tracking and position management
- **Risk Management**: Automated stop-loss and position sizing

### 💰 Indian Market Specific
- **Tax Integration**: STT, exchange charges, and GST calculations
- **Options Trading**: CE/PE buying with real-time options chain
- **Market Hours**: Indian market timing and holiday calendar
- **INR Currency**: All calculations in Indian Rupees

### 🔧 Technical Features
- **REST API**: Complete API for all trading operations
- **WebSocket**: Real-time data streaming
- **Database Integration**: Trade history and portfolio persistence
- **Logging**: Comprehensive audit trail

## Project Structure

```
AlgoBot/
├── app.py                    # Main Flask application
├── main_trading_bot.py      # Trading bot entry point
├── quick_start.py           # Quick setup script
├── controllers/             # Request handlers
│   └── tradingController.py
├── services/                # Business logic
│   └── tradingService.py
├── models/                  # Data models
│   └── models.py
├── static/                  # Frontend assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   └── images/             # Image assets
├── templates/               # HTML templates
│   ├── login.html
│   ├── dashboard.html
│   ├── 404.html
│   └── 500.html
├── live_trading/            # Live trading infrastructure
│   ├── options_trading_manager.py
│   ├── dhan_data_fetcher.py
│   ├── ai_ensemble.py
│   └── risk_manager.py
├── config/                  # Configuration files
│   └── config.py
├── utils/                   # Utility functions
├── tests/                   # Test files
└── logs/                    # Log files
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
```bash
cp config/config_template.py config/config.py
# Edit config.py with your Dhan API credentials
```

### 3. Run Quick Setup
```bash
python quick_start.py
```

### 4. Start Web Application
```bash
python app.py
```

### 5. Access Dashboard
Open your browser and go to: `http://localhost:5000`

**Default Login:**
- Email: `demo@dhanalgo.com`
- Password: `demo123`

## Configuration

### Dhan API Setup
1. Create account at [Dhan](https://dhan.co)
2. Generate API credentials
3. Update `config/config.py`:
```python
DHAN_CONFIG = {
    'client_id': 'your_client_id',
    'access_token': 'your_access_token'
}
```

### Trading Parameters
```python
TRADING_CONFIG = {
    'initial_capital': 10000,     # Starting capital
    'max_daily_loss': 0.10,      # 10% max daily loss
    'max_trades_per_day': 10,    # Trade limit
    'position_size_method': 'fixed_fractional'
}
```

### Risk Management
```python
RISK_CONFIG = {
    'stop_loss_percentage': 0.05,  # 5% stop loss
    'profit_target': 0.10,         # 10% profit target
    'position_size': 0.02          # 2% position size
}
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/verify` - Verify token

### Trading
- `GET /api/portfolio` - Get portfolio status
- `POST /api/trade/execute` - Execute trade
- `GET /api/trades/history` - Get trade history
- `GET /api/positions` - Get current positions

### Market Data
- `GET /api/market/overview` - Market overview
- `GET /api/market/nifty` - Nifty 50 data
- `GET /api/market/options` - Options chain

### System
- `GET /api/system/status` - System status
- `GET /api/system/logs` - System logs
- `POST /api/system/start` - Start trading
- `POST /api/system/stop` - Stop trading

## Live Trading Modes

### Paper Trading (Default)
- Simulated trading with virtual money
- Full feature testing without risk
- Real market data and timing

### Live Trading
- Real money trading through Dhan API
- All safety checks and risk management active
- Comprehensive logging and monitoring

## Performance Metrics

### Dashboard Metrics
- **Portfolio Value**: Real-time portfolio valuation
- **P&L**: Realized and unrealized profits/losses
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum portfolio decline

### Risk Metrics
- **VaR (Value at Risk)**: Potential loss estimation
- **Position Concentration**: Portfolio diversification
- **Leverage**: Current leverage ratio
- **Daily Loss**: Current day's loss tracking

## Security Features

### Authentication
- Secure session management
- Password encryption
- Token-based API authentication
- Session timeout protection

### Trading Security
- Trade confirmation workflows
- Position size limits
- Maximum loss protection
- Suspicious activity detection

### Data Security
- Encrypted API communications
- Secure credential storage
- Audit logging
- Data backup protocols

## Monitoring & Alerts

### Real-time Monitoring
- Live portfolio tracking
- Position monitoring
- Risk metric alerts
- System health checks

### Alert Types
- **Profit Targets**: When trades reach profit goals
- **Stop Losses**: When positions hit stop loss
- **Risk Limits**: When risk thresholds are exceeded
- **System Issues**: Technical problems or failures

## Troubleshooting

### Common Issues

**Connection Problems:**
```bash
# Check API credentials
python -c "from config.config import DHAN_CONFIG; print(DHAN_CONFIG)"

# Test API connection
python utils/test_connection.py
```

**Trading Issues:**
- Verify market hours (9:15 AM - 3:30 PM IST)
- Check available capital
- Confirm risk limits not exceeded
- Review error logs in `logs/` directory

**Performance Issues:**
- Check system resources
- Verify network connectivity
- Review database performance
- Check log file sizes

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

### Database Management
```bash
# Initialize database
python utils/init_db.py

# Backup database
python utils/backup_db.py

# Restore database
python utils/restore_db.py
```

## Deployment

### Production Deployment
1. Set environment to production
2. Configure production database
3. Set up SSL certificates
4. Configure load balancer
5. Set up monitoring

### Docker Deployment
```bash
# Build image
docker build -t algobot .

# Run container
docker run -p 5000:5000 algobot
```

## Support
- GitHub Issues: [Report bugs and feature requests](https://github.com/dhanalgo/algobot/issues)
- Email: support@dhanalgo.com
- Documentation: [Wiki](https://github.com/dhanalgo/algobot/wiki)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.