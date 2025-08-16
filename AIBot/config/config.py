"""
AIBot Configuration Template
===========================

Copy this file to config.py and update with your actual values
"""

# API Configuration
import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    'dhan_client_id': os.getenv('DHAN_CLIENT_ID', 'your_client_id_here'),
    'dhan_access_token': os.getenv('DHAN_ACCESS_TOKEN', 'your_access_token_here'),
    'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key_here'),
}

# Trading Configuration
TRADING_CONFIG = {
    'initial_capital': 10000,  # Initial capital in INR
    'max_daily_loss': 0.10,   # Maximum daily loss (10%)
    'max_trades_per_day': 10,
    'position_size_method': 'kelly',  # 'fixed', 'kelly', 'volatility'
    'risk_free_rate': 0.07,   # 7% risk-free rate
}

# Model Configuration
MODEL_CONFIG = {
    'iv_prediction': {
        'lookback_days': 30,
        'prediction_horizon': 5,
        'retrain_frequency': 'weekly',
        'model_type': 'lstm',
        'features': ['vix', 'nifty_returns', 'volume', 'open_interest']
    },
    'price_movement': {
        'lookback_candles': 50,
        'prediction_horizon': 1,  # Next candle
        'model_type': 'cnn_lstm',
        'timeframe': '5min',
        'features': ['ohlcv', 'technical_indicators', 'market_sentiment']
    },
    'options_anomaly': {
        'contamination': 0.1,  # Expected percentage of anomalies
        'model_type': 'isolation_forest',
        'features': ['iv_rank', 'delta', 'gamma', 'theta', 'vega', 'volume']
    },
    'risk_analysis': {
        'var_confidence': 0.95,  # 95% VaR
        'lookback_period': 252,  # 1 year of trading days
        'stress_scenarios': ['market_crash', 'volatility_spike', 'interest_rate_change']
    }
}

# Data Configuration
DATA_CONFIG = {
    'data_sources': ['yahoo_finance', 'dhan_api'],
    'update_frequency': 300,  # Update every 5 minutes
    'historical_data_days': 365,
    'cache_enabled': True,
    'cache_duration': 3600,  # 1 hour cache
}

# Market Configuration
MARKET_CONFIG = {
    'market_hours': {
        'start': '09:15',
        'end': '15:30',
        'timezone': 'Asia/Kolkata'
    },
    'instruments': {
        'index': 'NIFTY',
        'options_expiry': 'weekly',  # 'weekly' or 'monthly'
        'strike_range': 10,  # Number of strikes on each side
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/aibot.log',
    'max_file_size': '10MB',
    'backup_count': 5,
    'log_format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_gpu': True,
    'max_workers': 4,  # For parallel processing
    'batch_size': 32,
    'memory_limit': '4GB',
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'enabled': True,
    'channels': ['console', 'file'],  # 'email', 'telegram', 'discord'
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'to_addresses': ['recipient@gmail.com']
    },
    'telegram': {
        'bot_token': 'your_telegram_bot_token',
        'chat_id': 'your_chat_id'
    }
}

# Database Configuration (if using)
DATABASE_CONFIG = {
    'type': 'sqlite',  # 'sqlite', 'postgresql', 'mysql'
    'path': 'data/aibot.db',  # For SQLite
    'host': 'localhost',
    'port': 5432,
    'database': 'aibot',
    'username': 'your_username',
    'password': 'your_password'
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_indicators': [
        'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
        'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'adx_14'
    ],
    'price_features': [
        'returns', 'log_returns', 'volatility', 'high_low_ratio', 'volume_ratio'
    ],
    'market_features': [
        'vix', 'term_structure', 'put_call_ratio', 'advance_decline_ratio'
    ]
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'initial_capital': 100000,
    'commission': 0.0005,  # 0.05% commission
    'slippage': 0.0001,    # 0.01% slippage
    'benchmark': 'NIFTY50'
}