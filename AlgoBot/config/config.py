#!/usr/bin/env python3
"""
Configuration Management for Dhan Algorithmic Trading
=====================================================

Centralized configuration for all trading parameters and system settings
"""

import os
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Main configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///trading.db'
    
    # Trading Configuration
    INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', '10000'))
    MAX_DAILY_LOSS = float(os.environ.get('MAX_DAILY_LOSS', '0.10'))  # 10%
    MAX_PORTFOLIO_LOSS = float(os.environ.get('MAX_PORTFOLIO_LOSS', '0.15'))  # 15%
    MAX_TRADES_PER_DAY = int(os.environ.get('MAX_TRADES_PER_DAY', '10'))
    DAILY_PROFIT_TARGET = float(os.environ.get('DAILY_PROFIT_TARGET', '0.10'))  # 10%
    
    # Dhan API Configuration
    DHAN_CLIENT_ID = os.environ.get('DHAN_CLIENT_ID')
    DHAN_ACCESS_TOKEN = os.environ.get('DHAN_ACCESS_TOKEN')
    
    # Market Data Configuration
    MARKET_DATA_INTERVAL = int(os.environ.get('MARKET_DATA_INTERVAL', '30'))  # seconds
    SIGNAL_UPDATE_INTERVAL = int(os.environ.get('SIGNAL_UPDATE_INTERVAL', '60'))  # seconds
    
    # AI Model Configuration
    MODEL_UPDATE_FREQUENCY = int(os.environ.get('MODEL_UPDATE_FREQUENCY', '240'))  # minutes
    IV_PREDICTION_LOOKBACK = int(os.environ.get('IV_PREDICTION_LOOKBACK', '7'))  # days
    PRICE_PREDICTION_LOOKBACK = int(os.environ.get('PRICE_PREDICTION_LOOKBACK', '5'))  # days
    
    # Options Trading Configuration
    NIFTY_LOT_SIZE = int(os.environ.get('NIFTY_LOT_SIZE', '50'))
    MAX_LOTS_PER_TRADE = int(os.environ.get('MAX_LOTS_PER_TRADE', '2'))
    OPTION_PREMIUM_RANGE = {
        'min': float(os.environ.get('MIN_OPTION_PREMIUM', '10')),
        'max': float(os.environ.get('MAX_OPTION_PREMIUM', '200'))
    }
    
    # Risk Management
    POSITION_SIZE_METHOD = os.environ.get('POSITION_SIZE_METHOD', 'fixed_fractional')
    STOP_LOSS_PERCENTAGE = float(os.environ.get('STOP_LOSS_PERCENTAGE', '0.20'))  # 20%
    TAKE_PROFIT_PERCENTAGE = float(os.environ.get('TAKE_PROFIT_PERCENTAGE', '0.50'))  # 50%
    
    # Tax Configuration (Indian Markets)
    STT_RATE = float(os.environ.get('STT_RATE', '0.00017'))  # 0.017%
    EXCHANGE_CHARGES_RATE = float(os.environ.get('EXCHANGE_CHARGES_RATE', '0.000019'))  # 0.0019%
    BROKERAGE_PER_TRADE = float(os.environ.get('BROKERAGE_PER_TRADE', '20'))  # Rs 20
    GST_RATE = float(os.environ.get('GST_RATE', '0.18'))  # 18%
    
    # Technical Indicators Configuration
    EMA_FAST_PERIOD = int(os.environ.get('EMA_FAST_PERIOD', '6'))
    EMA_SLOW_PERIOD = int(os.environ.get('EMA_SLOW_PERIOD', '15'))
    RSI_PERIOD = int(os.environ.get('RSI_PERIOD', '14'))
    MACD_FAST = int(os.environ.get('MACD_FAST', '12'))
    MACD_SLOW = int(os.environ.get('MACD_SLOW', '26'))
    MACD_SIGNAL = int(os.environ.get('MACD_SIGNAL', '9'))
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    CONFIG_DIR = BASE_DIR / 'config'
    LOGS_DIR = BASE_DIR / 'logs'
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    
    # Create directories if they don't exist
    for directory in [CONFIG_DIR, LOGS_DIR, DATA_DIR, MODELS_DIR]:
        directory.mkdir(exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Override with production-specific settings
    MAX_DAILY_LOSS = 0.05  # More conservative in production
    MAX_TRADES_PER_DAY = 5

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    INITIAL_CAPITAL = 1000  # Smaller capital for testing

def load_trading_config():
    """Load trading configuration from JSON file"""
    config_file = Config.CONFIG_DIR / 'trading_config.json'
    
    default_config = {
        'capital_management': {
            'initial_capital': Config.INITIAL_CAPITAL,
            'max_daily_loss': Config.MAX_DAILY_LOSS,
            'max_portfolio_loss': Config.MAX_PORTFOLIO_LOSS,
            'daily_profit_target': Config.DAILY_PROFIT_TARGET,
            'position_size_method': Config.POSITION_SIZE_METHOD
        },
        'risk_management': {
            'max_trades_per_day': Config.MAX_TRADES_PER_DAY,
            'stop_loss_percentage': Config.STOP_LOSS_PERCENTAGE,
            'take_profit_percentage': Config.TAKE_PROFIT_PERCENTAGE,
            'max_lots_per_trade': Config.MAX_LOTS_PER_TRADE
        },
        'technical_indicators': {
            'ema_fast': Config.EMA_FAST_PERIOD,
            'ema_slow': Config.EMA_SLOW_PERIOD,
            'rsi_period': Config.RSI_PERIOD,
            'macd_fast': Config.MACD_FAST,
            'macd_slow': Config.MACD_SLOW,
            'macd_signal': Config.MACD_SIGNAL
        },
        'options_trading': {
            'lot_size': Config.NIFTY_LOT_SIZE,
            'premium_range': Config.OPTION_PREMIUM_RANGE,
            'focus': 'buying_only'  # CE/PE buying only
        },
        'market_data': {
            'update_interval': Config.MARKET_DATA_INTERVAL,
            'signal_interval': Config.SIGNAL_UPDATE_INTERVAL
        },
        'ai_models': {
            'iv_prediction': {
                'enabled': True,
                'lookback_days': Config.IV_PREDICTION_LOOKBACK,
                'update_frequency': Config.MODEL_UPDATE_FREQUENCY
            },
            'price_movement': {
                'enabled': True,
                'lookback_days': Config.PRICE_PREDICTION_LOOKBACK,
                'update_frequency': Config.MODEL_UPDATE_FREQUENCY
            },
            'anomaly_detection': {
                'enabled': True,
                'contamination': 0.1,
                'lookback_days': 3
            }
        },
        'taxes': {
            'stt_rate': Config.STT_RATE,
            'exchange_charges_rate': Config.EXCHANGE_CHARGES_RATE,
            'brokerage_per_trade': Config.BROKERAGE_PER_TRADE,
            'gst_rate': Config.GST_RATE
        }
    }
    
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
                logger.info(f"Loaded trading configuration from {config_file}")
        else:
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                logger.info(f"Created default trading configuration at {config_file}")
    
    except Exception as e:
        logger.error(f"Error loading trading config: {e}")
    
    return default_config

def save_trading_config(config_data):
    """Save trading configuration to JSON file"""
    config_file = Config.CONFIG_DIR / 'trading_config.json'
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved trading configuration to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving trading config: {e}")
        return False

def get_environment_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig
    elif env == 'testing':
        return TestingConfig
    else:
        return DevelopmentConfig

def validate_dhan_credentials():
    """Validate Dhan API credentials"""
    client_id = os.environ.get('DHAN_CLIENT_ID')
    access_token = os.environ.get('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        logger.warning("Dhan API credentials not found in environment variables")
        return False
    
    logger.info("Dhan API credentials found")
    return True

def setup_logging():
    """Setup logging configuration"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    Config.LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / 'trading.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured at {log_level} level")

# Initialize configuration on import
trading_config = load_trading_config()
setup_logging()

# Validate credentials
if not validate_dhan_credentials():
    logger.warning("Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables")

if __name__ == '__main__':
    print("Trading Configuration:")
    print(json.dumps(trading_config, indent=2))
    print(f"Environment: {get_environment_config().__name__}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Initial Capital: Rs. {Config.INITIAL_CAPITAL:,}")
    print(f"Max Daily Loss: {Config.MAX_DAILY_LOSS*100:.1f}%")
    print(f"Daily Profit Target: {Config.DAILY_PROFIT_TARGET*100:.1f}%")