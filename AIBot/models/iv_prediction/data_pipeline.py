"""
Nifty 50 Data Pipeline Module
============================

Data collection and preparation for volatility prediction
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


class Nifty50DataPipeline:
    """
    Data pipeline specifically for Nifty 50 volatility prediction
    """

    def __init__(self, period='3y'):
        self.period = period
        self.scaler = StandardScaler()

    def fetch_nifty_data(self):
        """Fetch Nifty 50 and related Indian market data"""
        print("Fetching Nifty 50 data...")

        # Nifty 50 data (using Yahoo Finance with NSE symbol)
        nifty = yf.Ticker('^NSEI')
        df = nifty.history(period=self.period)

        # India VIX (volatility index for Indian markets)
        try:
            india_vix = yf.Ticker('^INDIAVIX')
            vix_data = india_vix.history(period=self.period)
        except:
            # Fallback: create synthetic VIX from Nifty volatility
            print("India VIX data not available, creating synthetic volatility index...")
            vix_data = pd.DataFrame(index=df.index)
            vix_data['Close'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Bank Nifty for financial sector sentiment
        try:
            bank_nifty = yf.Ticker('^NSEBANK')
            bank_data = bank_nifty.history(period=self.period)
        except:
            print("Bank Nifty data not available, using Nifty data...")
            bank_data = df.copy()

        # USD/INR exchange rate (important for Indian markets)
        try:
            usdinr = yf.Ticker('USDINR=X')
            inr_data = usdinr.history(period=self.period)
        except:
            print("USD/INR data not available, creating synthetic...")
            inr_data = pd.DataFrame(index=df.index)
            inr_data['Close'] = 75 + np.random.normal(0, 0.5, len(df))  # Approximate INR rate

        print(f"Nifty 50 data shape: {df.shape}")
        return df, vix_data, bank_data, inr_data

    def calculate_indian_market_indicators(self, df):
        """Calculate technical indicators with Indian market considerations"""
        print("Calculating technical indicators for Nifty 50...")

        # Standard technical indicators
        df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)  # Important for Nifty
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # Volatility indicators
        df['bb_high'], df['bb_mid'], df['bb_low'] = (
            ta.volatility.bollinger_hband(df['Close']),
            ta.volatility.bollinger_mavg(df['Close']),
            ta.volatility.bollinger_lband(df['Close'])
        )
        df['atr_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['atr_20'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=20)

        # Momentum indicators
        df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['Close'], window=21)  # Alternative RSI period
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['macd_signal'] = ta.trend.macd_signal(df['Close'])
        df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

        # Volume indicators (important for Nifty)
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Support and Resistance levels (important for Indian markets)
        df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['pivot_sma'] = df['pivot'].rolling(20).mean()

        return df

    def prepare_nifty_features(self):
        """Main method to prepare all Nifty 50 features"""
        from .volatility_features import VolatilityFeatureCalculator
        from .market_features import IndianMarketFeatures
        
        # Fetch data
        df, vix_data, bank_data, inr_data = self.fetch_nifty_data()

        # Calculate basic indicators
        df = self.calculate_indian_market_indicators(df)

        # Calculate volatility features
        volatility_calc = VolatilityFeatureCalculator()
        df = volatility_calc.calculate_nifty_volatility_features(df)

        # Add market-specific features
        market_features = IndianMarketFeatures()
        df = market_features.add_indian_market_features(df, vix_data, bank_data, inr_data)
        df = market_features.create_nifty_target_variable(df)

        # Clean data
        df = df.dropna()

        print(f"Final Nifty 50 dataset shape: {df.shape}")
        print(f"Available features: {df.columns.tolist()}")
        return df