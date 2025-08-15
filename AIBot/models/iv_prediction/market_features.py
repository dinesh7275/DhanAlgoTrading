"""
Indian Market Specific Features
==============================

Calculate features specific to Indian financial markets
"""

import numpy as np
import pandas as pd
import ta


class IndianMarketFeatures:
    """
    Add Indian market specific features
    """

    def add_indian_market_features(self, df, vix_data, bank_data, inr_data):
        """Add Indian market specific features"""
        print("Adding Indian market features...")

        # India VIX features
        df['india_vix'] = vix_data['Close'].reindex(df.index, method='ffill')
        df['india_vix_change'] = df['india_vix'].pct_change()
        df['india_vix_sma_10'] = df['india_vix'].rolling(10).mean()
        df['india_vix_sma_20'] = df['india_vix'].rolling(20).mean()
        df['india_vix_rsi'] = ta.momentum.rsi(df['india_vix'])

        # Bank Nifty correlation (financial sector health)
        bank_returns = bank_data['Close'].pct_change().reindex(df.index, method='ffill')
        nifty_returns = df['returns']

        # Rolling correlation with Bank Nifty
        df['bank_nifty_corr_20'] = nifty_returns.rolling(20).corr(bank_returns)
        df['bank_nifty_corr_60'] = nifty_returns.rolling(60).corr(bank_returns)

        # Bank Nifty relative performance
        bank_close = bank_data['Close'].reindex(df.index, method='ffill')
        df['bank_nifty_ratio'] = bank_close / df['Close']
        df['bank_nifty_ratio_sma'] = df['bank_nifty_ratio'].rolling(20).mean()

        # USD/INR features (important for FII flows)
        df['usdinr'] = inr_data['Close'].reindex(df.index, method='ffill')
        df['usdinr_change'] = df['usdinr'].pct_change()
        df['usdinr_sma_20'] = df['usdinr'].rolling(20).mean()
        df['usdinr_deviation'] = (df['usdinr'] - df['usdinr_sma_20']) / df['usdinr_sma_20']

        # Currency volatility impact
        df['usdinr_vol_10'] = df['usdinr_change'].rolling(10).std() * np.sqrt(252)

        # Add time-based features
        df = self._add_time_features(df)

        return df

    def _add_time_features(self, df):
        """Add time-based features specific to Indian markets"""
        # Indian market timing features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = (df.index.day >= 25).astype(int)  # Month-end effects
        df['is_quarter_end'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)

        # Expiry-related features (Nifty options expire on Thursdays)
        df['days_to_thursday'] = (3 - df.index.dayofweek) % 7  # Days to next Thursday
        df['is_expiry_week'] = (df['days_to_thursday'] <= 3).astype(int)

        # Seasonal patterns in Indian markets
        df['is_festival_season'] = ((df.index.month >= 9) & (df.index.month <= 11)).astype(int)
        df['is_budget_season'] = ((df.index.month == 2) | (df.index.month == 7)).astype(int)

        return df

    def create_nifty_target_variable(self, df):
        """Create target variable for Nifty volatility prediction"""
        print("Creating Nifty volatility target variable...")

        # Forward-looking realized volatility
        df['future_vol_5'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
        df['future_vol_10'] = df['returns'].shift(-10).rolling(10).std() * np.sqrt(252)
        df['future_vol_15'] = df['returns'].shift(-15).rolling(15).std() * np.sqrt(252)
        df['future_vol_20'] = df['returns'].shift(-20).rolling(20).std() * np.sqrt(252)

        # Primary target: 15-day forward volatility (better for Nifty options)
        df['target_volatility'] = df['future_vol_15']

        # Alternative targets for different time horizons
        df['target_vol_weekly'] = df['future_vol_5']
        df['target_vol_monthly'] = df['future_vol_20']

        return df


class SectorCorrelationAnalyzer:
    """
    Analyze correlations with different sectors
    """

    def __init__(self):
        self.sector_symbols = {
            'banking': '^NSEBANK',
            'it': '^CNXIT',
            'pharma': '^CNXPHARMA',
            'auto': '^CNXAUTO',
            'metal': '^CNXMETAL',
            'fmcg': '^CNXFMCG'
        }

    def calculate_sector_correlations(self, nifty_returns, period='3y', window=20):
        """
        Calculate rolling correlations with major sectors
        """
        import yfinance as yf
        
        correlations = {}
        
        for sector, symbol in self.sector_symbols.items():
            try:
                sector_ticker = yf.Ticker(symbol)
                sector_data = sector_ticker.history(period=period)
                sector_returns = sector_data['Close'].pct_change()
                
                # Align indices
                sector_returns = sector_returns.reindex(nifty_returns.index, method='ffill')
                
                # Rolling correlation
                correlations[f'{sector}_corr_{window}'] = nifty_returns.rolling(window).corr(sector_returns)
                
            except Exception as e:
                print(f"Failed to fetch {sector} data: {e}")
                correlations[f'{sector}_corr_{window}'] = pd.Series(0.5, index=nifty_returns.index)
        
        return correlations