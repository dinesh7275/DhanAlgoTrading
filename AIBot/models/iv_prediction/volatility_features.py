"""
Volatility Features Calculator
=============================

Calculate various volatility-related features for Nifty 50
"""

import numpy as np
import pandas as pd


class VolatilityFeatureCalculator:
    """
    Calculate volatility features specific to Nifty 50
    """

    def calculate_nifty_volatility_features(self, df):
        """Calculate volatility features specific to Nifty 50"""
        print("Calculating Nifty 50 volatility features...")

        # Returns calculation
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['abs_returns'] = np.abs(df['returns'])

        # Historical volatility (multiple windows for Indian market patterns)
        for window in [5, 10, 15, 20, 30, 45]:
            df[f'hist_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
            df[f'hist_vol_ewm_{window}'] = df['returns'].ewm(span=window).std() * np.sqrt(252)

        # Parkinson volatility estimator (High-Low based)
        for window in [10, 20, 30]:
            df[f'parkinson_vol_{window}'] = np.sqrt(252 * (
                    np.log(df['High']/df['Low'])**2 * 0.25 -
                    (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2
            ).rolling(window).mean())

        # Garman-Klass volatility (more robust for Indian markets)
        df['gk_volatility_10'] = np.sqrt(252 * (
                np.log(df['High']/df['Close']) * np.log(df['High']/df['Open']) +
                np.log(df['Low']/df['Close']) * np.log(df['Low']/df['Open'])
        ).rolling(10).mean())

        df['gk_volatility_20'] = np.sqrt(252 * (
                np.log(df['High']/df['Close']) * np.log(df['High']/df['Open']) +
                np.log(df['Low']/df['Close']) * np.log(df['Low']/df['Open'])
        ).rolling(20).mean())

        # Rogers-Satchell volatility (no drift assumption)
        df['rs_volatility'] = np.sqrt(252 * (
                np.log(df['High']/df['Close']) * np.log(df['High']/df['Open']) +
                np.log(df['Low']/df['Close']) * np.log(df['Low']/df['Open'])
        ).rolling(20).mean())

        # Price gap analysis (important for Nifty due to overnight moves)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_abs'] = np.abs(df['gap'])
        df['gap_up'] = np.where(df['gap'] > 0, df['gap'], 0)
        df['gap_down'] = np.where(df['gap'] < 0, np.abs(df['gap']), 0)

        # Intraday ranges
        df['high_low_ratio'] = df['High'] / df['Low']
        df['high_open_ratio'] = df['High'] / df['Open']
        df['low_open_ratio'] = df['Low'] / df['Open']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']

        # True Range based features
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['true_range_pct'] = df['true_range'] / df['Close']

        return df


class VolatilityRegimeAnalyzer:
    """
    Analyze volatility regimes for better prediction
    """

    def __init__(self):
        self.regime_thresholds = None

    def calculate_volatility_regimes(self, volatility_series, n_regimes=3):
        """
        Classify volatility into regimes (Low, Medium, High)
        """
        if self.regime_thresholds is None:
            percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
            self.regime_thresholds = np.percentile(volatility_series.dropna(), percentiles)

        regimes = np.digitize(volatility_series, self.regime_thresholds)
        return regimes

    def get_regime_statistics(self, volatility_series, regimes):
        """
        Get statistics for each volatility regime
        """
        regime_stats = {}
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            vol_subset = volatility_series[mask]
            
            regime_stats[f'regime_{regime}'] = {
                'count': len(vol_subset),
                'mean': vol_subset.mean(),
                'std': vol_subset.std(),
                'min': vol_subset.min(),
                'max': vol_subset.max(),
                'median': vol_subset.median()
            }
            
        return regime_stats