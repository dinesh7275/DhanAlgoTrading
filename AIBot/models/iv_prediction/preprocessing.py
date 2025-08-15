"""
Feature Preprocessing for Nifty Volatility Prediction
====================================================

Handle feature selection, scaling, and sequence creation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


class NiftyFeaturePreprocessor:
    """
    Feature preprocessor optimized for Nifty 50 characteristics
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.selected_features = None
        self.feature_selector = None

    def select_nifty_features(self, df, selection_method='manual'):
        """Select features optimized for Nifty 50 volatility prediction"""

        if selection_method == 'manual':
            return self._manual_feature_selection(df)
        elif selection_method == 'statistical':
            return self._statistical_feature_selection(df)
        else:
            raise ValueError("selection_method must be 'manual' or 'statistical'")

    def _manual_feature_selection(self, df):
        """Manual feature selection based on domain knowledge"""
        # Core Nifty 50 features
        feature_columns = [
            # Price features
            'Close', 'High', 'Low', 'Volume', 'Open',

            # Technical indicators
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_high', 'bb_low', 'bb_mid', 'atr_14', 'atr_20',
            'rsi_14', 'rsi_21', 'macd', 'macd_signal',
            'stoch', 'stoch_signal', 'mfi',

            # Volume features
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',

            # Volatility features
            'hist_vol_5', 'hist_vol_10', 'hist_vol_15', 'hist_vol_20', 'hist_vol_30',
            'hist_vol_ewm_10', 'hist_vol_ewm_20', 'hist_vol_ewm_30',
            'parkinson_vol_10', 'parkinson_vol_20', 'parkinson_vol_30',
            'gk_volatility_10', 'gk_volatility_20', 'rs_volatility',

            # Market microstructure
            'gap', 'gap_abs', 'gap_up', 'gap_down',
            'high_low_ratio', 'high_open_ratio', 'low_open_ratio',
            'close_open_ratio', 'close_to_high', 'close_to_low',
            'true_range', 'true_range_pct',

            # Indian market specific
            'india_vix', 'india_vix_change', 'india_vix_sma_10', 'india_vix_sma_20',
            'india_vix_rsi', 'bank_nifty_corr_20', 'bank_nifty_corr_60',
            'bank_nifty_ratio', 'bank_nifty_ratio_sma',

            # Currency features
            'usdinr', 'usdinr_change', 'usdinr_sma_20', 'usdinr_deviation',
            'usdinr_vol_10',

            # Time features
            'day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end',
            'days_to_thursday', 'is_expiry_week', 'is_festival_season', 'is_budget_season',

            # Support/Resistance
            'pivot', 'pivot_sma'
        ]

        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]

        print(f"Selected {len(available_features)} features for Nifty 50 training")
        self.selected_features = available_features

        return df[available_features]

    def _statistical_feature_selection(self, df, k=50):
        """Statistical feature selection using SelectKBest"""
        # Exclude target variables and dates
        target_cols = ['target_volatility', 'target_vol_weekly', 'target_vol_monthly',
                      'future_vol_5', 'future_vol_10', 'future_vol_15', 'future_vol_20']
        
        feature_cols = [col for col in df.columns if col not in target_cols]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['target_volatility']

        # Remove rows with NaN in target
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Remove features with too many NaNs
        X = X.loc[:, X.isnull().sum() < len(X) * 0.3]

        # Fill remaining NaNs
        X = X.fillna(X.median())

        # Select k best features
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features using statistical selection")
        print(f"Top features: {selected_features[:10]}")
        
        return df[selected_features]

    def create_sequences(self, features, target, sequence_length=45, forecast_horizon=1):
        """Create sequences optimized for Nifty patterns (45 days = ~2 months trading)"""
        print(f"Creating Nifty sequences with length {sequence_length}...")

        X, y = [], []

        for i in range(sequence_length, len(features) - forecast_horizon + 1):
            # Feature sequence
            X.append(features[i-sequence_length:i])
            # Target value
            y.append(target[i + forecast_horizon - 1])

        return np.array(X), np.array(y)

    def preprocess_nifty_data(self, df, sequence_length=45, test_size=0.2, 
                             validation_size=0.2, selection_method='manual'):
        """Complete preprocessing pipeline for Nifty 50"""
        print("Starting Nifty 50 preprocessing...")

        # Select features
        features_df = self.select_nifty_features(df, selection_method)

        # Handle target
        target = df['target_volatility'].values

        # Remove any remaining NaN values
        valid_indices = ~(np.isnan(features_df.values).any(axis=1) | np.isnan(target))
        features_df = features_df[valid_indices]
        target = target[valid_indices]

        print(f"Nifty data shape after cleaning: {features_df.shape}")

        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(features_df)
        target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled, sequence_length)

        # Train-validation-test split (time-series aware)
        total_samples = len(X)
        train_size = int(total_samples * (1 - test_size - validation_size))
        val_size = int(total_samples * validation_size)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        print(f"Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
        print(f"Test set: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_importance(self, model=None):
        """
        Get feature importance scores if available
        """
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': scores[self.feature_selector.get_support()]
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print("Statistical feature selection not performed")
            return None