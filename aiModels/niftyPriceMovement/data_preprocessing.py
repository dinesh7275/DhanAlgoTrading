"""
Nifty Price Movement Data Preprocessing
======================================

Data preprocessing for Nifty price movement prediction
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class NiftyDataPreprocessor:
    """
    Preprocess Nifty data for price movement prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def fetch_nifty_data(self, period='5y', interval='1d'):
        """
        Fetch Nifty 50 data from Yahoo Finance
        """
        print(f"Fetching Nifty data for period: {period}")
        
        try:
            nifty = yf.Ticker("^NSEI")
            data = nifty.history(period=period, interval=interval)
            
            if data.empty:
                print("No data fetched. Using fallback data source...")
                # Fallback to alternative ticker
                nifty = yf.Ticker("NIFTY.NS")
                data = nifty.history(period=period, interval=interval)
            
            print(f"Fetched {len(data)} data points")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for Nifty data
        """
        print("Calculating technical indicators...")
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['price_change'] = df['Close'] - df['Open']
        df['price_change_pct'] = df['price_change'] / df['Open']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ATR (Average True Range)
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift())
        df['tr3'] = abs(df['Low'] - df['Close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Volatility measures
        df['volatility_5'] = df['returns'].rolling(5).std() * np.sqrt(252)
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        return df
    
    def add_time_features(self, df):
        """
        Add time-based features
        """
        print("Adding time features...")
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        
        # Market timing features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day_of_month'] >= 25)).astype(int)
        
        # Indian market specific
        df['is_expiry_week'] = ((df['day_of_week'] == 3) | 
                               ((df['day_of_week'] == 2) & (df.index.day >= 25))).astype(int)
        df['is_settlement_day'] = (df['day_of_week'] == 3).astype(int)
        
        return df
    
    def create_lag_features(self, df, lag_periods=[1, 2, 3, 5, 10]):
        """
        Create lagged features
        """
        print("Creating lag features...")
        
        lag_columns = ['returns', 'volume_ratio', 'rsi', 'macd', 'volatility_20']
        
        for col in lag_columns:
            if col in df.columns:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for window in [3, 5, 10]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume_ratio'].rolling(window).mean()
        
        return df
    
    def create_target_variable(self, df, prediction_horizon=1, classification_type='direction'):
        """
        Create target variable for prediction
        
        Parameters:
        prediction_horizon: Number of days ahead to predict
        classification_type: 'direction', 'magnitude', or 'regime'
        """
        print(f"Creating target variable for {classification_type} prediction...")
        
        if classification_type == 'direction':
            # Binary classification: Up (1) or Down (0)
            future_returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            df['target'] = (future_returns > 0).astype(int)
            
        elif classification_type == 'magnitude':
            # Multi-class: Large Down, Small Down, Small Up, Large Up
            future_returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            df['target'] = pd.cut(
                future_returns,
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=[0, 1, 2, 3, 4]  # Large Down, Small Down, Flat, Small Up, Large Up
            ).astype(int)
            
        elif classification_type == 'regime':
            # Volatility regime classification
            future_vol = df['volatility_20'].shift(-prediction_horizon)
            vol_percentiles = future_vol.quantile([0.33, 0.67])
            df['target'] = pd.cut(
                future_vol,
                bins=[-np.inf, vol_percentiles.iloc[0], vol_percentiles.iloc[1], np.inf],
                labels=[0, 1, 2]  # Low, Medium, High volatility
            ).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare final feature set
        """
        print("Preparing features...")
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
            'target', 'returns', 'log_returns', 'price_change'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with too many NaN values
        nan_threshold = 0.3
        feature_columns = [
            col for col in feature_columns 
            if df[col].isna().sum() / len(df) < nan_threshold
        ]
        
        self.feature_columns = feature_columns
        print(f"Selected {len(feature_columns)} features")
        
        return feature_columns
    
    def clean_data(self, df):
        """
        Clean data and handle missing values
        """
        print("Cleaning data...")
        
        # Remove rows with target = NaN
        df = df.dropna(subset=['target'])
        
        # Forward fill missing values for most features
        df = df.fillna(method='ffill')
        
        # Fill remaining NaN with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def preprocess_pipeline(self, period='3y', prediction_horizon=1, 
                          classification_type='direction'):
        """
        Complete preprocessing pipeline
        """
        print("Starting Nifty data preprocessing pipeline...")
        
        # Fetch data
        df = self.fetch_nifty_data(period)
        if df is None:
            return None, None, None
        
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        df = self.add_time_features(df)
        df = self.create_lag_features(df)
        
        # Create target
        df = self.create_target_variable(df, prediction_horizon, classification_type)
        
        # Prepare features
        feature_columns = self.prepare_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Split features and target
        X = df[feature_columns]
        y = df['target']
        
        # Remove rows where target is NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y, df
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_importance_names(self):
        """
        Get feature names for importance analysis
        """
        return self.feature_columns