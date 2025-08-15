# price_volume_features.py
# Price and Volume Feature Engineering

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class PriceVolumeFeatures:
    """
    Engineer price and volume features for trading models
    """
    
    def __init__(self):
        pass
    
    def create_basic_features(self, df):
        """
        Create basic price and volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic OHLCV
        features['Close'] = df['Close']
        features['Open'] = df['Open']
        features['High'] = df['High']
        features['Low'] = df['Low']
        features['Volume'] = df['Volume']
        
        # Price transformations
        features['Log_Close'] = np.log(features['Close'])
        features['Close_Normalized'] = features['Close'] / features['Close'].iloc[0]
        
        return features
    
    def create_returns_features(self, df):
        """
        Create return-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns (multiple timeframes)
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'Return_{period}d'] = df['Close'].pct_change(period)
            features[f'Log_Return_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Return statistics
        for window in [5, 10, 20]:
            features[f'Return_Mean_{window}d'] = features['Return_1d'].rolling(window).mean()
            features[f'Return_Std_{window}d'] = features['Return_1d'].rolling(window).std()
            features[f'Return_Skew_{window}d'] = features['Return_1d'].rolling(window).skew()
            features[f'Return_Kurt_{window}d'] = features['Return_1d'].rolling(window).kurt()
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'Volatility_{window}d'] = features['Return_1d'].rolling(window).std() * np.sqrt(252)
        
        return features
    
    def create_price_patterns(self, df):
        """
        Create price pattern features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic ratios
        features['High_Low_Ratio'] = df['High'] / df['Low']
        features['Close_Open_Ratio'] = df['Close'] / df['Open']
        features['High_Close_Ratio'] = df['High'] / df['Close']
        features['Low_Close_Ratio'] = df['Low'] / df['Close']
        
        # Candlestick components
        features['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        features['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        features['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        features['Body_Direction'] = np.sign(df['Close'] - df['Open'])
        
        # Position within day's range
        features['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        features['Open_Position'] = (df['Open'] - df['Low']) / (df['High'] - df['Low'])
        
        # Price gaps
        features['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        features['Gap_Up'] = np.where(features['Gap'] > 0, features['Gap'], 0)
        features['Gap_Down'] = np.where(features['Gap'] < 0, abs(features['Gap']), 0)
        
        return features
    
    def create_volume_features(self, df):
        """
        Create volume-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Volume moving averages
        features['Volume_MA_5'] = df['Volume'].rolling(5).mean()
        features['Volume_MA_10'] = df['Volume'].rolling(10).mean()
        features['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        
        # Volume ratios
        features['Volume_Ratio_5'] = df['Volume'] / features['Volume_MA_5']
        features['Volume_Ratio_10'] = df['Volume'] / features['Volume_MA_10']
        features['Volume_Ratio_20'] = df['Volume'] / features['Volume_MA_20']
        
        # Volume trends
        features['Volume_Trend_5'] = features['Volume_MA_5'] / features['Volume_MA_20']
        features['Volume_Slope'] = df['Volume'].pct_change(5)
        
        # Volume-price relationship
        features['Volume_Price_Trend'] = df['Volume'] * df['Close'].pct_change()
        features['Volume_Weighted_Price'] = (df['Volume'] * df['Close']).rolling(5).sum() / df['Volume'].rolling(5).sum()
        
        # Volume surprises
        features['Volume_Surprise'] = (df['Volume'] - features['Volume_MA_20']) / features['Volume_MA_20']
        features['Volume_Spike'] = (features['Volume_Ratio_20'] > 2).astype(int)
        
        return features
    
    def create_true_range_features(self, df):
        """
        Create True Range and related features
        """
        features = pd.DataFrame(index=df.index)
        
        # True Range calculation
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        
        features['True_Range'] = np.maximum(high_low, np.maximum(high_close, low_close))
        features['True_Range_Pct'] = features['True_Range'] / df['Close']
        
        # ATR variations
        for window in [5, 10, 14, 20]:
            features[f'ATR_{window}'] = features['True_Range'].rolling(window).mean()
            features[f'ATR_{window}_Ratio'] = features[f'ATR_{window}'] / df['Close']
        
        # Range expansion/contraction
        features['Range_Expansion'] = features['ATR_14'] / features['ATR_14'].rolling(20).mean()
        features['Range_Contraction'] = features['ATR_14'].rolling(20).min() / features['ATR_14']
        
        return features
    
    def create_all_price_volume_features(self, df):
        """
        Create all price and volume features
        """
        print("Creating comprehensive price and volume features...")
        
        # Create each category of features
        basic_features = self.create_basic_features(df)
        returns_features = self.create_returns_features(df)
        pattern_features = self.create_price_patterns(df)
        volume_features = self.create_volume_features(df)
        tr_features = self.create_true_range_features(df)
        
        # Combine all features
        all_features = pd.concat([
            basic_features,
            returns_features,
            pattern_features,
            volume_features,
            tr_features
        ], axis=1)
        
        print(f"Generated {len(all_features.columns)} price and volume features")
        return all_features

if __name__ == "__main__":
    # Test price volume features
    import yfinance as yf
    
    # Get sample data
    nifty = yf.Ticker('^NSEI')
    df = nifty.history(period='1y')
    
    # Calculate features
    pv_features = PriceVolumeFeatures()
    features = pv_features.create_all_price_volume_features(df)
    
    print(f"Price/Volume features shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Sample features: {features.columns[:10].tolist()}")
    
    # Check for missing values
    missing_pct = features.isnull().mean() * 100
    print(f"Average missing values: {missing_pct.mean():.2f}%")
