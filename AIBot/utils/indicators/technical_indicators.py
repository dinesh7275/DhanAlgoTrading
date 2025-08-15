# technical_indicators.py
# Technical Analysis Indicators for Nifty 50

import numpy as np
import pandas as pd
import ta
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Calculate comprehensive technical indicators
    """
    
    def __init__(self):
        pass
    
    def calculate_moving_averages(self, df):
        """
        Calculate various moving averages
        """
        indicators = pd.DataFrame(index=df.index)
        
        # Key EMAs for options trading
        indicators['EMA_6'] = ta.trend.ema_indicator(df['Close'], window=6)
        indicators['EMA_15'] = ta.trend.ema_indicator(df['Close'], window=15)
        
        # Additional EMAs (including 12 and 26 for MACD)
        for period in [5, 10, 12, 20, 26, 50]:
            indicators[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            indicators[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # Price relative to moving averages
        indicators['Price_to_SMA_20'] = df['Close'] / indicators['SMA_20'] - 1
        indicators['Price_to_SMA_50'] = df['Close'] / indicators['SMA_50'] - 1
        indicators['Price_to_EMA_20'] = df['Close'] / indicators['EMA_20'] - 1
        
        # Moving average slopes (trend strength)
        for period in [5, 10, 20, 50]:
            indicators[f'SMA_{period}_Slope'] = indicators[f'SMA_{period}'].pct_change(5)
            indicators[f'EMA_{period}_Slope'] = indicators[f'EMA_{period}'].pct_change(5)
        
        # Key crossovers for options trading
        indicators['EMA_6_15_Cross'] = (indicators['EMA_6'] > indicators['EMA_15']).astype(int)
        indicators['EMA_6_15_Signal'] = np.where(indicators['EMA_6'] > indicators['EMA_15'], 1, -1)
        
        # Traditional crossovers
        indicators['SMA_5_20_Cross'] = (indicators['SMA_5'] > indicators['SMA_20']).astype(int)
        indicators['EMA_12_26_Cross'] = (indicators['EMA_12'] > indicators['EMA_26']).astype(int)
        
        return indicators
    
    def calculate_momentum_indicators(self, df):
        """
        Calculate momentum oscillators
        """
        indicators = pd.DataFrame(index=df.index)
        
        # RSI
        indicators['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        indicators['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)
        indicators['RSI_9'] = ta.momentum.rsi(df['Close'], window=9)
        
        # RSI patterns
        indicators['RSI_Overbought'] = (indicators['RSI_14'] > 70).astype(int)
        indicators['RSI_Oversold'] = (indicators['RSI_14'] < 30).astype(int)
        indicators['RSI_Divergence'] = indicators['RSI_14'] - indicators['RSI_14'].shift(5)
        indicators['RSI_Momentum'] = indicators['RSI_14'].pct_change(3)
        
        # MACD
        indicators['MACD'] = ta.trend.macd_diff(df['Close'])
        indicators['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        indicators['MACD_Signal_Cross'] = (indicators['MACD'] > indicators['MACD_Signal']).astype(int)
        indicators['MACD_Zero_Cross'] = (indicators['MACD'] > 0).astype(int)
        
        # Stochastic
        indicators['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        indicators['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        indicators['Stoch_Cross'] = (indicators['Stoch_K'] > indicators['Stoch_D']).astype(int)
        indicators['Stoch_Overbought'] = (indicators['Stoch_K'] > 80).astype(int)
        indicators['Stoch_Oversold'] = (indicators['Stoch_K'] < 20).astype(int)
        
        # Williams %R
        indicators['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        indicators['Williams_R_Signal'] = (indicators['Williams_R'] > -50).astype(int)
        
        return indicators
    
    def calculate_trend_indicators(self, df):
        """
        Calculate trend strength indicators
        """
        indicators = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        indicators['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        indicators['ADX_Pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
        indicators['ADX_Neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
        indicators['ADX_Trend_Strength'] = (indicators['ADX'] > 25).astype(int)
        
        # CCI (Commodity Channel Index)
        indicators['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Parabolic SAR
        indicators['PSAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        
        return indicators
    
    def calculate_volatility_indicators(self, df):
        """
        Calculate volatility-based indicators
        """
        indicators = pd.DataFrame(index=df.index)
        
        # Bollinger Bands
        indicators['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        indicators['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
        indicators['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / indicators['BB_Middle']
        indicators['BB_Position'] = (df['Close'] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
        indicators['BB_Squeeze'] = (indicators['BB_Width'] < indicators['BB_Width'].rolling(20).quantile(0.1)).astype(int)
        
        # ATR (Average True Range)
        indicators['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        indicators['ATR_20'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=20)
        indicators['ATR_Ratio'] = indicators['ATR_14'] / df['Close']
        indicators['ATR_Expansion'] = indicators['ATR_14'] / indicators['ATR_14'].rolling(20).mean()
        
        # Keltner Channels
        indicators['Keltner_Upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
        indicators['Keltner_Middle'] = ta.volatility.keltner_channel_mband(df['High'], df['Low'], df['Close'])
        indicators['Keltner_Lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
        
        return indicators
    
    def calculate_volume_indicators(self, df):
        """
        Calculate volume-based indicators
        """
        indicators = pd.DataFrame(index=df.index)
        
        # On-Balance Volume
        indicators['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        indicators['OBV_Slope'] = indicators['OBV'].pct_change(5)
        
        # Volume moving averages
        indicators['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        indicators['Volume_Ratio'] = df['Volume'] / indicators['Volume_SMA_20']
        
        # Chaikin Money Flow
        indicators['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Money Flow Index
        indicators['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        indicators['AD_Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Price Trend
        indicators['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        
        return indicators
    
    def calculate_all_indicators(self, df):
        """
        Calculate all technical indicators
        """
        print("Calculating comprehensive technical indicators...")
        
        # Calculate each category
        ma_indicators = self.calculate_moving_averages(df)
        momentum_indicators = self.calculate_momentum_indicators(df)
        trend_indicators = self.calculate_trend_indicators(df)
        volatility_indicators = self.calculate_volatility_indicators(df)
        volume_indicators = self.calculate_volume_indicators(df)
        
        # Combine all indicators
        all_indicators = pd.concat([
            ma_indicators,
            momentum_indicators,
            trend_indicators,
            volatility_indicators,
            volume_indicators
        ], axis=1)
        
        print(f"Generated {len(all_indicators.columns)} technical indicators")
        return all_indicators

if __name__ == "__main__":
    # Test technical indicators
    import yfinance as yf
    
    # Get sample data
    nifty = yf.Ticker('^NSEI')
    df = nifty.history(period='1y')
    
    # Calculate indicators
    tech_calc = TechnicalIndicators()
    indicators = tech_calc.calculate_all_indicators(df)
    
    print(f"Technical indicators shape: {indicators.shape}")
    print(f"Date range: {indicators.index.min()} to {indicators.index.max()}")
    print(f"Sample indicators: {indicators.columns[:10].tolist()}")
    
    # Check for missing values
    missing_pct = indicators.isnull().mean() * 100
    print(f"Average missing values: {missing_pct.mean():.2f}%")
