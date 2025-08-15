# macro_features.py
# Macroeconomic and Calendar Features for Indian Markets

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MacroFeatures:
    """
    Engineer macroeconomic and calendar-based features for Indian markets
    """
    
    def __init__(self):
        pass
    
    def create_calendar_features(self, market_data):
        """
        Create calendar-based features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Basic calendar features
        features['Year'] = features.index.year
        features['Month'] = features.index.month
        features['Quarter'] = features.index.quarter
        features['Day_of_Week'] = features.index.dayofweek
        features['Day_of_Month'] = features.index.day
        features['Week_of_Year'] = features.index.isocalendar().week
        
        # Cyclical encoding for time features (preserves periodicity)
        features['Month_Sin'] = np.sin(2 * np.pi * features['Month'] / 12)
        features['Month_Cos'] = np.cos(2 * np.pi * features['Month'] / 12)
        features['Day_of_Week_Sin'] = np.sin(2 * np.pi * features['Day_of_Week'] / 7)
        features['Day_of_Week_Cos'] = np.cos(2 * np.pi * features['Day_of_Week'] / 7)
        
        return features
    
    def create_indian_market_timing_features(self, market_data):
        """
        Create Indian market specific timing features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Indian market specific timing
        features['Is_Month_End'] = (features.index.day >= 25).astype(int)
        features['Is_Quarter_End'] = ((features.index.month % 3 == 0) & (features.index.day >= 25)).astype(int)
        features['Is_Year_End'] = ((features.index.month == 12) & (features.index.day >= 25)).astype(int)
        features['Is_Financial_Year_End'] = ((features.index.month == 3) & (features.index.day >= 25)).astype(int)
        
        # Options expiry timing (Nifty options expire on Thursdays)
        features['Days_to_Thursday'] = (3 - features.index.dayofweek) % 7
        features['Is_Expiry_Week'] = (features['Days_to_Thursday'] <= 3).astype(int)
        features['Is_Expiry_Day'] = (features.index.dayofweek == 3).astype(int)  # Thursday
        features['Is_Pre_Expiry'] = (features.index.dayofweek == 2).astype(int)  # Wednesday
        
        # Settlement patterns
        features['Is_Settlement_Week'] = features['Is_Expiry_Week']
        features['Days_Since_Expiry'] = (features.index.dayofweek - 3) % 7
        
        return features
    
    def create_seasonal_features(self, market_data):
        """
        Create seasonal pattern features specific to Indian markets
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Indian festival seasons (typically bullish periods)
        features['Is_Festival_Season'] = ((features.index.month >= 9) & (features.index.month <= 11)).astype(int)
        features['Is_Diwali_Season'] = ((features.index.month == 10) | (features.index.month == 11)).astype(int)
        features['Is_Wedding_Season'] = ((features.index.month >= 10) & (features.index.month <= 12)).astype(int)
        
        # Budget and policy seasons
        features['Is_Budget_Season'] = ((features.index.month == 2) | (features.index.month == 7)).astype(int)
        features['Is_Pre_Budget'] = (features.index.month == 1).astype(int)
        features['Is_Post_Budget'] = (features.index.month == 3).astype(int)
        
        # Earnings seasons (quarterly results)
        features['Is_Results_Season'] = ((features.index.month % 3 == 1) | (features.index.month % 3 == 2)).astype(int)
        features['Is_Q1_Results'] = (features.index.month == 4).astype(int)  # Apr-Jun results
        features['Is_Q2_Results'] = (features.index.month == 7).astype(int)  # Jul-Sep results
        features['Is_Q3_Results'] = (features.index.month == 10).astype(int)  # Oct-Dec results
        features['Is_Q4_Results'] = (features.index.month == 1).astype(int)  # Jan-Mar results
        
        # Monsoon season (affects agriculture-dependent sectors)
        features['Is_Monsoon_Season'] = ((features.index.month >= 6) & (features.index.month <= 9)).astype(int)
        features['Is_Pre_Monsoon'] = (features.index.month == 5).astype(int)
        features['Is_Post_Monsoon'] = (features.index.month == 10).astype(int)
        
        return features
    
    def create_trading_session_features(self, market_data):
        """
        Create trading session pattern features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Day of week effects
        features['Is_Monday'] = (features.index.dayofweek == 0).astype(int)
        features['Is_Tuesday'] = (features.index.dayofweek == 1).astype(int)
        features['Is_Wednesday'] = (features.index.dayofweek == 2).astype(int)
        features['Is_Thursday'] = (features.index.dayofweek == 3).astype(int)
        features['Is_Friday'] = (features.index.dayofweek == 4).astype(int)
        
        # Week patterns
        features['Is_Week_Start'] = (features.index.dayofweek == 0).astype(int)
        features['Is_Week_End'] = (features.index.dayofweek == 4).astype(int)
        features['Is_Mid_Week'] = ((features.index.dayofweek >= 1) & (features.index.dayofweek <= 3)).astype(int)
        
        # Month patterns
        features['Is_Month_Start'] = (features.index.day <= 5).astype(int)
        features['Is_Month_Mid'] = ((features.index.day >= 10) & (features.index.day <= 20)).astype(int)
        features['Is_Month_End_Week'] = (features.index.day >= 20).astype(int)
        
        return features
    
    def create_currency_features(self, market_data):
        """
        Create USD/INR and currency-related features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Synthetic USD/INR data (replace with real data in production)
        np.random.seed(42)  # For reproducible results
        base_rate = 75
        random_walk = np.cumsum(np.random.normal(0, 0.3, len(market_data))) / 100
        
        features['USD_INR'] = base_rate + random_walk
        features['USD_INR_Change'] = features['USD_INR'].pct_change()
        features['USD_INR_MA_5'] = features['USD_INR'].rolling(5).mean()
        features['USD_INR_MA_20'] = features['USD_INR'].rolling(20).mean()
        features['USD_INR_MA_60'] = features['USD_INR'].rolling(60).mean()
        
        # Currency strength indicators
        features['USD_INR_Strength'] = features['USD_INR'] / features['USD_INR_MA_20'] - 1
        features['USD_INR_Trend'] = features['USD_INR_MA_5'] / features['USD_INR_MA_20'] - 1
        features['USD_INR_Volatility'] = features['USD_INR_Change'].rolling(20).std() * np.sqrt(252)
        
        # FII flow proxy (inverse relationship with USD/INR)
        features['FII_Flow_Proxy'] = -features['USD_INR_Change']
        features['FII_Sentiment'] = features['FII_Flow_Proxy'].rolling(5).sum()
        features['FII_Trend'] = features['FII_Sentiment'].rolling(10).mean()
        
        # Currency regime indicators
        features['USD_INR_Weak'] = (features['USD_INR_Strength'] < -0.02).astype(int)
        features['USD_INR_Strong'] = (features['USD_INR_Strength'] > 0.02).astype(int)
        features['USD_INR_Stable'] = ((abs(features['USD_INR_Strength']) <= 0.02)).astype(int)
        
        return features
    
    def create_global_risk_features(self, market_data):
        """
        Create global risk-on/risk-off indicators
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Seasonal risk patterns
        features['Global_Risk_On'] = (features.index.month.isin([11, 12, 1])).astype(int)  # Year-end rally
        features['Global_Risk_Off'] = (features.index.month.isin([3, 9])).astype(int)  # Typical risk-off periods
        features['Summer_Doldrums'] = (features.index.month.isin([6, 7, 8])).astype(int)  # Summer trading
        
        # Economic cycle indicators
        features['Business_Cycle_Phase'] = ((features.index.quarter - 1) % 4) / 3  # Normalized 0-1
        features['Calendar_Year_Progress'] = features.index.dayofyear / 365
        
        # Holiday proximity effects (simplified)
        features['Near_Major_Holiday'] = 0  # Would need holiday calendar in production
        features['Post_Holiday_Effect'] = 0  # Would need holiday calendar in production
        
        return features
    
    def create_policy_cycle_features(self, market_data):
        """
        Create policy and regulatory cycle features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # RBI policy meeting cycles (typically every 2 months)
        # Simplified - would need actual RBI calendar in production
        features['Is_RBI_Policy_Month'] = (features.index.month % 2 == 0).astype(int)
        features['Pre_RBI_Policy'] = ((features.index.month % 2 == 0) & (features.index.day <= 15)).astype(int)
        features['Post_RBI_Policy'] = ((features.index.month % 2 == 0) & (features.index.day > 15)).astype(int)
        
        # Government policy cycles
        features['Election_Year'] = 0  # Would need election calendar
        features['Policy_Announcement_Season'] = features['Is_Budget_Season']
        
        # Regulatory calendar
        features['Tax_Season'] = ((features.index.month >= 1) & (features.index.month <= 3)).astype(int)
        features['Corporate_Action_Season'] = ((features.index.month >= 3) & (features.index.month <= 6)).astype(int)
        
        return features
    
    def create_all_macro_features(self, market_data):
        """
        Create comprehensive macroeconomic and calendar features
        """
        print("Creating comprehensive macroeconomic features...")
        
        # Create each category of features
        calendar_features = self.create_calendar_features(market_data)
        timing_features = self.create_indian_market_timing_features(market_data)
        seasonal_features = self.create_seasonal_features(market_data)
        session_features = self.create_trading_session_features(market_data)
        currency_features = self.create_currency_features(market_data)
        global_features = self.create_global_risk_features(market_data)
        policy_features = self.create_policy_cycle_features(market_data)
        
        # Combine all macro features
        all_features = pd.concat([
            calendar_features,
            timing_features,
            seasonal_features,
            session_features,
            currency_features,
            global_features,
            policy_features
        ], axis=1)
        
        print(f"Generated {len(all_features.columns)} macroeconomic features")
        return all_features

if __name__ == "__main__":
    # Test macro features
    import yfinance as yf
    
    # Get sample data
    nifty = yf.Ticker('^NSEI')
    df = nifty.history(period='1y')
    
    # Calculate macro features
    macro_calc = MacroFeatures()
    features = macro_calc.create_all_macro_features(df)
    
    print(f"Macro features shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Sample features: {features.columns[:10].tolist()}")
    
    # Check for missing values
    missing_pct = features.isnull().mean() * 100
    print(f"Average missing values: {missing_pct.mean():.2f}%")
    
    # Show some interesting features
    print(f"\nSample macro features:")
    recent_data = features.tail(5)
    interesting_cols = ['Is_Expiry_Week', 'Is_Festival_Season', 'USD_INR_Strength', 'FII_Sentiment']
    available_cols = [col for col in interesting_cols if col in features.columns]
    if available_cols:
        print(recent_data[available_cols])

        