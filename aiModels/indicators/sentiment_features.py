# sentiment_features.py
# Market Sentiment and Alternative Data Features

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SentimentFeatures:
    """
    Engineer sentiment and behavioral finance features
    """
    
    def __init__(self):
        pass
    
    def create_vix_sentiment_features(self, vix_data, market_data):
        """
        Create VIX-based sentiment features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Align VIX data with market data
        vix_close = vix_data['Close'].reindex(market_data.index, method='ffill')
        
        # Basic VIX features
        features['VIX_Level'] = vix_close
        features['VIX_Change'] = vix_close.pct_change()
        features['VIX_MA_10'] = vix_close.rolling(10).mean()
        features['VIX_MA_20'] = vix_close.rolling(20).mean()
        features['VIX_Relative'] = vix_close / features['VIX_MA_20'] - 1
        
        # VIX percentiles (fear/greed indicator)
        features['VIX_Percentile_60d'] = vix_close.rolling(60).rank(pct=True)
        features['VIX_Percentile_252d'] = vix_close.rolling(252).rank(pct=True)
        features['Fear_Greed_Index'] = 100 - features['VIX_Percentile_252d'] * 100
        
        # VIX patterns
        features['VIX_Spike'] = (features['VIX_Change'] > 0.15).astype(int)
        features['VIX_Crush'] = (features['VIX_Change'] < -0.15).astype(int)
        features['VIX_Contango'] = (vix_close < features['VIX_MA_10']).astype(int)
        features['VIX_Backwardation'] = (vix_close > features['VIX_MA_10']).astype(int)
        
        return features
    
    def create_momentum_sentiment_features(self, market_data):
        """
        Create momentum-based sentiment features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Multi-timeframe momentum
        returns_1d = market_data['Close'].pct_change()
        returns_5d = market_data['Close'].pct_change(5)
        returns_20d = market_data['Close'].pct_change(20)
        returns_60d = market_data['Close'].pct_change(60)
        
        features['Momentum_1D'] = returns_1d
        features['Momentum_5D'] = returns_5d
        features['Momentum_20D'] = returns_20d
        features['Momentum_60D'] = returns_60d
        
        # Momentum ratios and relationships
        features['Momentum_5D_vs_20D'] = returns_5d / (returns_20d.rolling(60).std() + 1e-6)
        features['Momentum_Acceleration'] = returns_5d - returns_20d
        features['Momentum_Consistency'] = (returns_1d > 0).rolling(20).mean()
        
        # Momentum extremes
        features['Momentum_Extreme_Up'] = (returns_5d > returns_5d.rolling(252).quantile(0.9)).astype(int)
        features['Momentum_Extreme_Down'] = (returns_5d < returns_5d.rolling(252).quantile(0.1)).astype(int)
        
        return features
    
    def create_market_breadth_features(self, market_data_dict):
        """
        Create market breadth features from multiple indices
        """
        nifty_data = market_data_dict.get('nifty_50')
        features = pd.DataFrame(index=nifty_data.index)
        
        if len(market_data_dict) > 1:
            # Calculate returns for all indices
            indices_returns = pd.DataFrame()
            for name, data in market_data_dict.items():
                if name != 'nifty_50' and isinstance(data, pd.DataFrame):
                    indices_returns[name] = data['Close'].pct_change()
            
            if len(indices_returns.columns) > 0:
                # Market breadth indicators
                features['Market_Breadth'] = (indices_returns > 0).sum(axis=1) / len(indices_returns.columns)
                features['Advance_Decline_Ratio'] = features['Market_Breadth'] / (1 - features['Market_Breadth'] + 0.001)
                
                # Breadth momentum
                features['Breadth_Momentum'] = features['Market_Breadth'].rolling(5).mean()
                features['Breadth_Divergence'] = features['Market_Breadth'] - features['Market_Breadth'].rolling(20).mean()
                
                # Extreme breadth conditions
                features['Breadth_Extreme_Positive'] = (features['Market_Breadth'] > 0.8).astype(int)
                features['Breadth_Extreme_Negative'] = (features['Market_Breadth'] < 0.2).astype(int)
        else:
            # Default values if only one index available
            features['Market_Breadth'] = 0.5
            features['Advance_Decline_Ratio'] = 1.0
            features['Breadth_Momentum'] = 0.5
            features['Breadth_Divergence'] = 0.0
            features['Breadth_Extreme_Positive'] = 0
            features['Breadth_Extreme_Negative'] = 0
        
        return features
    
    def create_contrarian_indicators(self, sentiment_features):
        """
        Create contrarian sentiment indicators
        """
        features = pd.DataFrame(index=sentiment_features.index)
        
        # Contrarian signals based on fear/greed extremes
        fear_greed = sentiment_features.get('Fear_Greed_Index', pd.Series(50, index=sentiment_features.index))
        
        features['Contrarian_Signal'] = ((fear_greed < 20) | (fear_greed > 80)).astype(int)
        features['Extreme_Fear'] = (fear_greed < 10).astype(int)
        features['Extreme_Greed'] = (fear_greed > 90).astype(int)
        
        # Oversold/Overbought reversals
        momentum_5d = sentiment_features.get('Momentum_5D', pd.Series(0, index=sentiment_features.index))
        
        features['Oversold_Rebound'] = ((fear_greed < 20) & 
                                       (momentum_5d > momentum_5d.shift(1))).astype(int)
        features['Overbought_Reversal'] = ((fear_greed > 80) & 
                                          (momentum_5d < momentum_5d.shift(1))).astype(int)
        
        # Sentiment momentum vs price momentum divergence
        sentiment_momentum = fear_greed.rolling(5).mean()
        price_momentum = momentum_5d.rolling(5).mean()
        
        features['Sentiment_Price_Divergence'] = (
            (sentiment_momentum > sentiment_momentum.shift(5)) & 
            (price_momentum < price_momentum.shift(5))
        ).astype(int)
        
        return features
    
    def create_options_sentiment_features(self, options_data):
        """
        Create options-based sentiment features
        """
        if options_data is None or len(options_data) == 0:
            return pd.DataFrame()
        
        # Group options by date
        daily_options_sentiment = []
        
        for date in options_data['date'].unique() if 'date' in options_data.columns else []:
            daily_options = options_data[options_data['date'] == date]
            
            if len(daily_options) == 0:
                continue
            
            # Separate calls and puts
            calls = daily_options[daily_options['option_type'] == 'CE']
            puts = daily_options[daily_options['option_type'] == 'PE']
            
            sentiment_dict = {'date': date}
            
            # Put-Call ratios (sentiment indicators)
            call_volume = calls['volume'].sum() if len(calls) > 0 else 1
            put_volume = puts['volume'].sum() if len(puts) > 0 else 1
            call_oi = calls['open_interest'].sum() if len(calls) > 0 else 1
            put_oi = puts['open_interest'].sum() if len(puts) > 0 else 1
            
            sentiment_dict['PCR_Volume'] = put_volume / call_volume
            sentiment_dict['PCR_OI'] = put_oi / call_oi
            
            # Volatility skew (fear indicator)
            if len(calls) > 0 and len(puts) > 0:
                call_iv = calls['implied_vol'].mean()
                put_iv = puts['implied_vol'].mean()
                sentiment_dict['Vol_Skew'] = put_iv - call_iv
            else:
                sentiment_dict['Vol_Skew'] = 0
            
            # Options activity patterns
            total_volume = daily_options['volume'].sum()
            total_oi = daily_options['open_interest'].sum()
            
            # OTM activity (speculative behavior)
            spot_price = daily_options['underlying_price'].iloc[0]
            otm_calls = calls[calls['strike'] > spot_price * 1.05]
            otm_puts = puts[puts['strike'] < spot_price * 0.95]
            
            otm_volume = otm_calls['volume'].sum() + otm_puts['volume'].sum()
            sentiment_dict['OTM_Activity_Ratio'] = otm_volume / (total_volume + 1)
            
            daily_options_sentiment.append(sentiment_dict)
        
        if daily_options_sentiment:
            options_sentiment = pd.DataFrame(daily_options_sentiment)
            options_sentiment.set_index('date', inplace=True)
            
            # Rolling averages for smoothing
            options_sentiment['PCR_Volume_MA'] = options_sentiment['PCR_Volume'].rolling(5).mean()
            options_sentiment['Vol_Skew_MA'] = options_sentiment['Vol_Skew'].rolling(10).mean()
            
            return options_sentiment
        else:
            return pd.DataFrame()
    
    def create_news_sentiment_features(self, market_data):
        """
        Create simulated news sentiment features
        In production, integrate with real news APIs
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Simulated news sentiment (replace with real news sentiment in production)
        np.random.seed(42)  # For reproducible results
        features['News_Sentiment'] = np.random.normal(0, 0.1, len(market_data))
        features['News_Sentiment'] = features['News_Sentiment'].rolling(5).mean()
        
        # News sentiment momentum
        features['News_Sentiment_MA'] = features['News_Sentiment'].rolling(5).mean()
        features['News_Sentiment_Momentum'] = features['News_Sentiment'] - features['News_Sentiment'].shift(1)
        
        # News sentiment extremes
        features['News_Sentiment_Extreme'] = (abs(features['News_Sentiment']) > 0.15).astype(int)
        features['News_Sentiment_Positive'] = (features['News_Sentiment'] > 0.1).astype(int)
        features['News_Sentiment_Negative'] = (features['News_Sentiment'] < -0.1).astype(int)
        
        return features
    
    def create_all_sentiment_features(self, market_data_dict, vix_data, options_data=None):
        """
        Create comprehensive sentiment features
        """
        print("Creating comprehensive sentiment features...")
        
        nifty_data = market_data_dict['nifty_50']
        
        # Create each category of sentiment features
        vix_sentiment = self.create_vix_sentiment_features(vix_data, nifty_data)
        momentum_sentiment = self.create_momentum_sentiment_features(nifty_data)
        breadth_features = self.create_market_breadth_features(market_data_dict)
        news_features = self.create_news_sentiment_features(nifty_data)
        
        # Combine core sentiment features first
        combined_sentiment = pd.concat([
            vix_sentiment,
            momentum_sentiment,
            breadth_features,
            news_features
        ], axis=1)
        
        # Add contrarian indicators based on combined sentiment
        contrarian_features = self.create_contrarian_indicators(combined_sentiment)
        
        # Add options sentiment if available
        if options_data is not None and len(options_data) > 0:
            options_sentiment = self.create_options_sentiment_features(options_data)
            if len(options_sentiment) > 0:
                # Reindex to match market data
                options_sentiment = options_sentiment.reindex(nifty_data.index, method='ffill')
                combined_sentiment = pd.concat([combined_sentiment, options_sentiment], axis=1)
        
        # Add contrarian features
        combined_sentiment = pd.concat([combined_sentiment, contrarian_features], axis=1)
        
        print(f"Generated {len(combined_sentiment.columns)} sentiment features")
        return combined_sentiment

if __name__ == "__main__":
    # Test sentiment features
    import yfinance as yf
    
    # Get sample data
    nifty = yf.Ticker('^NSEI')
    nifty_data = nifty.history(period='1y')
    
    vix = yf.Ticker('^INDIAVIX')
    vix_data = vix.history(period='1y')
    
    market_data_dict = {'nifty_50': nifty_data}
    
    # Calculate sentiment features
    sentiment_calc = SentimentFeatures()
    features = sentiment_calc.create_all_sentiment_features(market_data_dict, vix_data)
    
    print(f"Sentiment features shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Sample features: {features.columns[:10].tolist()}")
    
    # Check for missing values
    missing_pct = features.isnull().mean() * 100
    print(f"Average missing values: {missing_pct.mean():.2f}%")
