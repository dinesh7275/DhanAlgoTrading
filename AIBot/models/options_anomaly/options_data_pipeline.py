"""
Nifty Options Data Pipeline
==========================

Data collection and preprocessing for Nifty 50 options
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class NiftyOptionsDataPipeline:
    """
    Data pipeline for Nifty 50 options analysis
    """
    
    def __init__(self):
        self.nifty_data = None
        self.options_data = {}
        self.processed_data = None
    
    def fetch_nifty_data(self, period='2y'):
        """Fetch Nifty 50 spot data"""
        print("Fetching Nifty 50 data...")
        
        try:
            nifty = yf.Ticker("^NSEI")
            self.nifty_data = nifty.history(period=period)
            
            # Calculate returns and basic features
            self.nifty_data['returns'] = self.nifty_data['Close'].pct_change()
            self.nifty_data['log_returns'] = np.log(self.nifty_data['Close'] / self.nifty_data['Close'].shift(1))
            
            # Volatility calculations
            for window in [5, 10, 20, 30]:
                self.nifty_data[f'volatility_{window}'] = (
                    self.nifty_data['returns'].rolling(window).std() * np.sqrt(252)
                )
            
            print(f"Nifty data fetched: {len(self.nifty_data)} rows")
            return self.nifty_data
            
        except Exception as e:
            print(f"Error fetching Nifty data: {e}")
            return None
    
    def fetch_options_chain(self, expiry_dates=None):
        """
        Fetch options chain data
        Note: In production, this would connect to NSE API or data provider
        """
        print("Note: Options chain fetching requires NSE API access")
        print("Using simulated options data for demonstration...")
        
        # Simulate options data structure
        if self.nifty_data is None:
            print("Fetch Nifty data first")
            return None
        
        current_price = self.nifty_data['Close'].iloc[-1]
        
        # Generate strike prices around current spot
        strikes = np.arange(
            int(current_price * 0.8 / 50) * 50,
            int(current_price * 1.2 / 50) * 50 + 50,
            50
        )
        
        # Simulate options data
        options_data = []
        
        for strike in strikes:
            # Simulate call and put prices (this would come from real API)
            moneyness = current_price / strike
            time_to_expiry = 30 / 365  # 30 days
            
            # Rough approximation for demo
            call_price = max(current_price - strike, 0) + 10 * (2 - abs(moneyness - 1))
            put_price = max(strike - current_price, 0) + 10 * (2 - abs(moneyness - 1))
            
            options_data.extend([
                {
                    'strike': strike,
                    'option_type': 'call',
                    'market_price': call_price,
                    'volume': np.random.randint(100, 10000),
                    'open_interest': np.random.randint(1000, 100000)
                },
                {
                    'strike': strike,
                    'option_type': 'put',
                    'market_price': put_price,
                    'volume': np.random.randint(100, 10000),
                    'open_interest': np.random.randint(1000, 100000)
                }
            ])
        
        self.options_data = pd.DataFrame(options_data)
        return self.options_data
    
    def calculate_option_features(self, risk_free_rate=0.06):
        """Calculate additional option features"""
        if self.options_data is None or len(self.options_data) == 0:
            print("No options data available")
            return None
        
        current_price = self.nifty_data['Close'].iloc[-1]
        time_to_expiry = 30 / 365  # Assuming 30-day expiry
        
        # Calculate moneyness
        self.options_data['moneyness'] = current_price / self.options_data['strike']
        self.options_data['log_moneyness'] = np.log(self.options_data['moneyness'])
        
        # Categorize options
        self.options_data['category'] = self.options_data['moneyness'].apply(
            lambda x: 'ITM' if (x > 1.02) else ('OTM' if x < 0.98 else 'ATM')
        )
        
        # Volume and Open Interest features
        self.options_data['voi_ratio'] = (
            self.options_data['volume'] / self.options_data['open_interest']
        ).fillna(0)
        
        # Put-Call ratio calculations
        call_data = self.options_data[self.options_data['option_type'] == 'call']
        put_data = self.options_data[self.options_data['option_type'] == 'put']
        
        # Aggregate metrics
        total_call_volume = call_data['volume'].sum()
        total_put_volume = put_data['volume'].sum()
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        total_call_oi = call_data['open_interest'].sum()
        total_put_oi = put_data['open_interest'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Store aggregate metrics
        self.aggregate_metrics = {
            'pcr_volume': pcr_volume,
            'pcr_oi': pcr_oi,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'current_spot': current_price
        }
        
        return self.options_data
    
    def create_training_dataset(self, lookback_window=20):
        """Create dataset for anomaly detection training"""
        if self.nifty_data is None:
            print("No Nifty data available")
            return None
        
        # Prepare features for each date
        features = []
        
        for i in range(lookback_window, len(self.nifty_data)):
            date = self.nifty_data.index[i]
            
            # Market features
            market_features = {
                'spot_price': self.nifty_data['Close'].iloc[i],
                'volume': self.nifty_data['Volume'].iloc[i],
                'volatility_5': self.nifty_data['volatility_5'].iloc[i],
                'volatility_10': self.nifty_data['volatility_10'].iloc[i],
                'volatility_20': self.nifty_data['volatility_20'].iloc[i],
                'returns_1d': self.nifty_data['returns'].iloc[i],
                'returns_5d': self.nifty_data['returns'].iloc[i-4:i+1].sum(),
                'date': date
            }
            
            # Add rolling statistics
            window_data = self.nifty_data.iloc[i-lookback_window:i]
            market_features.update({
                'price_trend': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0],
                'volume_trend': window_data['Volume'].rolling(5).mean().iloc[-1] / window_data['Volume'].mean(),
                'volatility_trend': window_data['volatility_10'].iloc[-1] / window_data['volatility_10'].mean()
            })
            
            features.append(market_features)
        
        self.processed_data = pd.DataFrame(features)
        return self.processed_data
    
    def get_data_summary(self):
        """Get summary of collected data"""
        summary = {}
        
        if self.nifty_data is not None:
            summary['nifty_data'] = {
                'shape': self.nifty_data.shape,
                'date_range': f"{self.nifty_data.index[0]} to {self.nifty_data.index[-1]}",
                'current_price': self.nifty_data['Close'].iloc[-1],
                'price_range': f"{self.nifty_data['Close'].min():.2f} - {self.nifty_data['Close'].max():.2f}"
            }
        
        if hasattr(self, 'options_data') and self.options_data is not None:
            summary['options_data'] = {
                'shape': self.options_data.shape,
                'strikes_range': f"{self.options_data['strike'].min()} - {self.options_data['strike'].max()}",
                'total_volume': self.options_data['volume'].sum(),
                'total_oi': self.options_data['open_interest'].sum()
            }
        
        if hasattr(self, 'aggregate_metrics'):
            summary['aggregate_metrics'] = self.aggregate_metrics
        
        return summary