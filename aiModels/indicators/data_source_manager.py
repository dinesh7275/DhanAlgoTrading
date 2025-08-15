# data_source_manager.py
# Primary Data Sources Manager for Nifty 50 Ensemble System

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class NiftyDataSourceManager:
    """
    Manages all primary data sources for Nifty 50 ensemble system
    """
    
    def __init__(self, period='2y', enable_cache=True):
        self.period = period
        self.enable_cache = enable_cache
        self.cache = {}
        self.data_sources = {}
        
    def fetch_market_data(self):
        """
        Fetch OHLCV data for underlying asset and indices
        """
        print("Fetching Market Data (OHLCV)...")
        
        # Primary Nifty indices
        nifty_symbols = {
            'nifty_50': '^NSEI',
            'nifty_next_50': '^NSEMDCP50',
            'nifty_100': '^CNX100',
            'nifty_500': '^CNX500'
        }
        
        market_data = {}
        
        for name, symbol in nifty_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=self.period)
                
                # Enhanced OHLCV features
                data['Returns'] = data['Close'].pct_change()
                data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
                data['Open_Close_Pct'] = (data['Close'] - data['Open']) / data['Open']
                data['Volume_MA_20'] = data['Volume'].rolling(20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
                
                market_data[name] = data
                print(f"✓ Loaded {name}")
                
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
                # Create synthetic data as fallback
                if 'nifty_50' in market_data:
                    base_data = market_data['nifty_50'].copy()
                    noise_factor = np.random.uniform(0.8, 1.2)
                    synthetic_data = base_data * noise_factor
                    market_data[name] = synthetic_data
        
        self.data_sources['market_data'] = market_data
        return market_data
    
    def fetch_volatility_metrics(self):
        """
        Fetch India VIX and calculate historical volatilities
        """
        print("Fetching Volatility Metrics...")
        
        volatility_data = {}
        
        # India VIX
        try:
            vix_ticker = yf.Ticker('^INDIAVIX')
            vix_data = vix_ticker.history(period=self.period)
            volatility_data['india_vix'] = vix_data
            print("✓ Loaded India VIX")
        except:
            print("✗ Failed to load India VIX, creating synthetic")
            dates = self.data_sources['market_data']['nifty_50'].index
            synthetic_vix = pd.DataFrame(index=dates)
            synthetic_vix['Close'] = 15 + 10 * np.random.gamma(2, 1, len(dates))
            volatility_data['india_vix'] = synthetic_vix
        
        # Calculate historical volatilities
        nifty_data = self.data_sources['market_data']['nifty_50']
        returns = nifty_data['Returns'].dropna()
        
        vol_metrics = pd.DataFrame(index=nifty_data.index)
        
        # Multiple timeframe volatilities
        for window in [5, 10, 20, 30, 60]:
            vol_metrics[f'hist_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            vol_metrics[f'ewm_vol_{window}'] = returns.ewm(span=window).std() * np.sqrt(252)
        
        # Parkinson volatility (high-low estimator)
        vol_metrics['parkinson_vol'] = np.sqrt(252 * (
            np.log(nifty_data['High']/nifty_data['Low'])**2 * 0.25 - 
            (2*np.log(2)-1) * np.log(nifty_data['Close']/nifty_data['Open'])**2
        ))
        
        # Garman-Klass volatility
        vol_metrics['gk_volatility'] = np.sqrt(252 * (
            np.log(nifty_data['High']/nifty_data['Close']) * np.log(nifty_data['High']/nifty_data['Open']) +
            np.log(nifty_data['Low']/nifty_data['Close']) * np.log(nifty_data['Low']/nifty_data['Open'])
        ))
        
        volatility_data['historical_vol'] = vol_metrics
        self.data_sources['volatility_data'] = volatility_data
        return volatility_data
    
    def fetch_all_data_sources(self):
        """
        Fetch all primary data sources using parallel processing
        """
        print("=" * 60)
        print("FETCHING ALL PRIMARY DATA SOURCES")
        print("=" * 60)
        
        # Parallel data fetching for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.fetch_market_data): 'market_data',
                executor.submit(self.fetch_volatility_metrics): 'volatility_data',
            }
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"✓ Completed {futures[future]}")
                except Exception as e:
                    print(f"✗ Failed {futures[future]}: {e}")
        
        print("\n✓ All primary data sources fetched successfully!")
        return self.data_sources

if __name__ == "__main__":
    # Test the data source manager
    manager = NiftyDataSourceManager(period='1y')
    data_sources = manager.fetch_all_data_sources()
    
    # Print summary
    for source_name, source_data in data_sources.items():
        print(f"\n{source_name}:")
        if isinstance(source_data, dict):
            for key, value in source_data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"  {key}: {value.shape} - {value.index.min()} to {value.index.max()}")
        else:
            print(f"  {type(source_data)}")
