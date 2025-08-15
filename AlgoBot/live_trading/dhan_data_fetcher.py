"""
Live Data Fetcher using Dhan API
===============================

Fetch real-time market data from Dhan API for AI model predictions
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
from dhanhq import dhanhq
import warnings
warnings.filterwarnings('ignore')


class DhanLiveDataFetcher:
    """
    Fetch live market data using Dhan API
    """
    
    def __init__(self, client_id, access_token):
        self.client_id = client_id
        self.access_token = access_token
        self.dhan = dhanhq(client_id, access_token)
        self.instruments = {}
        self.live_data_cache = {}
        
        # Load instruments on initialization
        self._load_instruments()
    
    def _load_instruments(self):
        """Load instrument mappings"""
        try:
            # Get instruments from Dhan
            instruments = self.dhan.get_instruments()
            if instruments:
                self.instruments = {inst['tradingSymbol']: inst for inst in instruments}
                print(f"Loaded {len(self.instruments)} instruments")
            else:
                print("Failed to load instruments, using fallback mapping")
                # Fallback mapping for key instruments
                self.instruments = {
                    'NIFTY 50': {'securityId': '26000', 'instrumentType': 'INDEX'},
                    'BANK NIFTY': {'securityId': '26009', 'instrumentType': 'INDEX'},
                    'INDIA VIX': {'securityId': '26017', 'instrumentType': 'INDEX'}
                }
        except Exception as e:
            print(f"Error loading instruments: {e}")
            self.instruments = {}
    
    def get_live_nifty_data(self):
        """Get live Nifty 50 data"""
        try:
            # Get live quote
            nifty_quote = self.dhan.get_quote('IDX', '26000')  # Nifty 50 security ID
            
            if nifty_quote and 'data' in nifty_quote:
                data = nifty_quote['data']
                
                live_data = {
                    'timestamp': datetime.now(),
                    'symbol': 'NIFTY 50',
                    'ltp': data.get('LTP', 0),
                    'open': data.get('open', 0),
                    'high': data.get('high', 0),
                    'low': data.get('low', 0),
                    'close': data.get('prev_close', 0),
                    'volume': data.get('volume', 0),
                    'change': data.get('change', 0),
                    'change_percent': data.get('pChange', 0)
                }
                
                self.live_data_cache['NIFTY'] = live_data
                return live_data
            
        except Exception as e:
            print(f"Error fetching Nifty data: {e}")
            return self._get_fallback_nifty_data()
    
    def _get_fallback_nifty_data(self):
        """Fallback to Yahoo Finance if Dhan API fails"""
        try:
            nifty = yf.Ticker("^NSEI")
            info = nifty.info
            hist = nifty.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'timestamp': datetime.now(),
                    'symbol': 'NIFTY 50',
                    'ltp': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'close': hist['Close'].iloc[0],  # Day's opening as previous close
                    'volume': latest['Volume'],
                    'change': latest['Close'] - hist['Close'].iloc[0],
                    'change_percent': ((latest['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                }
        except Exception as e:
            print(f"Fallback data fetch failed: {e}")
            return None
    
    def get_historical_data(self, symbol, days=100, interval='1d'):
        """Get historical data for AI model training"""
        try:
            # Try Dhan API first
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For now, use Yahoo Finance as primary source for historical data
            # In production, you might want to use Dhan's historical data API
            ticker_map = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK',
                'INDIAVIX': '^INDIAVIX'
            }
            
            if symbol in ticker_map:
                ticker = yf.Ticker(ticker_map[symbol])
                hist_data = ticker.history(period=f"{days}d", interval=interval)
                
                if not hist_data.empty:
                    # Convert to standard format
                    hist_data = hist_data.reset_index()
                    hist_data.columns = [col.lower() if col != 'Date' else 'date' for col in hist_data.columns]
                    
                    return hist_data
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
        
        return None
    
    def get_india_vix(self):
        """Get India VIX data"""
        try:
            # Try Dhan API
            vix_quote = self.dhan.get_quote('IDX', '26017')  # India VIX security ID
            
            if vix_quote and 'data' in vix_quote:
                data = vix_quote['data']
                return {
                    'timestamp': datetime.now(),
                    'vix_value': data.get('LTP', 0),
                    'vix_change': data.get('change', 0),
                    'vix_change_percent': data.get('pChange', 0)
                }
        except Exception as e:
            print(f"Error fetching VIX: {e}")
        
        # Fallback to Yahoo Finance
        try:
            vix = yf.Ticker("^INDIAVIX")
            hist = vix.history(period="1d", interval="1m")
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'timestamp': datetime.now(),
                    'vix_value': latest['Close'],
                    'vix_change': latest['Close'] - hist['Close'].iloc[0],
                    'vix_change_percent': ((latest['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                }
        except Exception as e:
            print(f"VIX fallback failed: {e}")
        
        return None
    
    def get_bank_nifty(self):
        """Get Bank Nifty data"""
        try:
            bank_quote = self.dhan.get_quote('IDX', '26009')  # Bank Nifty security ID
            
            if bank_quote and 'data' in bank_quote:
                data = bank_quote['data']
                return {
                    'timestamp': datetime.now(),
                    'bank_nifty': data.get('LTP', 0),
                    'bank_change': data.get('change', 0),
                    'bank_change_percent': data.get('pChange', 0)
                }
        except Exception as e:
            print(f"Error fetching Bank Nifty: {e}")
        
        # Fallback
        try:
            bank = yf.Ticker("^NSEBANK")
            hist = bank.history(period="1d", interval="1m")
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'timestamp': datetime.now(),
                    'bank_nifty': latest['Close'],
                    'bank_change': latest['Close'] - hist['Close'].iloc[0],
                    'bank_change_percent': ((latest['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                }
        except Exception as e:
            print(f"Bank Nifty fallback failed: {e}")
        
        return None
    
    def get_options_chain(self, underlying='NIFTY', expiry_date=None):
        """Get options chain data"""
        try:
            # Get options chain from Dhan
            # Note: You'll need to implement based on Dhan's options API
            # This is a placeholder structure
            
            if expiry_date is None:
                # Get nearest Thursday expiry
                today = datetime.now()
                days_until_thursday = (3 - today.weekday()) % 7
                if days_until_thursday == 0:
                    days_until_thursday = 7
                expiry_date = today + timedelta(days=days_until_thursday)
            
            # Placeholder - implement actual Dhan options API call
            options_data = {
                'underlying': underlying,
                'expiry': expiry_date,
                'calls': [],  # List of call options
                'puts': [],   # List of put options
                'timestamp': datetime.now()
            }
            
            return options_data
            
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return None
    
    def get_comprehensive_market_data(self):
        """Get comprehensive market data for AI models"""
        market_data = {
            'timestamp': datetime.now(),
            'nifty': self.get_live_nifty_data(),
            'bank_nifty': self.get_bank_nifty(),
            'india_vix': self.get_india_vix(),
            'options_chain': self.get_options_chain()
        }
        
        return market_data
    
    def get_real_time_feed(self, symbols=['NIFTY', 'BANKNIFTY'], callback=None):
        """
        Get real-time data feed
        Note: Implement WebSocket connection for real-time data
        """
        try:
            while True:
                market_data = self.get_comprehensive_market_data()
                
                if callback:
                    callback(market_data)
                else:
                    print(f"Market Data at {market_data['timestamp']}")
                    if market_data['nifty']:
                        print(f"Nifty: {market_data['nifty']['ltp']} ({market_data['nifty']['change_percent']:+.2f}%)")
                    if market_data['india_vix']:
                        print(f"VIX: {market_data['india_vix']['vix_value']:.2f}")
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("Real-time feed stopped")
        except Exception as e:
            print(f"Error in real-time feed: {e}")
    
    def validate_connection(self):
        """Validate Dhan API connection"""
        try:
            # Test API connection
            funds = self.dhan.get_fund_limits()
            if funds:
                print(" Dhan API connection successful")
                return True
            else:
                print(" Dhan API connection failed")
                return False
        except Exception as e:
            print(f" Dhan API validation error: {e}")
            return False
    
    def get_account_info(self):
        """Get account information"""
        try:
            funds = self.dhan.get_fund_limits()
            holdings = self.dhan.get_holdings()
            positions = self.dhan.get_positions()
            
            return {
                'funds': funds,
                'holdings': holdings,
                'positions': positions,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return None


class MarketDataProcessor:
    """
    Process live market data for AI model input
    """
    
    def __init__(self):
        self.data_buffer = []
        self.processed_features = {}
    
    def add_data_point(self, market_data):
        """Add new market data point to buffer"""
        self.data_buffer.append(market_data)
        
        # Keep only last 200 data points for feature calculation
        if len(self.data_buffer) > 200:
            self.data_buffer = self.data_buffer[-200:]
    
    def calculate_features(self):
        """Calculate features from buffered data for AI models"""
        if len(self.data_buffer) < 20:
            return None
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([
            {
                'timestamp': data['timestamp'],
                'close': data['nifty']['ltp'] if data['nifty'] else np.nan,
                'volume': data['nifty']['volume'] if data['nifty'] else np.nan,
                'india_vix': data['india_vix']['vix_value'] if data['india_vix'] else np.nan,
                'bank_nifty': data['bank_nifty']['bank_nifty'] if data['bank_nifty'] else np.nan
            }
            for data in self.data_buffer
        ])
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Return latest features
        latest_features = df.iloc[-1].to_dict()
        self.processed_features = latest_features
        
        return latest_features
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_model_input(self, model_type='volatility'):
        """Get processed data formatted for specific AI model"""
        if not self.processed_features:
            return None
        
        if model_type == 'volatility':
            # Features for volatility prediction
            return {
                'close': self.processed_features.get('close', 0),
                'volume': self.processed_features.get('volume', 0),
                'returns': self.processed_features.get('returns', 0),
                'sma_10': self.processed_features.get('sma_10', 0),
                'sma_20': self.processed_features.get('sma_20', 0),
                'rsi': self.processed_features.get('rsi', 50),
                'india_vix': self.processed_features.get('india_vix', 20),
                'volatility': self.processed_features.get('volatility', 0.2)
            }
        
        elif model_type == 'price_movement':
            # Features for price movement prediction
            return {
                'close': self.processed_features.get('close', 0),
                'returns': self.processed_features.get('returns', 0),
                'sma_10': self.processed_features.get('sma_10', 0),
                'sma_20': self.processed_features.get('sma_20', 0),
                'rsi': self.processed_features.get('rsi', 50),
                'volatility': self.processed_features.get('volatility', 0.2),
                'bank_nifty': self.processed_features.get('bank_nifty', 0)
            }
    
    def get_options_chain_data(self, underlying_symbol="NIFTY", expiry_date=None):
        """
        Fetch Nifty options chain data from Dhan API
        """
        try:
            # Dhan API endpoints for options data
            # Note: Update with actual Dhan API endpoints when available
            
            if expiry_date is None:
                # Get current weekly expiry
                expiry_date = self._get_current_weekly_expiry()
            
            # Placeholder for Dhan options chain API
            # In practice, use Dhan's options chain API:
            # options_data = self.dhan.get_option_chain(underlying_symbol, expiry_date)
            
            # For now, use NSE data or fallback to simulation
            options_chain = self._fetch_nse_options_chain(underlying_symbol, expiry_date)
            
            return {
                'success': True,
                'underlying': underlying_symbol,
                'expiry_date': expiry_date,
                'options_chain': options_chain,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _get_current_weekly_expiry(self):
        """Get current weekly expiry (Thursday)"""
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:  # After market close on Thursday
            days_until_thursday = 7
        
        expiry_date = today + timedelta(days=days_until_thursday)
        return expiry_date.strftime('%Y-%m-%d')
    
    def _fetch_nse_options_chain(self, underlying, expiry_date):
        """
        Fetch options chain from NSE or other sources
        This is a placeholder - integrate with actual NSE API or Dhan options API
        """
        try:
            # Get current Nifty price for strike generation
            current_price = self._get_current_nifty_price()
            
            # Generate strikes around current price
            base_strike = round(current_price / 50) * 50
            strikes = []
            for offset in range(-500, 550, 50):
                strikes.append(base_strike + offset)
            
            options_data = []
            
            for strike in strikes:
                # Call option data
                call_data = {
                    'strike': strike,
                    'option_type': 'CE',
                    'expiry': expiry_date,
                    'ltp': self._calculate_option_ltp(current_price, strike, 'CE'),
                    'bid': 0,
                    'ask': 0,
                    'volume': 0,
                    'oi': 0,
                    'change': 0,
                    'change_percent': 0
                }
                
                # Put option data
                put_data = {
                    'strike': strike,
                    'option_type': 'PE',
                    'expiry': expiry_date,
                    'ltp': self._calculate_option_ltp(current_price, strike, 'PE'),
                    'bid': 0,
                    'ask': 0,
                    'volume': 0,
                    'oi': 0,
                    'change': 0,
                    'change_percent': 0
                }
                
                options_data.extend([call_data, put_data])
            
            return options_data
            
        except Exception as e:
            print(f"Error fetching NSE options chain: {e}")
            return []
    
    def _calculate_option_ltp(self, spot_price, strike, option_type):
        """Calculate theoretical option LTP (simplified)"""
        if option_type == 'CE':
            intrinsic = max(0, spot_price - strike)
        else:
            intrinsic = max(0, strike - spot_price)
        
        # Add time value (simplified)
        time_value = abs(spot_price - strike) * 0.02 + 5
        
        return max(0.5, intrinsic + time_value)
    
    def _get_current_nifty_price(self):
        """Get current Nifty price"""
        try:
            market_data = self.get_comprehensive_market_data()
            return market_data.get('nifty', {}).get('ltp', 25000)  # Default fallback
        except:
            return 25000  # Default fallback
        
        return self.processed_features