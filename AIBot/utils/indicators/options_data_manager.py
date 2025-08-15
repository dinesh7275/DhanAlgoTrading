# options_data_manager.py
# Options Chain Data and Greeks Calculator

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class OptionsDataManager:
    """
    Manages options chain data and Greeks calculations
    """
    
    def __init__(self, risk_free_rate=0.065):
        self.risk_free_rate = risk_free_rate
        
    def generate_options_chain(self, spot_price, date=None):
        """
        Generate synthetic options chain for demonstration
        In production, replace with real NSE options data
        """
        if date is None:
            date = pd.Timestamp.now().date()
            
        strikes = np.arange(spot_price * 0.9, spot_price * 1.1, 50)
        expiries = [7, 14, 30, 45]  # Days to expiry
        
        options_chain = []
        
        for dte in expiries:
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Real Black-Scholes pricing with proper Greeks
                    T = dte / 365
                    vol = self.get_historical_volatility(spot_price)  # Real volatility
                    
                    greeks = self.calculate_option_greeks(spot_price, strike, T, self.risk_free_rate, vol, option_type)
                    
                    # Calculate fair value using Black-Scholes
                    option_price = self.black_scholes_price(spot_price, strike, T, self.risk_free_rate, vol, option_type)
                    option_price = max(0.5, option_price)  # Minimum price
                    
                    # Realistic bid-ask spread based on liquidity
                    spread_pct = self.calculate_spread(strike, spot_price, dte)
                    spread = option_price * spread_pct
                    bid = max(0.25, option_price - spread/2)
                    ask = option_price + spread/2
                    
                    # Get real market data (OI, Volume) - placeholder for API integration
                    oi = self.get_market_oi(strike, dte, option_type)
                    volume = self.get_market_volume(strike, dte, option_type)
                    
                    options_chain.append({
                        'date': date,
                        'strike': strike,
                        'expiry_days': dte,
                        'option_type': option_type,
                        'ltp': option_price,
                        'bid': bid,
                        'ask': ask,
                        'volume': volume,
                        'open_interest': oi,
                        'implied_vol': vol,
                        'underlying_price': spot_price,
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'theta': greeks['theta'],
                        'vega': greeks['vega'],
                        'rho': greeks['rho'],
                        'moneyness': spot_price / strike,
                        'intrinsic_value': max(0, (spot_price - strike) if option_type == 'CE' else (strike - spot_price))
                    })
        
        return pd.DataFrame(options_chain)
    
    def get_historical_volatility(self, spot_price, days=30):
        """Calculate historical volatility from real market data"""
        try:
            # Use yfinance to get Nifty data for volatility calculation
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            hist = ticker.history(period=f"{days}d")
            
            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                return max(0.1, min(0.8, volatility))  # Bounded between 10% and 80%
        except:
            pass
        
        return 0.20  # Default 20% volatility
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes option price"""
        if T <= 0:
            if option_type == 'CE':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'CE':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return max(0.5, price)
    
    def calculate_spread(self, strike, spot_price, dte):
        """Calculate realistic bid-ask spread"""
        # ATM options have tighter spreads
        moneyness = abs(spot_price - strike) / spot_price
        
        if moneyness < 0.02:  # ATM
            base_spread = 0.005  # 0.5%
        elif moneyness < 0.05:  # Near ATM
            base_spread = 0.01   # 1%
        else:  # OTM/Deep ITM
            base_spread = 0.02   # 2%
        
        # Increase spread for shorter DTE (less liquidity)
        if dte <= 7:
            base_spread *= 1.5
        elif dte <= 1:
            base_spread *= 2.0
        
        return base_spread
    
    def get_market_oi(self, strike, dte, option_type):
        """Get market open interest - placeholder for real API integration"""
        # ATM strikes have higher OI
        base_oi = 50000
        
        # Reduce for weekly expiry vs monthly
        if dte <= 7:
            base_oi *= 0.3
        
        # Add some randomness for now
        return int(base_oi * (0.5 + np.random.random()))
    
    def get_market_volume(self, strike, dte, option_type):
        """Get market volume - placeholder for real API integration"""
        base_volume = 10000
        
        # Weekly options have lower volume
        if dte <= 7:
            base_volume *= 0.4
        
        # Add some randomness for now  
        return int(base_volume * (0.2 + np.random.random()))
    
    def calculate_option_greeks(self, S, K, T, r, sigma, option_type):
        """
        Calculate option Greeks using Black-Scholes model
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'CE':
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r*T) * norm.cdf(d2)
        else:
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + 
                 r * K * np.exp(-r*T) * norm.cdf(d2 if option_type == 'CE' else -d2))
        theta = theta / 365  # Per day
        
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho / 100  # Per 1% rate change
        }
    
    def add_greeks_to_options(self, options_df):
        """
        Add Greeks to options DataFrame
        """
        print("Calculating Greeks for options chain...")
        
        greeks_data = []
        
        for _, option in options_df.iterrows():
            greeks = self.calculate_option_greeks(
                option['underlying_price'],
                option['strike'],
                option['expiry_days'] / 365,
                self.risk_free_rate,
                option['implied_vol'],
                option['option_type']
            )
            
            option_with_greeks = option.to_dict()
            option_with_greeks.update(greeks)
            greeks_data.append(option_with_greeks)
        
        return pd.DataFrame(greeks_data)
    
    def generate_options_for_dates(self, nifty_data, sample_every=5):
        """
        Generate options chain for multiple dates
        """
        print(f"Generating options data for multiple dates (sampling every {sample_every} days)...")
        
        sample_dates = nifty_data.index[::sample_every]
        all_options = []
        
        for i, date in enumerate(sample_dates):
            if i % 20 == 0:
                print(f"Processing {i+1}/{len(sample_dates)} dates...")
            
            nifty_price = nifty_data.loc[date, 'Close']
            daily_options = self.generate_options_chain(nifty_price, date)
            all_options.append(daily_options)
        
        # Combine all options data
        combined_options = pd.concat(all_options, ignore_index=True)
        
        # Add Greeks
        options_with_greeks = self.add_greeks_to_options(combined_options)
        
        print(f"Generated {len(options_with_greeks)} option contracts")
        return options_with_greeks

if __name__ == "__main__":
    # Test the options data manager
    options_manager = OptionsDataManager()
    
    # Generate sample options chain
    sample_options = options_manager.generate_options_chain(spot_price=19500)
    print(f"Generated {len(sample_options)} options contracts")
    print(sample_options.head())
    
    # Add Greeks
    options_with_greeks = options_manager.add_greeks_to_options(sample_options)
    print(f"Added Greeks to {len(options_with_greeks)} contracts")
    print(options_with_greeks[['strike', 'option_type', 'ltp', 'delta', 'gamma', 'theta']].head())
