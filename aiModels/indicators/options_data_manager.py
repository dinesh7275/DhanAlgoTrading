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
                    # Simplified Black-Scholes pricing
                    vol = 0.2 + np.random.normal(0, 0.05)  # Base vol with noise
                    T = dte / 365
                    
                    if option_type == 'CE':
                        intrinsic = max(0, spot_price - strike)
                        time_value = spot_price * vol * np.sqrt(T) * 0.4
                    else:
                        intrinsic = max(0, strike - spot_price)
                        time_value = spot_price * vol * np.sqrt(T) * 0.4
                    
                    option_price = intrinsic + time_value + np.random.uniform(-2, 2)
                    option_price = max(0.5, option_price)  # Minimum price
                    
                    # Simulate bid-ask spread
                    spread = option_price * 0.02  # 2% spread
                    bid = option_price - spread/2
                    ask = option_price + spread/2
                    
                    # Simulate open interest and volume
                    oi = max(100, int(np.random.exponential(5000)))
                    volume = max(0, int(np.random.poisson(oi * 0.1)))
                    
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
                        'underlying_price': spot_price
                    })
        
        return pd.DataFrame(options_chain)
    
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
