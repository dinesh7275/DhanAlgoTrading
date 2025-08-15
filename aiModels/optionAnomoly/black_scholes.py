"""
Black-Scholes Option Pricing Calculator
======================================

Implementation of Black-Scholes model with Greeks calculation
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BlackScholesCalculator:
    """
    Black-Scholes option pricing calculator with Greeks
    """
    
    def __init__(self):
        pass
    
    def calculate_d1_d2(self, S, K, T, r, sigma):
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_option_price(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return max(price, 0)
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        if T <= 0:
            return {
                'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
            }
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        common_theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                       - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
        
        if option_type == 'call':
            theta = common_theta / 365  # Convert to daily
        else:
            theta = (common_theta + r * K * np.exp(-r * T) * 
                    (stats.norm.cdf(d2) - stats.norm.cdf(-d2))) / 365
        
        # Vega (same for call and put)
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_implied_volatility(self, market_price, S, K, T, r, option_type='call',
                                   max_iterations=100, tolerance=1e-5):
        """
        Calculate implied volatility using Newton-Raphson method
        """
        if T <= 0:
            return 0
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            # Calculate theoretical price and vega
            theoretical_price = self.calculate_option_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back from per 1% change
            
            # Newton-Raphson update
            price_diff = theoretical_price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if vega == 0:
                break
            
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)
            
            # Prevent unreasonable values
            if sigma > 5.0:
                sigma = 5.0
        
        return sigma
    
    def get_option_summary(self, S, K, T, r, sigma, option_type='call', market_price=None):
        """Get comprehensive option summary"""
        theoretical_price = self.calculate_option_price(S, K, T, r, sigma, option_type)
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
        
        summary = {
            'spot_price': S,
            'strike_price': K,
            'time_to_expiry': T,
            'risk_free_rate': r,
            'volatility': sigma,
            'option_type': option_type,
            'theoretical_price': theoretical_price,
            **greeks
        }
        
        if market_price is not None:
            summary['market_price'] = market_price
            summary['price_difference'] = theoretical_price - market_price
            summary['implied_volatility'] = self.calculate_implied_volatility(
                market_price, S, K, T, r, option_type
            )
        
        return summary