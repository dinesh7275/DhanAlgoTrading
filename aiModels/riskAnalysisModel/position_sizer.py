"""
Position Sizing and Risk Management
==================================

Calculate optimal position sizes based on risk management rules
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PositionSizer:
    """
    Calculate position sizes based on various risk management methods
    """
    
    def __init__(self, account_balance, max_risk_per_trade=0.02, max_portfolio_risk=0.20):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.current_positions = {}
        
    def fixed_fractional_sizing(self, stop_loss_distance, entry_price):
        """
        Fixed fractional position sizing
        Risk amount = Account * Risk%
        Position size = Risk amount / (Entry price - Stop loss)
        """
        risk_amount = self.account_balance * self.max_risk_per_trade
        
        if stop_loss_distance <= 0:
            return 0
        
        position_size = risk_amount / (stop_loss_distance * entry_price)
        return max(0, int(position_size))
    
    def kelly_criterion_sizing(self, win_rate, avg_win, avg_loss):
        """
        Kelly Criterion for optimal position sizing
        f = (bp - q) / b
        where:
        f = fraction of capital to wager
        b = odds of winning (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly fraction to avoid excessive risk
        kelly_fraction = max(0, min(kelly_fraction, self.max_risk_per_trade * 2))
        
        return kelly_fraction
    
    def volatility_based_sizing(self, asset_volatility, target_volatility=0.15):
        """
        Position sizing based on asset volatility
        Position weight = Target volatility / Asset volatility
        """
        if asset_volatility <= 0:
            return 0
        
        position_weight = target_volatility / asset_volatility
        
        # Cap position weight
        position_weight = min(position_weight, self.max_portfolio_risk)
        
        return position_weight
    
    def atr_based_sizing(self, atr, entry_price, atr_multiplier=2):
        """
        ATR-based position sizing
        Stop loss = ATR * multiplier
        Position size based on this stop loss
        """
        stop_loss_distance = atr * atr_multiplier
        return self.fixed_fractional_sizing(stop_loss_distance, entry_price)
    
    def calculate_nifty_position_size(self, entry_price, stop_loss=None, 
                                    atr=None, volatility=None, method='fixed_fractional'):
        """
        Calculate position size for Nifty options/futures
        """
        if method == 'fixed_fractional':
            if stop_loss is None:
                return 0
            stop_loss_distance = abs(entry_price - stop_loss)
            return self.fixed_fractional_sizing(stop_loss_distance, entry_price)
        
        elif method == 'atr':
            if atr is None:
                return 0
            return self.atr_based_sizing(atr, entry_price)
        
        elif method == 'volatility':
            if volatility is None:
                return 0
            position_weight = self.volatility_based_sizing(volatility)
            position_value = self.account_balance * position_weight
            return int(position_value / entry_price)
        
        else:
            return 0
    
    def calculate_options_position_size(self, option_price, delta, underlying_price, 
                                      max_loss_per_contract=None):
        """
        Calculate position size for options
        """
        if max_loss_per_contract is None:
            max_loss_per_contract = option_price  # Maximum loss is premium paid
        
        # Risk per contract
        risk_per_contract = max_loss_per_contract
        
        # Maximum risk amount
        max_risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Position size
        position_size = int(max_risk_amount / risk_per_contract)
        
        # Adjust for delta (effective exposure)
        if delta is not None and delta > 0:
            effective_position_value = position_size * option_price * delta
            max_position_value = self.account_balance * self.max_portfolio_risk
            
            if effective_position_value > max_position_value:
                position_size = int(max_position_value / (option_price * delta))
        
        return max(0, position_size)
    
    def add_position(self, symbol, quantity, entry_price, position_type='long'):
        """
        Add a position to the portfolio
        """
        position_value = quantity * entry_price
        
        self.current_positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'position_value': position_value,
            'position_type': position_type,
            'weight': position_value / self.account_balance
        }
    
    def remove_position(self, symbol):
        """
        Remove a position from the portfolio
        """
        if symbol in self.current_positions:
            del self.current_positions[symbol]
    
    def get_portfolio_exposure(self):
        """
        Calculate current portfolio exposure
        """
        total_long_exposure = 0
        total_short_exposure = 0
        
        for symbol, position in self.current_positions.items():
            if position['position_type'] == 'long':
                total_long_exposure += position['position_value']
            else:
                total_short_exposure += position['position_value']
        
        total_exposure = total_long_exposure + total_short_exposure
        
        return {
            'total_exposure': total_exposure,
            'long_exposure': total_long_exposure,
            'short_exposure': total_short_exposure,
            'net_exposure': total_long_exposure - total_short_exposure,
            'exposure_ratio': total_exposure / self.account_balance,
            'long_ratio': total_long_exposure / self.account_balance,
            'short_ratio': total_short_exposure / self.account_balance
        }
    
    def check_risk_limits(self, new_position_value, position_type='long'):
        """
        Check if new position violates risk limits
        """
        current_exposure = self.get_portfolio_exposure()
        
        # Check portfolio risk limit
        if position_type == 'long':
            new_total_exposure = current_exposure['total_exposure'] + new_position_value
        else:
            new_total_exposure = current_exposure['total_exposure'] + new_position_value
        
        new_exposure_ratio = new_total_exposure / self.account_balance
        
        # Risk checks
        checks = {
            'portfolio_risk_ok': new_exposure_ratio <= self.max_portfolio_risk,
            'position_size_ok': new_position_value / self.account_balance <= self.max_risk_per_trade * 5,
            'current_exposure_ratio': current_exposure['exposure_ratio'],
            'new_exposure_ratio': new_exposure_ratio,
            'max_allowed_exposure': self.max_portfolio_risk
        }
        
        checks['all_checks_passed'] = all([
            checks['portfolio_risk_ok'],
            checks['position_size_ok']
        ])
        
        return checks
    
    def get_max_position_size(self, entry_price, stop_loss_price=None):
        """
        Get maximum allowed position size
        """
        if stop_loss_price is not None:
            # Based on stop loss
            risk_per_share = abs(entry_price - stop_loss_price)
            max_risk_amount = self.account_balance * self.max_risk_per_trade
            max_position_size = int(max_risk_amount / risk_per_share)
        else:
            # Based on portfolio exposure limit
            max_position_value = self.account_balance * self.max_portfolio_risk
            max_position_size = int(max_position_value / entry_price)
        
        return max_position_size
    
    def calculate_stop_loss_levels(self, entry_price, method='percentage', 
                                 percentage=0.02, atr=None, atr_multiplier=2):
        """
        Calculate stop loss levels
        """
        if method == 'percentage':
            stop_loss = entry_price * (1 - percentage)
        
        elif method == 'atr' and atr is not None:
            stop_loss = entry_price - (atr * atr_multiplier)
        
        elif method == 'fixed_amount':
            # Fixed amount based on risk per trade
            max_risk_amount = self.account_balance * self.max_risk_per_trade
            # Assume 100 shares for calculation
            risk_per_share = max_risk_amount / 100
            stop_loss = entry_price - risk_per_share
        
        else:
            stop_loss = entry_price * 0.98  # Default 2% stop
        
        return max(0, stop_loss)
    
    def get_position_sizing_summary(self):
        """
        Get summary of position sizing rules and current state
        """
        exposure = self.get_portfolio_exposure()
        
        summary = {
            'account_balance': self.account_balance,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'current_positions': len(self.current_positions),
            'current_exposure': exposure,
            'available_risk_capital': self.account_balance * self.max_portfolio_risk - exposure['total_exposure'],
            'max_single_position_value': self.account_balance * self.max_risk_per_trade * 5
        }
        
        return summary