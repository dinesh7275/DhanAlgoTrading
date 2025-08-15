"""
Trading Utilities
================

Common trading calculations and utilities for AlgoBot
"""

def calculate_indian_taxes(trade_value: float, option_type: str = None) -> dict:
    """
    Calculate Indian trading taxes and charges
    
    Args:
        trade_value: Trade value in INR
        option_type: 'CE', 'PE' for options, None for equity
    
    Returns:
        dict: Breakdown of all charges
    """
    charges = {}
    
    # STT (Securities Transaction Tax)
    if option_type:  # Options
        charges['stt'] = trade_value * 0.00017  # 0.017%
    else:  # Equity
        charges['stt'] = trade_value * 0.00025  # 0.025%
    
    # Exchange charges
    charges['exchange_charges'] = trade_value * 0.000019  # 0.0019%
    
    # Brokerage (flat rate)
    charges['brokerage'] = 20.0  # ₹20 per trade
    
    # GST on brokerage
    charges['gst'] = charges['brokerage'] * 0.18  # 18% GST
    
    # Total charges
    charges['total'] = sum(charges.values())
    
    return charges

def calculate_position_size(capital: float, risk_per_trade: float, stop_loss_pct: float) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        capital: Available capital
        risk_per_trade: Risk per trade as percentage (0.02 = 2%)
        stop_loss_pct: Stop loss percentage (0.05 = 5%)
    
    Returns:
        float: Position size in INR
    """
    max_risk_amount = capital * risk_per_trade
    position_size = max_risk_amount / stop_loss_pct
    
    # Ensure position size doesn't exceed available capital
    return min(position_size, capital * 0.25)  # Max 25% of capital per trade

def calculate_options_intrinsic_value(spot_price: float, strike_price: float, option_type: str) -> float:
    """
    Calculate intrinsic value of option
    
    Args:
        spot_price: Current spot price
        strike_price: Strike price of option
        option_type: 'CE' or 'PE'
    
    Returns:
        float: Intrinsic value
    """
    if option_type.upper() == 'CE':
        return max(0, spot_price - strike_price)
    elif option_type.upper() == 'PE':
        return max(0, strike_price - spot_price)
    else:
        return 0

def is_market_open() -> bool:
    """
    Check if Indian market is currently open
    
    Returns:
        bool: True if market is open
    """
    from datetime import datetime
    import pytz
    
    # Indian timezone
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Weekend check
    if now.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def format_indian_currency(amount: float) -> str:
    """
    Format amount in Indian currency style
    
    Args:
        amount: Amount to format
    
    Returns:
        str: Formatted currency string
    """
    return f"₹{amount:,.2f}"

def calculate_pnl_percentage(entry_price: float, current_price: float, quantity: int) -> float:
    """
    Calculate P&L percentage
    
    Args:
        entry_price: Entry price per unit
        current_price: Current price per unit
        quantity: Number of units
    
    Returns:
        float: P&L percentage
    """
    if entry_price == 0:
        return 0
    
    return ((current_price - entry_price) / entry_price) * 100

def validate_trade_limits(trade_data: dict, daily_limits: dict) -> dict:
    """
    Validate trade against daily limits
    
    Args:
        trade_data: Trade information
        daily_limits: Daily trading limits
    
    Returns:
        dict: Validation result with success/failure and message
    """
    # Check trade count limit
    if daily_limits.get('trades_today', 0) >= daily_limits.get('max_trades_per_day', 10):
        return {
            'valid': False,
            'message': 'Daily trade limit exceeded'
        }
    
    # Check loss limit
    if daily_limits.get('daily_loss', 0) >= daily_limits.get('max_daily_loss', 0.10):
        return {
            'valid': False,
            'message': 'Daily loss limit reached'
        }
    
    # Check capital requirement
    trade_value = trade_data.get('quantity', 0) * trade_data.get('price', 0)
    if trade_value > daily_limits.get('available_capital', 0):
        return {
            'valid': False,
            'message': 'Insufficient capital'
        }
    
    return {
        'valid': True,
        'message': 'Trade validation passed'
    }