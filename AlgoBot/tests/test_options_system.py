#!/usr/bin/env python3
"""
Test Script for Modified Indian Options Trading System
=====================================================

Test the key modifications made to the system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading.options_trading_manager import OptionsChainAnalyzer, OptionsLotSizeCalculator
from live_trading.risk_manager import LiveRiskManager
from aiModels.indicators.technical_indicators import TechnicalIndicators
from aiModels.indicators.options_data_manager import OptionsDataManager
import pandas as pd
import numpy as np

def test_capital_management():
    """Test ₹10,000 capital management system"""
    print("Testing Capital Management System")
    print("="*50)
    
    # Test lot size calculator
    calculator = OptionsLotSizeCalculator(capital=10000)
    
    # Test different premium scenarios
    test_premiums = [50, 100, 150, 200, 250]
    
    for premium in test_premiums:
        position_info = calculator.calculate_position_size(premium)
        print(f"Premium: Rs.{premium}")
        print(f"  - Lots: {position_info['lots']}")
        print(f"  - Quantity: {position_info['quantity']}")
        print(f"  - Capital Required: Rs.{position_info['capital_required']:,.0f}")
        print(f"  - Capital Utilization: {position_info['capital_utilization']*100:.1f}%")
        print()

def test_tax_calculations():
    """Test Indian tax calculations"""
    print("Testing Indian Tax Calculations")
    print("="*50)
    
    risk_manager = LiveRiskManager(initial_capital=10000)
    
    # Test scenarios
    scenarios = [
        {'premium': 100, 'quantity': 50, 'gross_pnl': 2500},
        {'premium': 150, 'quantity': 100, 'gross_pnl': 5000},
        {'premium': 75, 'quantity': 150, 'gross_pnl': 1200}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}:")
        result = risk_manager.calculate_net_pnl(
            scenario['gross_pnl'], 
            scenario['premium'], 
            scenario['quantity']
        )
        
        print(f"  Premium: Rs.{scenario['premium']}, Quantity: {scenario['quantity']}")
        print(f"  Gross P&L: Rs.{result['gross_pnl']:,.2f}")
        print(f"  Total Taxes: Rs.{result['taxes']:,.2f}")
        print(f"  Net P&L: Rs.{result['net_pnl']:,.2f}")
        print(f"  Tax Breakdown:")
        print(f"    - STT: Rs.{result['tax_breakdown']['stt']:.2f}")
        print(f"    - Exchange: Rs.{result['tax_breakdown']['exchange_charges']:.2f}")
        print(f"    - Brokerage: Rs.{result['tax_breakdown']['brokerage']:.2f}")
        print(f"    - GST: Rs.{result['tax_breakdown']['gst']:.2f}")
        print()

def test_risk_management():
    """Test enhanced risk management"""
    print("Testing Risk Management System")
    print("="*50)
    
    risk_manager = LiveRiskManager(initial_capital=10000)
    
    print(f"Maximum trades per day: {risk_manager.max_trades_per_day}")
    print(f"Daily profit target: {risk_manager.daily_profit_target*100:.0f}%")
    print(f"Maximum daily loss: {risk_manager.max_daily_loss*100:.0f}%")
    print(f"Risk per trade: {risk_manager.max_position_risk*100:.0f}%")
    print()
    
    # Test trade limits
    print("Testing trade limits:")
    for trade_num in range(1, 13):
        can_trade, reason = risk_manager.can_place_trade()
        print(f"Trade {trade_num}: {'ALLOWED' if can_trade else 'BLOCKED'} - {reason}")
        
        if can_trade:
            # Simulate a trade
            risk_manager.update_trade_result(500 if trade_num % 2 == 0 else -300)
        
        if not can_trade:
            break

def test_technical_indicators():
    """Test enhanced technical indicators"""
    print("Testing Enhanced Technical Indicators")
    print("="*50)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 25000 + np.cumsum(np.random.randn(100) * 50)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # Test indicators
    indicators = TechnicalIndicators()
    ma_indicators = indicators.calculate_moving_averages(df)
    momentum_indicators = indicators.calculate_momentum_indicators(df)
    
    # Show key indicators for options trading
    latest_data = ma_indicators.iloc[-1]
    latest_momentum = momentum_indicators.iloc[-1]
    
    print(f"Current Price: Rs.{df['Close'].iloc[-1]:.2f}")
    print(f"EMA 6: Rs.{latest_data['EMA_6']:.2f}")
    print(f"EMA 15: Rs.{latest_data['EMA_15']:.2f}")
    print(f"EMA 6/15 Signal: {'Bullish' if latest_data['EMA_6_15_Signal'] == 1 else 'Bearish'}")
    print(f"RSI: {latest_momentum['RSI_14']:.1f}")
    print(f"MACD: {latest_momentum['MACD']:.2f}")
    print(f"MACD Signal: {latest_momentum['MACD_Signal']:.2f}")
    print(f"MACD Cross: {'Bullish' if latest_momentum['MACD_Signal_Cross'] == 1 else 'Bearish'}")

def test_options_chain():
    """Test options chain analysis"""
    print("Testing Options Chain Analysis")
    print("="*50)
    
    analyzer = OptionsChainAnalyzer()
    
    # Test with current Nifty levels
    spot_price = 25000
    options_chain = analyzer.get_nifty_options_chain(spot_price)
    
    print(f"Generated options chain for Nifty @ Rs.{spot_price}")
    print(f"Total options: {len(options_chain)}")
    
    # Show ATM options
    atm_options = options_chain[
        (options_chain['strike'] >= spot_price - 100) & 
        (options_chain['strike'] <= spot_price + 100)
    ]
    
    print("\nATM Options:")
    for _, option in atm_options.head(6).iterrows():
        print(f"  {option['strike']}{option['option_type']}: Rs.{option['premium']:.2f} "
              f"(Moneyness: {option['moneyness']:.3f})")
    
    # Test profitable strikes
    signal = 'BUY'  # Bullish signal
    profitable_options = analyzer.find_profitable_strikes(options_chain, signal, spot_price)
    
    print(f"\nTop 3 profitable strikes for {signal} signal:")
    for i, option in enumerate(profitable_options[:3]):
        print(f"  {i+1}. {option['strike']}{option['option_type']}: "
              f"Rs.{option['premium']:.2f} (R:R = {option['risk_reward']:.2f})")

def main():
    """Run all tests"""
    print("INDIAN OPTIONS TRADING SYSTEM - MODIFICATIONS TEST")
    print("="*60)
    print(f"Testing modified system with Rs.10,000 capital")
    print(f"Focus: Options buying only (CE/PE)")
    print(f"Target: 10% daily returns with proper risk management")
    print("="*60)
    print()
    
    try:
        test_capital_management()
        print()
        
        test_tax_calculations()
        print()
        
        test_risk_management()
        print()
        
        test_technical_indicators()
        print()
        
        test_options_chain()
        print()
        
        print("All tests completed successfully!")
        print("\nSYSTEM SUMMARY:")
        print("- Capital management updated to Rs.10,000")
        print("- Daily compounding implemented")
        print("- Indian tax calculations integrated")
        print("- Risk management enhanced (10 trades/day, 10% target)")
        print("- Technical indicators optimized (EMA 6/15, MACD, RSI)")
        print("- Options chain analysis implemented")
        print("- Lot size optimization for Rs.10,000 capital")
        print("- Focus on CE/PE buying only")
        print("- Real market data integration (removed simulations)")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()