#!/usr/bin/env python3
"""
Indian Options Trading Bot
=========================

Modified version focusing on options buying with ₹10,000 capital
"""

import sys
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading.options_trading_manager import IndianOptionsTrader
from live_trading.config_manager import ConfigManager

def main():
    """Main function for Indian Options Trading Bot"""
    parser = argparse.ArgumentParser(description='Indian Options Trading Bot')
    parser.add_argument('--capital', type=int, default=10000, 
                       help='Initial capital in INR (default: 10000)')
    parser.add_argument('--paper', action='store_true', 
                       help='Run in paper trading mode')
    parser.add_argument('--live', action='store_true', 
                       help='Run in LIVE trading mode (REAL MONEY!)')
    parser.add_argument('--setup', action='store_true', 
                       help='Run setup to configure credentials')
    
    args = parser.parse_args()
    
    print(f"🇮🇳 Indian Options Trading Bot")
    print(f"Capital: ₹{args.capital:,}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup configuration if requested
    if args.setup:
        print("Running setup...")
        config_manager, credentials = setup_trading_environment()
        if config_manager and credentials:
            print("✅ Setup completed successfully!")
            print("Run the bot again without --setup to start trading")
        else:
            print("❌ Setup failed")
        return
    
    # Load existing configuration
    try:
        config_manager, credentials = setup_trading_environment()
        if not config_manager or not credentials:
            print("❌ Configuration missing. Run with --setup first")
            return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Update configuration for Indian options trading
    trading_config = config_manager.get_trading_config()
    trading_config['initial_capital'] = args.capital
    trading_config['max_daily_loss'] = 0.10  # 10% max daily loss
    trading_config['max_positions'] = 3      # Max 3 positions
    trading_config['max_risk_per_trade'] = 0.10  # 10% risk per trade
    
    # Determine trading mode
    paper_trading = True  # Default to paper trading
    
    if args.live:
        confirm = input("⚠️ LIVE TRADING MODE - Real money will be used!\n" +
                       "Type 'I UNDERSTAND THE RISKS' to proceed: ")
        if confirm == 'I UNDERSTAND THE RISKS':
            paper_trading = False
            print("🔴 LIVE TRADING MODE ACTIVATED")
        else:
            print("Live trading not confirmed - using paper trading")
    elif args.paper:
        print("📄 Paper Trading Mode")
    
    # Create and start options trader
    try:
        print("\n🚀 Initializing Options Trader...")
        
        trader = IndianOptionsTrader(
            client_id=credentials['client_id'],
            access_token=credentials['access_token'],
            initial_capital=args.capital
        )
        
        trader.paper_trading = paper_trading
        
        print("✅ Trader initialized successfully")
        print("\n🎯 Trading Parameters:")
        print(f"   • Daily Profit Target: 10% (₹{args.capital * 0.1:,.0f})")
        print(f"   • Maximum Trades/Day: 10")
        print(f"   • Maximum Positions: 3")
        print(f"   • Risk per Trade: 10%")
        print(f"   • Focus: CE/PE Options Buying Only")
        
        print(f"\n📊 Technical Indicators Used:")
        print(f"   • EMA 6 & EMA 15 crossovers")
        print(f"   • MACD signals")
        print(f"   • RSI momentum")
        print(f"   • Volume analysis")
        print(f"   • Options Greeks (Delta, Gamma, Theta, Vega)")
        
        print(f"\n💰 Tax Optimization:")
        print(f"   • STT: 0.017% on options premium")
        print(f"   • Exchange charges: 0.0019%")
        print(f"   • GST on brokerage: 18%")
        print(f"   • Net P&L calculation after all taxes")
        
        print(f"\n🎲 Lot Size Optimization:")
        print(f"   • Nifty lot size: 50")
        print(f"   • Max affordable lots: {args.capital // (200 * 50)}")  # Assuming ₹200 avg premium
        print(f"   • Position sizing based on premium and risk")
        
        if paper_trading:
            print(f"\n⚠️ PAPER TRADING MODE - No real money involved")
        else:
            print(f"\n🔴 LIVE TRADING MODE - Real money will be used!")
            
        print(f"\n▶️ Starting options trading...")
        print(f"Press Ctrl+C to stop\n")
        
        # Start trading
        trader.start_options_trading(update_interval=30)
        
    except KeyboardInterrupt:
        print("\n⏹️ Trading stopped by user")
    except Exception as e:
        print(f"❌ Trading error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()