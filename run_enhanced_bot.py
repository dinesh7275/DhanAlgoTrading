#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Trading Bot Launcher
===============================

Simple launcher script for the enhanced AI trading bot with all features:
- 30-day learning models
- Multi-timeframe analysis
- Live candlestick charts
- Real-time indicator monitoring
- Paper trading simulation
- Adaptive learning system
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 Enhanced AI Trading Bot for Dhan")
    print("====================================")
    print()
    print("Select mode:")
    print("1. Paper Trading (Recommended for testing)")
    print("2. Live Trading (Requires Dhan API credentials)")
    print("3. Dashboard Only")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    # Get the AIBot directory
    bot_dir = Path(__file__).parent / "AIBot"
    
    if choice == "1":
        print("\n🚀 Starting Paper Trading Mode...")
        print("Features enabled:")
        print("✅ 30-day AI model learning")
        print("✅ Multi-timeframe analysis (1m, 5m, 15m, 1h, 1d)")
        print("✅ Live candlestick charts with ML patterns")
        print("✅ Real-time indicator monitoring")
        print("✅ Comprehensive signal generation")
        print("✅ Paper trading with ₹10,000 virtual capital")
        print("✅ Adaptive learning from trading results")
        print("✅ Enhanced dashboard at http://localhost:5002")
        print()
        print("Starting in 3 seconds...")
        
        try:
            # Change to AIBot directory and run
            os.chdir(bot_dir)
            subprocess.run([
                sys.executable, "main_enhanced_trading_bot.py", 
                "--paper", "--debug"
            ])
        except KeyboardInterrupt:
            print("\n🛑 Paper trading stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif choice == "2":
        print("\n⚠️  Live Trading Mode")
        print("This mode requires valid Dhan API credentials.")
        print("Make sure you have configured your API keys in config.py")
        print()
        confirm = input("Are you sure you want to proceed with live trading? (y/N): ").strip().lower()
        
        if confirm == 'y':
            print("\n🚀 Starting Live Trading Mode...")
            try:
                os.chdir(bot_dir)
                subprocess.run([
                    sys.executable, "main_enhanced_trading_bot.py", 
                    "--live", "--debug"
                ])
            except KeyboardInterrupt:
                print("\n🛑 Live trading stopped by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print("Live trading cancelled.")
    
    elif choice == "3":
        print("\n📊 Starting Dashboard Only...")
        try:
            os.chdir(bot_dir)
            subprocess.run([sys.executable, "enhanced_dashboard.py"])
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif choice == "4":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()