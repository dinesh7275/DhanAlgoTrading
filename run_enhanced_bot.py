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
    print("Enhanced AI Trading Bot for Dhan")
    print("================================")
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
        print("\nStarting Simplified Paper Trading Mode...")
        print("Features enabled:")
        print("- Basic technical analysis (RSI, Moving Averages)")
        print("- Simple signal generation")
        print("- Paper trading with Rs.10,000 virtual capital")
        print("- Real-time NIFTY monitoring")
        print("- Performance tracking")
        print()
        print("Starting bot...")
        
        try:
            # Change to AIBot directory and run simplified version
            os.chdir(bot_dir)
            subprocess.run([
                sys.executable, "main_simplified_trading_bot.py", 
                "--debug", "--interval", "60"
            ])
        except KeyboardInterrupt:
            print("\nPaper trading stopped by user")
        except Exception as e:
            print(f"\nError: {e}")
        
        # Option for advanced features
        print("\nTo enable advanced features (ML models, multi-timeframe, etc.):")
        print("Run: python main_enhanced_trading_bot.py --paper --debug")
    
    elif choice == "2":
        print("\nLive Trading Mode")
        print("This mode requires valid Dhan API credentials.")
        print("Make sure you have configured your API keys in config.py")
        print()
        confirm = input("Are you sure you want to proceed with live trading? (y/N): ").strip().lower()
        
        if confirm == 'y':
            print("\nStarting Live Trading Mode...")
            try:
                os.chdir(bot_dir)
                subprocess.run([
                    sys.executable, "main_enhanced_trading_bot.py", 
                    "--live", "--debug"
                ])
            except KeyboardInterrupt:
                print("\nLive trading stopped by user")
            except Exception as e:
                print(f"\nError: {e}")
        else:
            print("Live trading cancelled.")
    
    elif choice == "3":
        print("\nStarting Dashboard Only...")
        try:
            os.chdir(bot_dir)
            subprocess.run([sys.executable, "enhanced_dashboard.py"])
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        except Exception as e:
            print(f"\nError: {e}")
    
    elif choice == "4":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()