#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Trading Bot Launcher
===========================
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Simple AI Trading Bot")
    print("====================")
    print()
    print("1. Run Trading Bot")
    print("2. View Dashboard")
    print("3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    bot_dir = Path(__file__).parent / "AIBot"
    
    if choice == "1":
        print("\nStarting Simple Trading Bot...")
        print("- Basic NIFTY analysis")
        print("- Paper trading simulation")
        print("- Rs.10,000 virtual capital")
        print("\nPress Ctrl+C to stop")
        print()
        
        try:
            os.chdir(bot_dir)
            subprocess.run([sys.executable, "main_simplified_trading_bot.py", "--debug"])
        except KeyboardInterrupt:
            print("\nBot stopped")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == "2":
        print("\nStarting Dashboard at http://localhost:5001")
        try:
            os.chdir(bot_dir)
            subprocess.run([sys.executable, "simple_dashboard.py"])
        except KeyboardInterrupt:
            print("\nDashboard stopped")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice")
        main()

if __name__ == "__main__":
    main()