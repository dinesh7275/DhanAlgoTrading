#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIBot Startup Script
===================

Simple script to run AIBot with dependency checking
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'yfinance', 
        'scipy', 'matplotlib', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    print("="*60)
    print("AIBot - AI-Powered Trading System")
    print("="*60)
    
    # Change to AIBot directory
    aibot_dir = Path(__file__).parent
    os.chdir(aibot_dir)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    print("All dependencies found!")
    print()
    
    # Import and run main
    try:
        from main import main as run_main
        run_main()
    except KeyboardInterrupt:
        print("\nAIBot stopped by user.")
    except Exception as e:
        print(f"Error running AIBot: {e}")
        print("Please check the configuration and try again.")

if __name__ == "__main__":
    main()