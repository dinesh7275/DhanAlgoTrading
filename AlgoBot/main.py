#!/usr/bin/env python3
"""
AlgoBot Main Entry Point
========================

Main script to run the web-based algorithmic trading platform.
Choose between web application or trading bot modes.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='AlgoBot - Web-Based Trading Platform')
    parser.add_argument('--mode', choices=['web', 'bot', 'setup'], 
                       default='web', help='Run mode: web app, trading bot, or setup')
    parser.add_argument('--host', default='localhost', help='Host for web app')
    parser.add_argument('--port', type=int, default=5000, help='Port for web app')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AlgoBot - Web-Based Trading Platform")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    
    if args.mode == 'web':
        print(f"Starting web application at http://{args.host}:{args.port}")
        print("="*60)
        
        # Import and run Flask app
        from app import app
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    elif args.mode == 'bot':
        print("Starting trading bot...")
        print("="*60)
        
        # Import and run trading bot
        from main_trading_bot import main as run_trading_bot
        run_trading_bot()
        
    elif args.mode == 'setup':
        print("Starting setup process...")
        print("="*60)
        
        # Import and run setup
        from quick_start import main as run_setup
        run_setup()
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()