#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Trading Bot - Single Entry Point (Fixed)
===================================================

Complete AI trading system that starts all components from one file:
- 30-day learning models
- Multi-timeframe analysis  
- Live candlestick charts
- Real-time monitoring
- Paper trading
- Adaptive learning
- Enhanced dashboard
"""

import sys
import os
import logging
import threading
import time
import signal
import subprocess
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import json

# Add AIBot to path
sys.path.append(str(Path(__file__).parent / "AIBot"))

# Install missing dependencies
def install_requirements():
    """Install missing dependencies"""
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    except:
        pass

try:
    # Try importing advanced components
    from AIBot.models.enhanced_learning.signal_generator import ComprehensiveSignalGenerator
    from AIBot.trading.paper_trading_engine import PaperTradingEngine
    from AIBot.enhanced_dashboard import app as dashboard_app
    
    COMPONENTS_AVAILABLE = True
    print("Advanced components loaded successfully")
except ImportError as e:
    print(f"Some advanced components not available: {e}")
    print("Running in simplified mode...")
    COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimplifiedTradingSystem:
    """
    Simplified but complete trading system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        
        # Core components
        self.components = {}
        self.processes = {}
        
        # Stats
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0
        }
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize available components"""
        try:
            print("Initializing AI Trading System...")
            print("=" * 50)
            
            if COMPONENTS_AVAILABLE:
                # Signal Generator
                print("Initializing Signal Generator...")
                self.components['signal_generator'] = ComprehensiveSignalGenerator(
                    symbol="^NSEI", 
                    capital=self.config.get('initial_capital', 10000)
                )
                print("   Signal generation ready")
                
                # Paper Trading
                print("Initializing Paper Trading Engine...")
                self.components['paper_trading'] = PaperTradingEngine(
                    initial_capital=self.config.get('initial_capital', 10000)
                )
                print("   Paper trading ready")
                
                print("=" * 50)
                print(f"System ready with {len(self.components)} components")
            else:
                print("Running basic signal generation...")
                self._setup_basic_components()
                
        except Exception as e:
            logger.error(f"Error initializing: {e}")
            self._setup_basic_components()
    
    def _setup_basic_components(self):
        """Setup basic components if advanced ones fail"""
        from AIBot.main_simplified_trading_bot import SimpleSignalGenerator, SimplePaperTrading
        
        print("Setting up basic components...")
        self.components['signal_generator'] = SimpleSignalGenerator()
        self.components['paper_trading'] = SimplePaperTrading(
            initial_capital=self.config.get('initial_capital', 10000)
        )
        print("Basic components ready")
    
    def start_system(self):
        """Start the trading system"""
        self.is_running = True
        
        print("\nSTARTING AI TRADING SYSTEM")
        print("=" * 40)
        print(f"Mode: Paper Trading")
        print(f"Capital: Rs.{self.config.get('initial_capital', 10000):,}")
        print(f"Dashboard: http://localhost:5002")
        print("=" * 40)
        
        try:
            # Start dashboard if available
            if COMPONENTS_AVAILABLE and dashboard_app:
                self.processes['dashboard'] = multiprocessing.Process(
                    target=self._run_dashboard,
                    daemon=True
                )
                self.processes['dashboard'].start()
                print("Dashboard started at http://localhost:5002")
            
            # Start main trading loop
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop_system()
    
    def _run_dashboard(self):
        """Run dashboard"""
        try:
            dashboard_app.run(host='0.0.0.0', port=5002, debug=False)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        print("\nMain trading loop started...")
        print("Press Ctrl+C to stop")
        print("-" * 40)
        
        while self.is_running:
            try:
                # Generate signal
                if 'signal_generator' in self.components:
                    if hasattr(self.components['signal_generator'], 'generate_comprehensive_signal'):
                        signal = self.components['signal_generator'].generate_comprehensive_signal()
                        signal_data = {
                            'signal': signal.signal,
                            'confidence': signal.confidence,
                            'price': signal.entry_price,
                            'reason': ' | '.join(signal.reasoning[:2]) if signal.reasoning else 'AI Analysis'
                        }
                    else:
                        signal_data = self.components['signal_generator'].generate_simple_signal()
                    
                    self.session_stats['signals_generated'] += 1
                    
                    # Print signal
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] Signal: {signal_data['signal']} "
                          f"(Confidence: {signal_data['confidence']:.1%})")
                    print(f"   Price: Rs.{signal_data['price']:.2f}")
                    print(f"   Reason: {signal_data['reason']}")
                    
                    # Execute trade if confidence is high
                    if signal_data['confidence'] > 0.6 and signal_data['signal'] != 'HOLD':
                        success = False
                        if hasattr(self.components['paper_trading'], 'place_order'):
                            # Advanced paper trading
                            try:
                                trade_id = self.components['paper_trading'].place_order(
                                    symbol="NIFTY50",
                                    trade_type=signal_data['signal'],
                                    quantity=50,
                                    order_type="MARKET",
                                    strategy="AI Signal"
                                )
                                success = bool(trade_id)
                            except:
                                success = False
                        else:
                            # Basic paper trading
                            success = self.components['paper_trading'].place_trade(signal_data)
                        
                        if success:
                            self.session_stats['trades_executed'] += 1
                            print(f"   -> Trade executed: {signal_data['signal']}")
                    
                    # Update positions if available
                    if hasattr(self.components['paper_trading'], 'update_positions'):
                        self.components['paper_trading'].update_positions(signal_data['price'])
                    
                    # Print summary every 5 signals
                    if self.session_stats['signals_generated'] % 5 == 0:
                        self._print_summary()
                    
                    print("-" * 40)
                
                # Wait before next signal
                time.sleep(60)  # 1 minute intervals
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(30)
        
        print("\nTrading loop stopped")
    
    def _print_summary(self):
        """Print trading summary"""
        try:
            runtime = datetime.now() - self.session_stats['start_time']
            
            print("\nTRADING SUMMARY")
            print("-" * 30)
            print(f"Runtime: {runtime}")
            print(f"Signals Generated: {self.session_stats['signals_generated']}")
            print(f"Trades Executed: {self.session_stats['trades_executed']}")
            
            # Get detailed summary if available
            if hasattr(self.components['paper_trading'], 'get_portfolio_summary'):
                summary = self.components['paper_trading'].get_portfolio_summary()
                print(f"Current Capital: Rs.{summary['capital']['current']:,.2f}")
                print(f"Total P&L: Rs.{summary['capital']['total_return']:+,.2f}")
                print(f"Win Rate: {summary['performance']['win_rate']:.1%}")
            elif hasattr(self.components['paper_trading'], 'get_summary'):
                summary = self.components['paper_trading'].get_summary()
                print(f"Current Capital: Rs.{summary['current_capital']:,.2f}")
                print(f"Total P&L: Rs.{summary['total_pnl']:+,.2f}")
                print(f"Win Rate: {summary['win_rate']:.1f}%")
            
            print("-" * 30)
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
    
    def stop_system(self):
        """Stop the system"""
        print("\nSTOPPING AI TRADING SYSTEM")
        print("=" * 40)
        
        self.is_running = False
        
        try:
            # Close positions if available
            if 'paper_trading' in self.components:
                if hasattr(self.components['paper_trading'], 'close_all_positions'):
                    results = self.components['paper_trading'].close_all_positions()
                    if results:
                        print(f"Closed {len(results)} positions")
            
            # Stop processes
            for name, process in self.processes.items():
                if process.is_alive():
                    process.terminate()
                    print(f"{name} stopped")
            
            # Final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Error stopping: {e}")
        
        print("=" * 40)
        print("AI TRADING SYSTEM STOPPED")
        print("=" * 40)
    
    def _print_final_summary(self):
        """Print final summary"""
        try:
            runtime = datetime.now() - self.session_stats['start_time']
            
            print(f"\nFINAL SUMMARY")
            print(f"Total Runtime: {runtime}")
            print(f"Signals Generated: {self.session_stats['signals_generated']}")
            print(f"Trades Executed: {self.session_stats['trades_executed']}")
            
            if 'paper_trading' in self.components:
                if hasattr(self.components['paper_trading'], 'get_portfolio_summary'):
                    summary = self.components['paper_trading'].get_portfolio_summary()
                    print(f"Final Capital: Rs.{summary['capital']['current']:,.2f}")
                    print(f"Total Return: Rs.{summary['capital']['total_return']:+,.2f} ({summary['capital']['total_return_percent']:+.2f}%)")
                elif hasattr(self.components['paper_trading'], 'get_summary'):
                    summary = self.components['paper_trading'].get_summary()
                    print(f"Final Capital: Rs.{summary['current_capital']:,.2f}")
                    print(f"Total Return: Rs.{summary['total_pnl']:+,.2f}")
            
        except Exception as e:
            logger.error(f"Error in final summary: {e}")

def load_config() -> Dict[str, Any]:
    """Load configuration"""
    return {
        'initial_capital': 10000,
        'symbols': ["^NSEI"],
        'paper_trading': True
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced AI Trading System')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--debug', action='store_true', help='Debug logging')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load config
    config = load_config()
    config['initial_capital'] = args.capital
    
    # Create and start system
    trading_system = None
    
    try:
        print("ENHANCED AI TRADING SYSTEM")
        print("=" * 30)
        print("Complete trading solution")
        print("=" * 30)
        
        # Create system
        trading_system = SimplifiedTradingSystem(config)
        
        # Signal handlers
        def signal_handler(signum, frame):
            print("\nShutdown signal received...")
            if trading_system:
                trading_system.stop_system()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start system
        trading_system.start_system()
    
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
    finally:
        if trading_system:
            trading_system.stop_system()

if __name__ == "__main__":
    main()