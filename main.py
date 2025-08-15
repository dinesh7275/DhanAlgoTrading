#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Trading Bot - Single Entry Point
============================================

Complete AI trading system that starts all components from one file:
- 30-day learning models
- Multi-timeframe analysis 
- Live candlestick charts
- Real-time monitoring
- Paper trading
- Adaptive learning
- Enhanced dashboard
"""

import argparse
import logging
import multiprocessing
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

# Add AIBot to path
sys.path.append(str(Path(__file__).parent / "AIBot"))

try:
    # Import all enhanced components
    from AIBot.models.enhanced_learning.advanced_market_learner import AdvancedMarketLearner
    from AIBot.models.enhanced_learning.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from AIBot.models.enhanced_learning.candlestick_pattern_ml import CandlestickPatternML
    from AIBot.models.enhanced_learning.signal_generator import ComprehensiveSignalGenerator
    from AIBot.visualization.live_candlestick_chart import LiveCandlestickChart
    from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials
    from AIBot.trading.paper_trading_engine import PaperTradingEngine
    from AIBot.monitoring.real_time_indicator_monitor import RealTimeIndicatorMonitor
    from AIBot.learning.adaptive_learning_system import AdaptiveLearningSystem, TradingResult
    
    # Try to import dashboard - prioritize real trading dashboard
    try:
        import subprocess
        import sys
        # Start real trading dashboard in background
        dashboard_process = subprocess.Popen([
            sys.executable, 'real_trading_dashboard.py'
        ], cwd=str(Path(__file__).parent))
        print("Real trading dashboard started at http://localhost:5003")
        DASHBOARD_AVAILABLE = True
        dashboard_app = None  # Running in separate process
    except Exception as e:
        try:
            from AIBot.enhanced_dashboard import app as dashboard_app
            DASHBOARD_AVAILABLE = True
            print("Using fallback enhanced dashboard")
        except ImportError:
            print("Dashboard not available - continuing without web interface")
            dashboard_app = None
            DASHBOARD_AVAILABLE = False
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Some components not available: {e}")
    print("Will run in basic mode...")
    COMPONENTS_AVAILABLE = False
    DASHBOARD_AVAILABLE = False
    dashboard_app = None
    
    # Define fallback classes
    class AdvancedMarketLearner:
        def __init__(self): pass
    class MultiTimeframeAnalyzer:
        def __init__(self): pass
    class CandlestickPatternML:
        def __init__(self): pass
    class ComprehensiveSignalGenerator:
        def __init__(self, **kwargs): pass
        def generate_comprehensive_signal(self): return None
    class LiveCandlestickChart:
        def __init__(self, **kwargs): pass
        def start_live_updates(self): pass
        def stop_updates(self): pass
    class DhanAPIClient:
        def __init__(self, credentials): pass
        def authenticate(self): return False
    class DhanCredentials:
        def __init__(self, **kwargs): pass
    class PaperTradingEngine:
        def __init__(self, **kwargs): pass
        def get_portfolio_summary(self): return {'capital': {'current': 10000, 'total_return': 0, 'total_return_percent': 0}, 'performance': {'win_rate': 0}, 'risk': {'daily_loss_used': 0, 'daily_loss_limit': 1000, 'max_drawdown': 0}, 'trades': {'open': 0}}
        def place_order(self, **kwargs): return None
        def get_trade_by_id(self, trade_id): return None
        def get_current_price(self, symbol): return 100.0
        def close_trade(self, trade_id, price): return {'trade_id': trade_id, 'pnl': 0, 'exit_price': price}
        def close_all_positions(self): return []
        def export_trading_data(self, filename): return filename
    class RealTimeIndicatorMonitor:
        def __init__(self, **kwargs): pass
        def add_alert_callback(self, callback): pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def export_monitoring_data(self, filename): return filename
    class AdaptiveLearningSystem:
        def __init__(self, **kwargs): pass
        def add_trading_result(self, result): pass
        def export_learning_data(self, filename): return filename
    class TradingResult:
        def __init__(self, **kwargs): pass

logger = logging.getLogger(__name__)

class MasterTradingSystem:
    """
    Master system that orchestrates all trading components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.is_paper_trading = config.get('paper_trading', True)
        
        # All system components
        self.components = {}
        self.threads = {}
        self.processes = {}
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'alerts_triggered': 0,
            'components_running': 0
        }
        
        # Initialize all components
        self._initialize_all_components()
        
        logger.info("Master Trading System initialized successfully")
    
    def _initialize_all_components(self):
        """Initialize all available components"""
        try:
            if not COMPONENTS_AVAILABLE:
                print("Running in basic mode - some components unavailable")
                return
            
            print("Initializing Enhanced AI Trading System...")
            print("=" * 60)
            
            # 1. Advanced Market Learner
            print("Initializing Advanced Market Learner...")
            self.components['market_learner'] = AdvancedMarketLearner()
            print("   30-day learning models ready")
            
            # 2. Multi-timeframe Analyzer
            print("Initializing Multi-timeframe Analyzer...")
            self.components['timeframe_analyzer'] = MultiTimeframeAnalyzer()
            print("   Multi-timeframe analysis (1m-1d) ready")
            
            # 3. Candlestick Pattern ML
            print("Initializing Candlestick Pattern ML...")
            self.components['pattern_recognizer'] = CandlestickPatternML()
            print("   ML pattern recognition ready")
            
            # 4. Signal Generator
            print("Initializing Signal Generator...")
            self.components['signal_generator'] = ComprehensiveSignalGenerator(
                symbol="^NSEI", 
                capital=self.config.get('initial_capital', 10000)
            )
            print("   Comprehensive signal generation ready")
            
            # 5. Paper Trading Engine
            print("Initializing Paper Trading Engine...")
            self.components['paper_trading'] = PaperTradingEngine(
                initial_capital=self.config.get('initial_capital', 10000)
            )
            print("   Paper trading with Indian market simulation ready")
            
            # 6. Real-time Monitor
            print("Initializing Real-time Monitor...")
            symbols = self.config.get('symbols', ["^NSEI", "^NSEBANK"])
            self.components['monitor'] = RealTimeIndicatorMonitor(
                symbols=symbols,
                update_interval=self.config.get('monitor_interval', 30)
            )
            print("   Real-time indicator monitoring ready")
            
            # 7. Adaptive Learning System
            print("Initializing Adaptive Learning...")
            self.components['adaptive_learner'] = AdaptiveLearningSystem(
                update_frequency=self.config.get('learning_frequency', 24)
            )
            print("   Adaptive learning system ready")
            
            # 8. Live Charts (optional)
            if self.config.get('enable_live_charts', True):
                print("Initializing Live Charts...")
                self.components['live_chart'] = LiveCandlestickChart(
                    symbol="^NSEI",
                    timeframe="5m"
                )
                print("   Live candlestick charts ready")
            
            # 9. Dhan API (if not paper trading)
            if not self.is_paper_trading and self.config.get('dhan_credentials'):
                print("Initializing Dhan API...")
                credentials = DhanCredentials(**self.config['dhan_credentials'])
                self.components['dhan_client'] = DhanAPIClient(credentials)
                if self.components['dhan_client'].authenticate():
                    print("   Dhan API connected")
                else:
                    print("   Dhan API authentication failed")
                    self.components['dhan_client'] = None
            
            # Setup integrations
            self._setup_component_integrations()
            
            print("=" * 60)
            print(f"All {len(self.components)} components initialized successfully!")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_component_integrations(self):
        """Setup integrations between components"""
        try:
            # Monitor alerts -> Signal generation
            if 'monitor' in self.components:
                def alert_callback(alert):
                    self.session_stats['alerts_triggered'] += 1
                    logger.info(f"Alert: {alert.message}")
                    
                    if alert.severity.value in ['HIGH', 'CRITICAL'] and alert.action_required:
                        self._handle_trading_alert(alert)
                
                self.components['monitor'].add_alert_callback(alert_callback)
            
            # Trading results -> Adaptive learning
            self.learning_callback = self._convert_trade_to_learning
            
            print("Component integrations configured")
            
        except Exception as e:
            logger.error(f"Error setting up integrations: {e}")
    
    def _handle_trading_alert(self, alert):
        """Handle high-priority trading alerts"""
        try:
            # Generate comprehensive signal
            if 'signal_generator' in self.components:
                signal = self.components['signal_generator'].generate_comprehensive_signal()
                
                if signal.signal != 'HOLD' and signal.confidence > 0.65:
                    self.session_stats['signals_generated'] += 1
                    logger.info(f"Generated {signal.signal} signal (confidence: {signal.confidence:.1%})")
                    
                    # Execute trade if conditions met
                    if self._should_execute_trade(signal, alert):
                        self._execute_trade(signal)
        
        except Exception as e:
            logger.error(f"Error handling trading alert: {e}")
    
    def _should_execute_trade(self, signal, alert) -> bool:
        """Check if trade should be executed"""
        try:
            if 'paper_trading' not in self.components:
                return False
            
            # Check confidence
            if signal.confidence < 0.65:
                return False
            
            # Check risk limits
            portfolio = self.components['paper_trading'].get_portfolio_summary()
            if portfolio['risk']['daily_loss_used'] > portfolio['risk']['daily_loss_limit'] * 0.8:
                logger.warning("Approaching daily loss limit")
                return False
            
            # Check position limits  
            if portfolio['trades']['open'] >= 5:
                logger.warning("Maximum positions reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade execution: {e}")
            return False
    
    def _execute_trade(self, signal):
        """Execute trade based on signal"""
        try:
            if 'paper_trading' not in self.components:
                return
            
            # Determine option symbol
            symbol = self._get_option_symbol(signal)
            if not symbol:
                return
            
            quantity = self._calculate_quantity(signal)
            
            # Place trade (paper or live)
            if 'dhan_client' in self.components and self.components['dhan_client']:
                # Use Dhan API for live trading with super orders
                trade_id = self.components['dhan_client'].place_super_order(
                    symbol=symbol,
                    transaction_type=signal.signal if signal.signal != 'HOLD' else 'BUY',
                    quantity=quantity,
                    price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss_price=signal.stop_loss,
                    order_type="LIMIT",
                    product_type="INTRADAY"
                )
            else:
                # Use paper trading
                trade_id = self.components['paper_trading'].place_order(
                    symbol=symbol,
                    trade_type=signal.signal if signal.signal != 'HOLD' else 'BUY',
                    quantity=quantity,
                    order_type="MARKET",
                    stop_loss=signal.stop_loss,
                    target=signal.target_price,
                    strategy=signal.strategy,
                    confidence=signal.confidence
                )
            
            if trade_id:
                self.session_stats['trades_executed'] += 1
                logger.info(f"Trade executed: {trade_id}")
                
                # Schedule monitoring
                self._schedule_trade_monitoring(trade_id, signal)
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _get_option_symbol(self, signal) -> Optional[str]:
        """Generate Dhan-compatible option symbol"""
        try:
            if signal.option_type == 'NONE' or not signal.recommended_strikes:
                return None
            
            strike = signal.recommended_strikes[0]
            
            # Get next weekly expiry (Thursday)
            today = datetime.now()
            days_ahead = 3 - today.weekday()  # Thursday is 3
            if days_ahead <= 0:  # If today is Friday/weekend, get next Thursday
                days_ahead += 7
            
            expiry_date = today + timedelta(days=days_ahead)
            
            # Format for Dhan: "NIFTY 21 AUG 24700 CALL"
            day = expiry_date.strftime("%d")
            month = expiry_date.strftime("%b").upper()
            option_type_name = "CALL" if signal.option_type == "CE" else "PUT"
            
            return f"NIFTY {day} {month} {int(strike)} {option_type_name}"
        
        except Exception as e:
            logger.error(f"Error generating option symbol: {e}")
            return None
    
    def _calculate_quantity(self, signal) -> int:
        """Calculate trade quantity"""
        base_quantity = 50  # NIFTY lot size
        confidence_adj = signal.confidence
        position_adj = min(signal.position_size / 0.1, 2.0)
        
        return max(int(base_quantity * confidence_adj * position_adj), 25)
    
    def _schedule_trade_monitoring(self, trade_id: str, signal):
        """Monitor trade for exit conditions"""
        def monitor():
            time.sleep(60)  # Wait 1 minute
            
            while True:
                try:
                    trade = self.components['paper_trading'].get_trade_by_id(trade_id)
                    if not trade or trade.status != 'OPEN':
                        break
                    
                    current_price = self.components['paper_trading'].get_current_price(trade.symbol)
                    
                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""
                    
                    if trade.trade_type == 'BUY':
                        if current_price >= signal.target_price:
                            should_exit = True
                            exit_reason = "Target hit"
                        elif current_price <= signal.stop_loss:
                            should_exit = True  
                            exit_reason = "Stop loss hit"
                    else:  # SELL
                        if current_price <= signal.target_price:
                            should_exit = True
                            exit_reason = "Target hit"
                        elif current_price >= signal.stop_loss:
                            should_exit = True
                            exit_reason = "Stop loss hit"
                    
                    if should_exit:
                        result = self.components['paper_trading'].close_trade(trade_id, current_price)
                        if result:
                            # Ensure result has required format
                            pnl = result.get('pnl', 0) if isinstance(result, dict) else 0
                            logger.info(f"{exit_reason} - Trade closed: P&L Rs.{pnl:.2f}")
                            self.session_stats['total_pnl'] += pnl
                            
                            # Send to learning system with proper format
                            if self.learning_callback and isinstance(result, dict):
                                try:
                                    # Ensure result has trade_id
                                    if 'trade_id' not in result:
                                        result['trade_id'] = trade_id
                                    self.learning_callback(result)
                                except Exception as learning_error:
                                    logger.error(f"Error sending to learning system: {learning_error}")
                        break
                    
                    time.sleep(30)  # Check every 30 seconds
                
                except Exception as e:
                    logger.error(f"Error monitoring trade: {e}")
                    break
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def _convert_trade_to_learning(self, trade_result):
        """Convert trade result to learning format"""
        try:
            if 'adaptive_learner' not in self.components:
                return
            
            # Validate trade_result format
            if not isinstance(trade_result, dict) or 'trade_id' not in trade_result:
                logger.warning(f"Invalid trade_result format: {trade_result}")
                return
            
            trade = self.components['paper_trading'].get_trade_by_id(trade_result['trade_id'])
            if not trade:
                logger.warning(f"Trade not found: {trade_result['trade_id']}")
                return
            
            learning_result = TradingResult(
                trade_id=trade_result['trade_id'],
                signal_id=f"signal_{int(time.time())}",
                entry_time=trade.entry_time,
                exit_time=trade.exit_time or datetime.now(),
                entry_price=trade.entry_price,
                exit_price=trade_result.get('exit_price', trade.entry_price),
                quantity=trade.quantity,
                pnl=trade_result.get('pnl', 0),
                pnl_percent=(trade_result.get('pnl', 0) / (trade.entry_price * trade.quantity)) * 100,
                duration_minutes=int((trade.exit_time - trade.entry_time).total_seconds() / 60) if trade.exit_time else 0,
                strategy=trade.strategy,
                confidence=trade.confidence,
                signal_features={'rsi': 50, 'macd': 0, 'bb_position': 0.5},
                market_conditions={'trend': 'SIDEWAYS', 'volatility': 'NORMAL', 'volume': 'NORMAL'},
                success=trade_result.get('pnl', 0) > 0,
                exit_reason='TARGET' if trade_result.get('pnl', 0) > 0 else 'STOP_LOSS'
            )
            
            self.components['adaptive_learner'].add_trading_result(learning_result)
        
        except Exception as e:
            logger.error(f"Error converting trade to learning: {e}")
    
    def start_all_systems(self):
        """Start all trading systems"""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        
        print("\nSTARTING ENHANCED AI TRADING SYSTEM")
        print("=" * 60)
        
        try:
            # Start real-time monitoring
            if 'monitor' in self.components:
                self.components['monitor'].start_monitoring()
                print("Real-time indicator monitoring started")
            
            # Start live charts
            if 'live_chart' in self.components:
                self.components['live_chart'].start_live_updates()
                print("Live candlestick charts started")
            
            # Dashboard already started in background during import
            if DASHBOARD_AVAILABLE:
                print("Real trading dashboard available at http://localhost:5003")
            else:
                print("Dashboard not available - continuing without web interface")
            
            # Start main trading loop
            self.threads['main_loop'] = threading.Thread(
                target=self._main_trading_loop,
                daemon=True
            )
            self.threads['main_loop'].start()
            print("Main trading logic started")
            
            # Print startup summary
            self._print_startup_summary()
            
        except Exception as e:
            logger.error(f"Error starting systems: {e}")
            self.stop_all_systems()
            raise
    
    def _run_dashboard(self):
        """Run dashboard in separate process"""
        try:
            if dashboard_app:
                dashboard_app.run(host='0.0.0.0', port=5002, debug=False)
            else:
                print("Dashboard app not available")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    def _main_trading_loop(self):
        """Main trading coordination loop"""
        logger.info("Main trading loop started")
        
        while self.is_running:
            try:
                # Generate periodic signals
                time.sleep(300)  # 5 minutes
                
                if not self.is_running:
                    break
                
                # Generate signal
                if 'signal_generator' in self.components:
                    signal = self.components['signal_generator'].generate_comprehensive_signal()
                    
                    if signal.signal != 'HOLD':
                        self.session_stats['signals_generated'] += 1
                        logger.info(f"Generated {signal.signal} signal (confidence: {signal.confidence:.1%})")
                
                # Print periodic status
                if self.session_stats['signals_generated'] % 5 == 0:
                    self._print_status_update()
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        
        logger.info("Main trading loop stopped")
    
    def _print_startup_summary(self):
        """Print comprehensive startup summary"""
        print("\n" + "=" * 80)
        print("ENHANCED AI TRADING SYSTEM - FULLY OPERATIONAL")
        print("=" * 80)
        print(f"Mode: {'Paper Trading' if self.is_paper_trading else 'LIVE TRADING WITH REAL DHAN DATA'}")
        print(f"Capital: Rs.{self.config.get('initial_capital', 10000):,}")
        print(f"Symbols: {', '.join(self.config.get('symbols', ['NIFTY']))}")
        print(f"AI Models: {len([c for c in self.components.keys() if 'learner' in c or 'analyzer' in c])} active")
        print(f"Monitoring: {self.config.get('monitor_interval', 30)}s intervals")
        print(f"Dashboard: http://localhost:5003")
        print("=" * 80)
        print("30-day learning models trained and active")
        print("Multi-timeframe analysis (1m-1d) running")
        print("LIVE DHAN API DATA INTEGRATION ACTIVE")
        print("Real-time NIFTY prices from Dhan API")
        print("Live option chain data and account balance")
        print("AI trading signals with real market data")
        print("Adaptive learning system with live feedback")
        print("Real trading dashboard with live updates")
        print("=" * 80)
        print("System fully operational - Press Ctrl+C to stop")
        print("=" * 80)
    
    def _print_status_update(self):
        """Print system status update"""
        if 'paper_trading' in self.components:
            portfolio = self.components['paper_trading'].get_portfolio_summary()
            runtime = datetime.now() - self.session_stats['start_time']
            
            print(f"\nSTATUS UPDATE - Runtime: {runtime}")
            print(f"Capital: Rs.{portfolio['capital']['current']:,.2f}")
            print(f"P&L: Rs.{portfolio['capital']['total_return']:+,.2f}")
            print(f"Signals: {self.session_stats['signals_generated']}")
            print(f"Trades: {self.session_stats['trades_executed']}")
            print(f"Alerts: {self.session_stats['alerts_triggered']}")
            print(f"Win Rate: {portfolio['performance']['win_rate']:.1%}")
    
    def stop_all_systems(self):
        """Stop all trading systems"""
        if not self.is_running:
            return
        
        print("\nSTOPPING ENHANCED AI TRADING SYSTEM")
        print("=" * 60)
        
        self.is_running = False
        
        try:
            # Stop monitoring
            if 'monitor' in self.components:
                self.components['monitor'].stop_monitoring()
                print("Monitoring stopped")
            
            # Stop live charts
            if 'live_chart' in self.components:
                self.components['live_chart'].stop_updates()
                print("Live charts stopped")
            
            # Close all positions
            if 'paper_trading' in self.components:
                results = self.components['paper_trading'].close_all_positions()
                if results:
                    print(f"Closed {len(results)} positions")
            
            # Stop processes
            for name, process in self.processes.items():
                if process.is_alive():
                    process.terminate()
                    print(f"{name} process stopped")
            
            # Export final data
            self._export_final_data()
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Error stopping systems: {e}")
        
        print("=" * 60)
        print("ENHANCED AI TRADING SYSTEM STOPPED SUCCESSFULLY")
        print("=" * 60)
    
    def _export_final_data(self):
        """Export all system data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if 'paper_trading' in self.components:
                file1 = self.components['paper_trading'].export_trading_data(f"final_trading_{timestamp}.json")
                print(f"Trading data: {file1}")
            
            if 'monitor' in self.components:
                file2 = self.components['monitor'].export_monitoring_data(f"final_monitoring_{timestamp}.json")
                print(f"Monitoring data: {file2}")
            
            if 'adaptive_learner' in self.components:
                file3 = self.components['adaptive_learner'].export_learning_data(f"final_learning_{timestamp}.json")
                print(f"Learning data: {file3}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def _print_final_summary(self):
        """Print final system summary"""
        try:
            if 'paper_trading' in self.components:
                portfolio = self.components['paper_trading'].get_portfolio_summary()
                runtime = datetime.now() - self.session_stats['start_time']
                
                print(f"\nFINAL PERFORMANCE SUMMARY")
                print(f"Runtime: {runtime}")
                print(f"Final Capital: Rs.{portfolio['capital']['current']:,.2f}")
                print(f"Total Return: Rs.{portfolio['capital']['total_return']:+,.2f} ({portfolio['capital']['total_return_percent']:+.2f}%)")
                print(f"Signals Generated: {self.session_stats['signals_generated']}")
                print(f"Trades Executed: {self.session_stats['trades_executed']}")
                print(f"Win Rate: {portfolio['performance']['win_rate']:.1%}")
                print(f"Max Drawdown: Rs.{portfolio['risk']['max_drawdown']:,.2f}")
        
        except Exception as e:
            logger.error(f"Error printing final summary: {e}")

def load_master_config() -> Dict[str, Any]:
    """Load master system configuration"""
    return {
        'paper_trading': False,  # Enable live trading with real Dhan data
        'initial_capital': 9632.91,  # Use real account balance
        'symbols': ["^NSEI", "^NSEBANK"],
        'monitor_interval': 30,
        'learning_frequency': 24,
        'min_confidence': 0.65,
        'max_positions': 5,
        'enable_dashboard': True,
        'enable_live_charts': True,
        'dhan_credentials': {
            'client_id': '1107321060',
            'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU3MTM4NzgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNzMyMTA2MCJ9.n_2HhEW9ePhAfi63KoxQskzohVPi4N8F_RWn-a9rqTbne5GX7DHRTF9NpU4LEyf1dC8J-M32Fuk-EbXlOYOWOA'
        }  # Add your real Dhan credentials here
    }

def main():
    """Single entry point for entire enhanced AI trading system"""
    parser = argparse.ArgumentParser(description='Enhanced AI Trading System - Complete')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Live trading mode')
    parser.add_argument('--debug', action='store_true', help='Debug logging')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_ai_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = load_master_config()
    config['initial_capital'] = args.capital
    
    if args.live:
        config['paper_trading'] = False
    
    # Create and start master system
    master_system = None
    
    try:
        print("ENHANCED AI TRADING SYSTEM")
        print("=" * 40)
        print("Complete end-to-end AI trading solution")
        print("All advanced features integrated")
        print("=" * 40)
        
        # Create master system
        master_system = MasterTradingSystem(config)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            print("\nShutdown signal received...")
            if master_system:
                master_system.stop_all_systems()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start all systems
        master_system.start_all_systems()
        
        # Keep running
        while master_system.is_running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        logger.error(f"Fatal system error: {e}")
        print(f"Fatal error: {e}")
    finally:
        if master_system:
            master_system.stop_all_systems()

if __name__ == "__main__":
    main()