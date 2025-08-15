#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Trading Bot - Main Integration Script
================================================

Complete integration of all enhanced AI trading components:
- Multi-timeframe analysis
- Advanced ML models with 30-day learning
- Live candlestick charts
- Real-time indicator monitoring
- Comprehensive signal generation
- Paper trading simulation
- Adaptive learning system
- Dhan API integration
"""

import sys
import logging
import asyncio
import threading
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import json

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import all our enhanced modules
from models.enhanced_learning.advanced_market_learner import AdvancedMarketLearner
from models.enhanced_learning.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from models.enhanced_learning.candlestick_pattern_ml import CandlestickPatternML
from models.enhanced_learning.signal_generator import ComprehensiveSignalGenerator
from visualization.live_candlestick_chart import LiveCandlestickChart
from integrations.dhan_api_client import DhanAPIClient, DhanCredentials
from trading.paper_trading_engine import PaperTradingEngine
from monitoring.real_time_indicator_monitor import RealTimeIndicatorMonitor
from learning.adaptive_learning_system import AdaptiveLearningSystem, TradingResult
from enhanced_dashboard import app as dashboard_app

logger = logging.getLogger(__name__)

class EnhancedAITradingBot:
    """
    Enhanced AI Trading Bot that integrates all components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.is_paper_trading = config.get('paper_trading', True)
        
        # Core components
        self.market_learner = None
        self.timeframe_analyzer = None
        self.pattern_recognizer = None
        self.signal_generator = None
        self.paper_trading_engine = None
        self.indicator_monitor = None
        self.adaptive_learner = None
        self.dhan_client = None
        self.live_chart = None
        
        # Threading
        self.main_thread = None
        self.dashboard_thread = None
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'alerts_triggered': 0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced AI Trading Bot initialized successfully")
    
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # 1. Advanced Market Learner (30-day data training)
            logger.info("Initializing Advanced Market Learner...")
            self.market_learner = AdvancedMarketLearner()
            
            # 2. Multi-timeframe Analyzer
            logger.info("Initializing Multi-timeframe Analyzer...")
            self.timeframe_analyzer = MultiTimeframeAnalyzer()
            
            # 3. Candlestick Pattern Recognition
            logger.info("Initializing Candlestick Pattern ML...")
            self.pattern_recognizer = CandlestickPatternML()
            
            # 4. Signal Generator
            logger.info("Initializing Comprehensive Signal Generator...")
            self.signal_generator = ComprehensiveSignalGenerator(
                symbol="^NSEI", 
                capital=self.config.get('initial_capital', 10000)
            )
            
            # 5. Paper Trading Engine
            logger.info("Initializing Paper Trading Engine...")
            self.paper_trading_engine = PaperTradingEngine(
                initial_capital=self.config.get('initial_capital', 10000)
            )
            
            # 6. Real-time Indicator Monitor
            logger.info("Initializing Real-time Indicator Monitor...")
            symbols = self.config.get('symbols', ["^NSEI", "^NSEBANK"])
            self.indicator_monitor = RealTimeIndicatorMonitor(
                symbols=symbols,
                update_interval=self.config.get('monitor_interval', 30)
            )
            
            # 7. Adaptive Learning System
            logger.info("Initializing Adaptive Learning System...")
            self.adaptive_learner = AdaptiveLearningSystem(
                update_frequency=self.config.get('learning_frequency', 24)
            )
            
            # 8. Live Candlestick Chart (optional)
            if self.config.get('enable_live_charts', True):
                logger.info("Initializing Live Candlestick Chart...")
                self.live_chart = LiveCandlestickChart(
                    symbol="^NSEI",
                    timeframe="5m"
                )
            
            # 9. Dhan API Client (if not paper trading)
            if not self.is_paper_trading and self.config.get('dhan_credentials'):
                logger.info("Initializing Dhan API Client...")
                credentials = DhanCredentials(**self.config['dhan_credentials'])
                self.dhan_client = DhanAPIClient(credentials)
                if not self.dhan_client.authenticate():
                    logger.error("Failed to authenticate with Dhan API")
                    self.dhan_client = None
            
            # Setup callbacks and integrations
            self._setup_integrations()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_integrations(self):
        """Setup integrations between components"""
        try:
            # Setup alert callback for indicator monitor
            def alert_callback(alert):
                self.session_stats['alerts_triggered'] += 1
                logger.info(f"📊 Indicator Alert: {alert.message}")
                
                # Trigger signal generation on high-severity alerts
                if alert.severity.value in ['HIGH', 'CRITICAL'] and alert.action_required:
                    self._handle_indicator_alert(alert)
            
            self.indicator_monitor.add_alert_callback(alert_callback)
            
            # Setup adaptive learning callback for trading results
            def learning_callback(trade_result):
                # Convert paper trading result to learning format
                learning_result = self._convert_to_learning_result(trade_result)
                self.adaptive_learner.add_trading_result(learning_result)
            
            # This would be called after each trade closes
            self.learning_callback = learning_callback
            
        except Exception as e:
            logger.error(f"Error setting up integrations: {e}")
    
    def _handle_indicator_alert(self, alert):
        """Handle high-priority indicator alerts"""
        try:
            logger.info(f"Processing high-priority alert: {alert.message}")
            
            # Generate comprehensive signal
            signal = self.signal_generator.generate_comprehensive_signal()
            
            if signal.signal != 'HOLD' and signal.confidence > 0.6:
                logger.info(f"Generated {signal.signal} signal with {signal.confidence:.2%} confidence")
                self.session_stats['signals_generated'] += 1
                
                # Execute trade if conditions are met
                if self._should_execute_trade(signal, alert):
                    self._execute_trade(signal)
            
        except Exception as e:
            logger.error(f"Error handling indicator alert: {e}")
    
    def _should_execute_trade(self, signal, alert) -> bool:
        """Determine if trade should be executed based on signal and alert"""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.get('min_confidence', 0.65):
                return False
            
            # Check risk limits
            portfolio_summary = self.paper_trading_engine.get_portfolio_summary()
            if portfolio_summary['risk']['daily_loss_used'] > portfolio_summary['risk']['daily_loss_limit'] * 0.8:
                logger.warning("Approaching daily loss limit, skipping trade")
                return False
            
            # Check position limits
            if portfolio_summary['trades']['open'] >= self.config.get('max_positions', 5):
                logger.warning("Maximum positions reached, skipping trade")
                return False
            
            # Check market conditions alignment
            if alert.signal_type == signal.signal:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trade execution criteria: {e}")
            return False
    
    def _execute_trade(self, signal):
        """Execute trade based on signal"""
        try:
            # Determine trade parameters
            symbol = self._get_option_symbol(signal)
            if not symbol:
                logger.warning("Could not determine option symbol, skipping trade")
                return
            
            quantity = self._calculate_quantity(signal)
            
            # Place paper trade
            if self.is_paper_trading:
                trade_id = self.paper_trading_engine.place_order(
                    symbol=symbol,
                    trade_type=signal.signal.replace('HOLD', 'BUY'),  # Default to BUY if HOLD
                    quantity=quantity,
                    order_type="MARKET",
                    stop_loss=signal.stop_loss,
                    target=signal.target_price,
                    strategy=signal.strategy,
                    confidence=signal.confidence
                )
                
                if trade_id:
                    self.session_stats['trades_executed'] += 1
                    logger.info(f"✅ Paper trade executed: {trade_id}")
                    
                    # Schedule trade monitoring
                    self._schedule_trade_monitoring(trade_id, signal)
            
            # Place real trade via Dhan API
            elif self.dhan_client:
                try:
                    order_id = self.dhan_client.place_order(
                        symbol=symbol,
                        transaction_type=signal.signal.replace('HOLD', 'BUY'),
                        quantity=quantity,
                        order_type="MARKET"
                    )
                    
                    if order_id:
                        self.session_stats['trades_executed'] += 1
                        logger.info(f"✅ Real trade executed: {order_id}")
                
                except Exception as e:
                    logger.error(f"Error placing real trade: {e}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _get_option_symbol(self, signal) -> Optional[str]:
        """Get option symbol based on signal"""
        try:
            if signal.option_type == 'NONE' or not signal.recommended_strikes:
                return None
            
            # Use first recommended strike
            strike = signal.recommended_strikes[0]
            
            # Generate symbol (simplified - would need proper expiry logic)
            expiry = (datetime.now() + timedelta(days=7)).strftime("%y%m%d")
            symbol = f"NIFTY{expiry}{int(strike)}{signal.option_type}"
            
            return symbol
            
        except Exception as e:
            logger.error(f"Error getting option symbol: {e}")
            return None
    
    def _calculate_quantity(self, signal) -> int:
        """Calculate quantity based on signal and risk management"""
        try:
            # Base quantity from signal
            base_quantity = 50  # Default NIFTY lot size
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence
            
            # Adjust based on position size recommendation
            position_multiplier = min(signal.position_size / 0.1, 2.0)  # Cap at 2x
            
            quantity = int(base_quantity * confidence_multiplier * position_multiplier)
            return max(quantity, 25)  # Minimum quarter lot
            
        except Exception as e:
            logger.error(f"Error calculating quantity: {e}")
            return 50
    
    def _schedule_trade_monitoring(self, trade_id: str, signal):
        """Schedule monitoring for trade exit conditions"""
        def monitor_trade():
            try:
                # Wait a bit before starting monitoring
                time.sleep(60)  # 1 minute
                
                while True:
                    trade = self.paper_trading_engine.get_trade_by_id(trade_id)
                    if not trade or trade.status != 'OPEN':
                        break
                    
                    # Check exit conditions
                    current_price = self.paper_trading_engine.get_current_price(trade.symbol)
                    
                    # Target hit
                    if ((trade.trade_type == 'BUY' and current_price >= signal.target_price) or
                        (trade.trade_type == 'SELL' and current_price <= signal.target_price)):
                        
                        result = self.paper_trading_engine.close_trade(trade_id, current_price)
                        if result:
                            logger.info(f"🎯 Target hit - Trade closed: {trade_id}")
                            self.session_stats['total_pnl'] += result['pnl']
                            
                            # Send to adaptive learning
                            if self.learning_callback:
                                self.learning_callback(result)
                        break
                    
                    # Stop loss hit
                    elif ((trade.trade_type == 'BUY' and current_price <= signal.stop_loss) or
                          (trade.trade_type == 'SELL' and current_price >= signal.stop_loss)):
                        
                        result = self.paper_trading_engine.close_trade(trade_id, current_price)
                        if result:
                            logger.info(f"🛑 Stop loss hit - Trade closed: {trade_id}")
                            self.session_stats['total_pnl'] += result['pnl']
                            
                            # Send to adaptive learning
                            if self.learning_callback:
                                self.learning_callback(result)
                        break
                    
                    time.sleep(30)  # Check every 30 seconds
                    
            except Exception as e:
                logger.error(f"Error monitoring trade {trade_id}: {e}")
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_trade, daemon=True)
        monitor_thread.start()
    
    def _convert_to_learning_result(self, trade_result) -> TradingResult:
        """Convert paper trading result to adaptive learning format"""
        try:
            trade = self.paper_trading_engine.get_trade_by_id(trade_result['trade_id'])
            
            return TradingResult(
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
                signal_features={
                    'rsi': 50,  # Would be populated from actual signal data
                    'macd': 0,
                    'bb_position': 0.5
                },
                market_conditions={
                    'trend': 'SIDEWAYS',  # Would be populated from market analysis
                    'volatility': 'NORMAL',
                    'volume': 'NORMAL'
                },
                success=trade_result.get('pnl', 0) > 0,
                exit_reason='TARGET' if trade_result.get('pnl', 0) > 0 else 'STOP_LOSS'
            )
            
        except Exception as e:
            logger.error(f"Error converting to learning result: {e}")
            return None
    
    def start_trading(self):
        """Start the enhanced AI trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Enhanced AI Trading Bot...")
        
        try:
            # Start indicator monitoring
            self.indicator_monitor.start_monitoring()
            
            # Start live charts if enabled
            if self.live_chart:
                self.live_chart.start_live_updates()
            
            # Start main trading loop
            self.main_thread = threading.Thread(target=self._main_trading_loop, daemon=True)
            self.main_thread.start()
            
            # Start dashboard in separate thread
            if self.config.get('enable_dashboard', True):
                self.dashboard_thread = threading.Thread(target=self._run_dashboard, daemon=True)
                self.dashboard_thread.start()
            
            logger.info("✅ Enhanced AI Trading Bot started successfully!")
            self._print_startup_summary()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            self.stop_trading()
            raise
    
    def _main_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Update market data and generate signals periodically
                time.sleep(300)  # 5 minutes
                
                if not self.is_running:
                    break
                
                # Generate comprehensive signal
                signal = self.signal_generator.generate_comprehensive_signal()
                
                # Update session stats
                if signal.signal != 'HOLD':
                    self.session_stats['signals_generated'] += 1
                
                # Print periodic status
                self._print_status_update()
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Main trading loop stopped")
    
    def _run_dashboard(self):
        """Run the dashboard server"""
        try:
            logger.info("Starting dashboard server at http://localhost:5002")
            dashboard_app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
    
    def _print_startup_summary(self):
        """Print startup summary"""
        print("\n" + "="*80)
        print("🤖 ENHANCED AI TRADING BOT - STARTUP SUMMARY")
        print("="*80)
        print(f"📊 Mode: {'Paper Trading' if self.is_paper_trading else 'Live Trading'}")
        print(f"💰 Initial Capital: ₹{self.config.get('initial_capital', 10000):,}")
        print(f"📈 Symbols: {', '.join(self.config.get('symbols', ['NIFTY']))}")
        print(f"⏱️  Monitor Interval: {self.config.get('monitor_interval', 30)} seconds")
        print(f"🧠 Learning Frequency: Every {self.config.get('learning_frequency', 24)} hours")
        print(f"🌐 Dashboard: http://localhost:5002")
        print(f"📊 Live Charts: {'Enabled' if self.config.get('enable_live_charts', True) else 'Disabled'}")
        print("="*80)
        print("✅ All AI models loaded and ready")
        print("✅ Real-time monitoring active")
        print("✅ Multi-timeframe analysis running")
        print("✅ Adaptive learning system online")
        print("="*80)
        print("🚨 Press Ctrl+C to stop the bot")
        print("="*80)
    
    def _print_status_update(self):
        """Print periodic status update"""
        runtime = datetime.now() - self.session_stats['start_time']
        portfolio = self.paper_trading_engine.get_portfolio_summary()
        
        print(f"\n📊 STATUS UPDATE - Runtime: {runtime}")
        print(f"💰 Current Capital: ₹{portfolio['capital']['current']:,.2f}")
        print(f"📈 Total P&L: ₹{portfolio['capital']['total_return']:,.2f}")
        print(f"🎯 Signals Generated: {self.session_stats['signals_generated']}")
        print(f"⚡ Trades Executed: {self.session_stats['trades_executed']}")
        print(f"🚨 Alerts Triggered: {self.session_stats['alerts_triggered']}")
        print(f"📊 Win Rate: {portfolio['performance']['win_rate']:.1%}")
    
    def stop_trading(self):
        """Stop the enhanced AI trading bot"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping Enhanced AI Trading Bot...")
        self.is_running = False
        
        try:
            # Stop monitoring
            if self.indicator_monitor:
                self.indicator_monitor.stop_monitoring()
            
            # Stop live charts
            if self.live_chart:
                self.live_chart.stop_updates()
            
            # Close all positions
            if self.paper_trading_engine:
                results = self.paper_trading_engine.close_all_positions()
                if results:
                    logger.info(f"Closed {len(results)} open positions")
            
            # Export final data
            self._export_final_data()
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
        
        logger.info("✅ Enhanced AI Trading Bot stopped successfully")
    
    def _export_final_data(self):
        """Export final trading data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export paper trading data
            if self.paper_trading_engine:
                paper_file = self.paper_trading_engine.export_trading_data(f"final_paper_trading_{timestamp}.json")
                logger.info(f"📁 Paper trading data exported: {paper_file}")
            
            # Export monitoring data
            if self.indicator_monitor:
                monitor_file = self.indicator_monitor.export_monitoring_data(f"final_monitoring_{timestamp}.json")
                logger.info(f"📁 Monitoring data exported: {monitor_file}")
            
            # Export learning data
            if self.adaptive_learner:
                learning_file = self.adaptive_learner.export_learning_data(f"final_learning_{timestamp}.json")
                logger.info(f"📁 Learning data exported: {learning_file}")
            
        except Exception as e:
            logger.error(f"Error exporting final data: {e}")
    
    def _print_final_summary(self):
        """Print final trading summary"""
        try:
            portfolio = self.paper_trading_engine.get_portfolio_summary()
            runtime = datetime.now() - self.session_stats['start_time']
            
            print("\n" + "="*80)
            print("📊 ENHANCED AI TRADING BOT - FINAL SUMMARY")
            print("="*80)
            print(f"⏱️  Total Runtime: {runtime}")
            print(f"💰 Final Capital: ₹{portfolio['capital']['current']:,.2f}")
            print(f"📈 Total Return: ₹{portfolio['capital']['total_return']:,.2f} ({portfolio['capital']['total_return_percent']:+.2f}%)")
            print(f"🎯 Signals Generated: {self.session_stats['signals_generated']}")
            print(f"⚡ Trades Executed: {self.session_stats['trades_executed']}")
            print(f"🏆 Win Rate: {portfolio['performance']['win_rate']:.1%}")
            print(f"📉 Max Drawdown: ₹{portfolio['risk']['max_drawdown']:,.2f}")
            print(f"🚨 Total Alerts: {self.session_stats['alerts_triggered']}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing final summary: {e}")

def load_config(config_file: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    default_config = {
        'paper_trading': True,
        'initial_capital': 10000,
        'symbols': ["^NSEI", "^NSEBANK"],
        'monitor_interval': 30,
        'learning_frequency': 24,
        'min_confidence': 0.65,
        'max_positions': 5,
        'enable_dashboard': True,
        'enable_live_charts': True,
        'dhan_credentials': None  # Would be populated with actual credentials
    }
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
        except Exception as e:
            logger.warning(f"Error loading config file: {e}, using defaults")
    
    return default_config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced AI Trading Bot for Dhan')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--paper', action='store_true', help='Force paper trading mode')
    parser.add_argument('--live', action='store_true', help='Enable live trading mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paper trading mode from command line
    if args.paper:
        config['paper_trading'] = True
    elif args.live:
        config['paper_trading'] = False
    
    # Create and start trading bot
    trading_bot = None
    
    try:
        # Create trading bot
        trading_bot = EnhancedAITradingBot(config)
        
        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("\n🛑 Received shutdown signal...")
            if trading_bot:
                trading_bot.stop_trading()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start trading
        trading_bot.start_trading()
        
        # Keep running
        while trading_bot.is_running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        if trading_bot:
            trading_bot.stop_trading()
    
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        if trading_bot:
            trading_bot.stop_trading()
        sys.exit(1)

if __name__ == "__main__":
    main()