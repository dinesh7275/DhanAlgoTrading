"""
Live Trading Strategy Manager
============================

Execute live trades based on AI ensemble signals using Dhan API
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import logging
from dhanhq import dhanhq
import warnings
warnings.filterwarnings('ignore')

from .dhan_data_fetcher import DhanLiveDataFetcher, MarketDataProcessor
from .ai_ensemble import TradingSignalEnsemble


class LiveTradingManager:
    """
    Manage live trading operations
    """
    
    def __init__(self, client_id, access_token, initial_capital=1000000, max_risk_per_trade=0.02):
        self.client_id = client_id
        self.access_token = access_token
        self.dhan = dhanhq(client_id, access_token)
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = 5
        
        # Components
        self.data_fetcher = DhanLiveDataFetcher(client_id, access_token)
        self.data_processor = MarketDataProcessor()
        self.ai_ensemble = TradingSignalEnsemble()
        
        # Trading state
        self.active_positions = {}
        self.pending_orders = {}
        self.trade_history = []
        self.strategy_performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0
        }
        
        # Control flags
        self.is_trading_active = False
        self.market_hours_only = True
        self.paper_trading = True  # Set to False for live trading
        
        # Setup logging
        self._setup_logging()
        
        print("🤖 Live Trading Manager Initialized")
        print(f"💰 Initial Capital: ₹{initial_capital:,}")
        print(f"⚠️  Paper Trading: {self.paper_trading}")
    
    def _setup_logging(self):
        """Setup logging for trades and strategies"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_trading_setup(self):
        """Validate all components before starting trading"""
        print("🔍 Validating trading setup...")
        
        validation_results = {
            'dhan_connection': False,
            'data_fetcher': False,
            'ai_models': False,
            'account_info': False
        }
        
        try:
            # Test Dhan connection
            validation_results['dhan_connection'] = self.data_fetcher.validate_connection()
            
            # Test data fetching
            market_data = self.data_fetcher.get_comprehensive_market_data()
            validation_results['data_fetcher'] = market_data is not None
            
            # Test AI models
            model_health = self.ai_ensemble.get_model_health()
            validation_results['ai_models'] = model_health['overall_health'] in ['HEALTHY', 'DEGRADED']
            
            # Get account info
            account_info = self.data_fetcher.get_account_info()
            validation_results['account_info'] = account_info is not None
            
            if account_info and account_info.get('funds'):
                self.current_capital = account_info['funds'].get('availabelBalance', self.initial_capital)
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
        
        # Print validation results
        for component, status in validation_results.items():
            status_emoji = "✅" if status else "❌"
            print(f"{status_emoji} {component}: {'OK' if status else 'FAILED'}")
        
        all_valid = all(validation_results.values())
        print(f"\n{'🟢 All systems ready!' if all_valid else '🔴 Some components failed validation'}")
        
        return all_valid
    
    def is_market_open(self):
        """Check if market is open for trading"""
        now = datetime.now()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() > 4:  # Saturday or Sunday
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def execute_order(self, order_details):
        """Execute order through Dhan API"""
        try:
            if self.paper_trading:
                return self._execute_paper_order(order_details)
            
            # Real order execution
            order_response = self.dhan.place_order(
                security_id=order_details['security_id'],
                exchange_segment=order_details['exchange'],
                transaction_type=order_details['transaction_type'],  # BUY/SELL
                quantity=order_details['quantity'],
                order_type=order_details['order_type'],  # MARKET/LIMIT
                product_type=order_details['product_type'],  # INTRADAY/DELIVERY
                price=order_details.get('price', 0)
            )
            
            if order_response and order_response.get('status') == 'success':
                order_id = order_response.get('data', {}).get('orderId')
                self.pending_orders[order_id] = {
                    'order_details': order_details,
                    'timestamp': datetime.now(),
                    'status': 'PENDING'
                }
                
                self.logger.info(f"Order placed: {order_id} - {order_details}")
                return order_id
            else:
                self.logger.error(f"Order failed: {order_response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return None
    
    def _execute_paper_order(self, order_details):
        """Execute paper trade (simulation)"""
        order_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.pending_orders)}"
        
        # Simulate order execution with slight slippage
        executed_price = order_details.get('price', 0)
        if order_details['order_type'] == 'MARKET':
            slippage = 0.001  # 0.1% slippage
            if order_details['transaction_type'] == 'BUY':
                executed_price = executed_price * (1 + slippage)
            else:
                executed_price = executed_price * (1 - slippage)
        
        # Create position
        position = {
            'order_id': order_id,
            'symbol': order_details['symbol'],
            'quantity': order_details['quantity'],
            'side': order_details['transaction_type'],
            'entry_price': executed_price,
            'current_price': executed_price,
            'timestamp': datetime.now(),
            'stop_loss': order_details.get('stop_loss'),
            'target': order_details.get('target'),
            'status': 'OPEN'
        }
        
        self.active_positions[order_id] = position
        
        self.logger.info(f"📄 PAPER TRADE: {order_details['transaction_type']} {order_details['quantity']} {order_details['symbol']} @ ₹{executed_price:.2f}")
        
        return order_id
    
    def update_positions(self, current_market_data):
        """Update all active positions with current market data"""
        current_price = current_market_data.get('nifty', {}).get('ltp', 0)
        
        for order_id, position in list(self.active_positions.items()):
            if position['status'] != 'OPEN':
                continue
            
            # Update current price
            position['current_price'] = current_price
            
            # Calculate P&L
            if position['side'] == 'BUY':
                position['pnl'] = (current_price - position['entry_price']) * position['quantity']
            else:  # SELL
                position['pnl'] = (position['entry_price'] - current_price) * position['quantity']
            
            position['pnl_percent'] = (position['pnl'] / (position['entry_price'] * position['quantity'])) * 100
            
            # Check stop loss and target
            should_close = False
            close_reason = ""
            
            if position['stop_loss']:
                if ((position['side'] == 'BUY' and current_price <= position['stop_loss']) or
                    (position['side'] == 'SELL' and current_price >= position['stop_loss'])):
                    should_close = True
                    close_reason = "Stop Loss Hit"
            
            if not should_close and position['target']:
                if ((position['side'] == 'BUY' and current_price >= position['target']) or
                    (position['side'] == 'SELL' and current_price <= position['target'])):
                    should_close = True
                    close_reason = "Target Reached"
            
            # Close position if needed
            if should_close:
                self._close_position(order_id, close_reason, current_price)
    
    def _close_position(self, order_id, reason, exit_price):
        """Close a position"""
        if order_id not in self.active_positions:
            return
        
        position = self.active_positions[order_id]
        
        # Calculate final P&L
        if position['side'] == 'BUY':
            final_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            final_pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Update position
        position['exit_price'] = exit_price
        position['final_pnl'] = final_pnl
        position['close_reason'] = reason
        position['exit_timestamp'] = datetime.now()
        position['status'] = 'CLOSED'
        
        # Update strategy performance
        self.strategy_performance['total_trades'] += 1
        self.strategy_performance['total_pnl'] += final_pnl
        self.current_capital += final_pnl
        
        if final_pnl > 0:
            self.strategy_performance['winning_trades'] += 1
        
        # Add to trade history
        self.trade_history.append(position.copy())
        
        # Remove from active positions
        del self.active_positions[order_id]
        
        self.logger.info(f"🔄 POSITION CLOSED: {position['symbol']} - {reason} - P&L: ₹{final_pnl:.2f}")
    
    def generate_trading_signal(self):
        """Generate trading signal using AI ensemble"""
        try:
            # Get current market data
            market_data = self.data_fetcher.get_comprehensive_market_data()
            if not market_data or not market_data['nifty']:
                return None
            
            # Process data for AI models
            self.data_processor.add_data_point(market_data)
            processed_features = self.data_processor.calculate_features()
            
            if not processed_features:
                return None
            
            # Generate ensemble signal
            signal_result = self.ai_ensemble.generate_ensemble_signal(
                market_data=processed_features,
                portfolio_value=self.current_capital,
                current_positions=list(self.active_positions.values()),
                options_data=market_data.get('options_chain')
            )
            
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None
    
    def process_trading_signal(self, signal_result):
        """Process trading signal and execute trades if appropriate"""
        if not signal_result:
            return
        
        try:
            signal = signal_result['final_signal']
            confidence = signal_result['confidence']
            current_price = signal_result['market_data'].get('close', 0)
            
            # Check if we should trade
            if confidence < 0.6:
                self.logger.info(f"Signal confidence too low: {confidence:.2f}")
                return
            
            if len(self.active_positions) >= self.max_positions:
                self.logger.info(f"Maximum positions reached: {len(self.active_positions)}")
                return
            
            # Get position recommendation
            recommendation = self.ai_ensemble.get_position_recommendation(
                signal_result, current_price, self.current_capital
            )
            
            if recommendation['action'] in ['BUY', 'SELL']:
                self._execute_signal(recommendation, current_price)
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {e}")
    
    def _execute_signal(self, recommendation, current_price):
        """Execute trading signal"""
        try:
            action = recommendation['action']
            quantity = min(recommendation.get('quantity', 0), 100)  # Limit quantity
            
            if quantity <= 0:
                return
            
            # Create order details
            order_details = {
                'symbol': 'NIFTY',
                'security_id': '26000',  # Nifty 50 security ID
                'exchange': 'NSE_EQ',
                'transaction_type': action,
                'quantity': quantity,
                'order_type': 'MARKET',
                'product_type': 'INTRADAY',
                'price': current_price,
                'stop_loss': recommendation.get('stop_loss'),
                'target': recommendation.get('target_price')
            }
            
            # Execute order
            order_id = self.execute_order(order_details)
            
            if order_id:
                self.logger.info(f"🎯 SIGNAL EXECUTED: {action} {quantity} NIFTY @ ₹{current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def start_live_trading(self, update_interval=30):
        """Start live trading loop"""
        if not self.validate_trading_setup():
            print("❌ Cannot start trading - validation failed")
            return
        
        self.is_trading_active = True
        print(f"\n🚀 Starting Live Trading")
        print(f"⏱️  Update Interval: {update_interval} seconds")
        print(f"💰 Current Capital: ₹{self.current_capital:,}")
        print(f"📊 Paper Trading: {self.paper_trading}")
        
        try:
            while self.is_trading_active:
                # Check market hours
                if self.market_hours_only and not self.is_market_open():
                    print("🕐 Market closed - waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Get current market data and update positions
                market_data = self.data_fetcher.get_comprehensive_market_data()
                if market_data:
                    self.update_positions(market_data)
                
                # Generate and process trading signals
                signal_result = self.generate_trading_signal()
                if signal_result:
                    self.process_trading_signal(signal_result)
                
                # Print status update
                self._print_status_update()
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop trading and close all positions"""
        self.is_trading_active = False
        
        print("\n🛑 Stopping live trading...")
        
        # Close all open positions
        if self.active_positions:
            current_market_data = self.data_fetcher.get_comprehensive_market_data()
            current_price = current_market_data.get('nifty', {}).get('ltp', 0)
            
            for order_id in list(self.active_positions.keys()):
                self._close_position(order_id, "Trading Stopped", current_price)
        
        # Print final summary
        self._print_final_summary()
    
    def _print_status_update(self):
        """Print current status"""
        print(f"\n{'='*60}")
        print(f"📊 Trading Status Update - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"💰 Current Capital: ₹{self.current_capital:,.2f}")
        print(f"📈 Total P&L: ₹{self.strategy_performance['total_pnl']:,.2f}")
        print(f"🎯 Active Positions: {len(self.active_positions)}")
        print(f"📊 Total Trades: {self.strategy_performance['total_trades']}")
        
        if self.strategy_performance['total_trades'] > 0:
            win_rate = (self.strategy_performance['winning_trades'] / self.strategy_performance['total_trades']) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        # Show active positions
        if self.active_positions:
            print(f"\n📋 Active Positions:")
            for order_id, position in self.active_positions.items():
                pnl_color = "🟢" if position.get('pnl', 0) > 0 else "🔴"
                print(f"{pnl_color} {position['side']} {position['quantity']} @ ₹{position['entry_price']:.2f} | P&L: ₹{position.get('pnl', 0):.2f}")
    
    def _print_final_summary(self):
        """Print final trading summary"""
        print(f"\n{'='*60}")
        print(f"📊 FINAL TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,}")
        print(f"💰 Final Capital: ₹{self.current_capital:,}")
        print(f"📈 Total P&L: ₹{self.strategy_performance['total_pnl']:,}")
        print(f"📊 Total Trades: {self.strategy_performance['total_trades']}")
        print(f"🎯 Winning Trades: {self.strategy_performance['winning_trades']}")
        
        if self.strategy_performance['total_trades'] > 0:
            win_rate = (self.strategy_performance['winning_trades'] / self.strategy_performance['total_trades']) * 100
            return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"📈 Return: {return_pct:+.2f}%")
        
        # Save trading log
        self._save_trading_log()
    
    def _save_trading_log(self):
        """Save trading history to file"""
        try:
            log_data = {
                'trading_session': {
                    'start_time': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'performance': self.strategy_performance
                },
                'trade_history': self.trade_history,
                'ai_signals': [signal for signal in self.ai_ensemble.signals_history]
            }
            
            filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            print(f"📁 Trading log saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving trading log: {e}")


# Quick start function
def start_live_trading_bot(client_id, access_token, paper_trading=True):
    """
    Quick start function for live trading
    """
    print("🤖 Initializing Live Trading Bot...")
    
    # Create trading manager
    trading_manager = LiveTradingManager(
        client_id=client_id,
        access_token=access_token,
        initial_capital=1000000,
        max_risk_per_trade=0.02
    )
    
    # Set paper trading mode
    trading_manager.paper_trading = paper_trading
    
    # Start trading
    try:
        trading_manager.start_live_trading(update_interval=30)
    except KeyboardInterrupt:
        print("\n👋 Trading stopped by user")
    except Exception as e:
        print(f"❌ Trading error: {e}")
    
    return trading_manager