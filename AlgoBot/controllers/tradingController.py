#!/usr/bin/env python3
"""
Trading Controller
=================

Main controller for handling trading operations and coordination
"""

import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

from services.tradingService import TradingService
from live_trading.options_trading_manager import IndianOptionsTrader
from live_trading.ai_ensemble import TradingSignalEnsemble
from live_trading.risk_manager import LiveRiskManager
from live_trading.dhan_data_fetcher import DhanLiveDataFetcher
from config.config import trading_config, Config

logger = logging.getLogger(__name__)

class TradingController:
    """
    Main controller coordinating all trading operations
    """
    
    def __init__(self):
        self.trading_service = TradingService()
        self.config = trading_config
        self.trading_active = False
        self.options_trader = None
        self.ai_ensemble = None
        self.risk_manager = None
        self.data_fetcher = None
        
        logger.info("TradingController initialized")
    
    def initialize_trading_system(self, client_id: str = None, access_token: str = None) -> Dict[str, Any]:
        """Initialize the complete trading system"""
        try:
            # Use provided credentials or environment
            client_id = client_id or Config.DHAN_CLIENT_ID
            access_token = access_token or Config.DHAN_ACCESS_TOKEN
            
            if not client_id or not access_token:
                return {
                    'success': False,
                    'message': 'Dhan API credentials required',
                    'error': 'Missing credentials'
                }
            
            # Initialize components
            self.data_fetcher = DhanLiveDataFetcher(client_id, access_token)
            self.ai_ensemble = TradingSignalEnsemble(capital=self.config['capital_management']['initial_capital'])
            self.risk_manager = LiveRiskManager(
                initial_capital=self.config['capital_management']['initial_capital'],
                max_daily_loss=self.config['capital_management']['max_daily_loss'],
                max_portfolio_loss=self.config['capital_management']['max_portfolio_loss']
            )
            
            self.options_trader = IndianOptionsTrader(
                client_id=client_id,
                access_token=access_token,
                initial_capital=self.config['capital_management']['initial_capital']
            )
            
            logger.info("Trading system initialized successfully")
            return {
                'success': True,
                'message': 'Trading system initialized',
                'components': {
                    'data_fetcher': True,
                    'ai_ensemble': True,
                    'risk_manager': True,
                    'options_trader': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            return {
                'success': False,
                'message': 'Failed to initialize trading system',
                'error': str(e)
            }
    
    def start_automated_trading(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start automated trading with given configuration"""
        try:
            if self.trading_active:
                return {
                    'success': False,
                    'message': 'Trading is already active'
                }
            
            # Initialize system if not already done
            if not self.options_trader:
                init_result = self.initialize_trading_system()
                if not init_result['success']:
                    return init_result
            
            # Update configuration if provided
            if config:
                self.trading_service.update_trading_configuration(config)
                self.config = self.trading_service.get_trading_configuration()
            
            # Start trading
            self.trading_active = True
            
            # Start the options trader (in paper trading mode by default)
            if self.options_trader:
                self.options_trader.paper_trading = config.get('paper_trading', True)
            
            logger.info("Automated trading started")
            return {
                'success': True,
                'message': 'Automated trading started',
                'config': self.config,
                'paper_trading': config.get('paper_trading', True)
            }
            
        except Exception as e:
            logger.error(f"Error starting automated trading: {e}")
            return {
                'success': False,
                'message': 'Failed to start automated trading',
                'error': str(e)
            }
    
    def stop_automated_trading(self) -> Dict[str, Any]:
        """Stop automated trading"""
        try:
            if not self.trading_active:
                return {
                    'success': False,
                    'message': 'Trading is not currently active'
                }
            
            self.trading_active = False
            
            # Close all positions if options trader is active
            if self.options_trader:
                self.options_trader.close_all_positions()
            
            logger.info("Automated trading stopped")
            return {
                'success': True,
                'message': 'Automated trading stopped',
                'final_summary': self.get_trading_summary()
            }
            
        except Exception as e:
            logger.error(f"Error stopping automated trading: {e}")
            return {
                'success': False,
                'message': 'Failed to stop automated trading',
                'error': str(e)
            }
    
    def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trade"""
        try:
            # Validate trade data
            required_fields = ['symbol', 'action', 'quantity', 'price']
            for field in required_fields:
                if field not in trade_data:
                    return {
                        'success': False,
                        'message': f'Missing required field: {field}'
                    }
            
            # Check if system is initialized
            if not self.risk_manager:
                init_result = self.initialize_trading_system()
                if not init_result['success']:
                    return init_result
            
            # Risk validation
            can_trade, reason = self.risk_manager.can_place_trade()
            if not can_trade:
                return {
                    'success': False,
                    'message': f'Trade blocked by risk management: {reason}'
                }
            
            # Execute through trading service
            result = self.trading_service.execute_trade(trade_data)
            
            # Update risk manager with result
            if result.get('success') and 'pnl' in result:
                self.risk_manager.update_trade_result(result['pnl'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'message': 'Failed to execute trade',
                'error': str(e)
            }
    
    def get_current_signal(self) -> Dict[str, Any]:
        """Get current trading signal"""
        try:
            if not self.ai_ensemble or not self.data_fetcher:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'System not initialized'
                }
            
            # Get market data
            market_data = self.data_fetcher.get_comprehensive_market_data()
            
            if not market_data:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No market data available'
                }
            
            # Get current positions
            positions = self.trading_service.get_current_positions()
            
            # Generate signal
            signal_result = self.ai_ensemble.generate_ensemble_signal(
                market_data=market_data,
                portfolio_value=self.risk_manager.current_capital if self.risk_manager else 10000,
                current_positions=positions
            )
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error getting current signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            status = {
                'timestamp': datetime.now(),
                'trading_active': self.trading_active,
                'system_initialized': bool(self.options_trader)
            }
            
            # Get basic portfolio info
            if self.risk_manager:
                status.update({
                    'current_capital': self.risk_manager.current_capital,
                    'initial_capital': self.risk_manager.initial_capital,
                    'daily_pnl': self.risk_manager.daily_pnl,
                    'trades_today': self.risk_manager.trades_today,
                    'max_trades_per_day': self.risk_manager.max_trades_per_day,
                    'daily_target_achieved': self.risk_manager.target_achieved
                })
            
            # Get positions
            status['positions'] = self.trading_service.get_current_positions()
            
            # Get risk metrics
            if self.risk_manager:
                status['risk_metrics'] = self.risk_manager.get_risk_dashboard()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {
                'timestamp': datetime.now(),
                'trading_active': False,
                'error': str(e)
            }
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading summary"""
        try:
            summary = {
                'timestamp': datetime.now(),
                'session_active': self.trading_active
            }
            
            # Portfolio performance
            if self.risk_manager:
                total_return = (self.risk_manager.current_capital - self.risk_manager.initial_capital) / self.risk_manager.initial_capital
                summary.update({
                    'portfolio': {
                        'initial_capital': self.risk_manager.initial_capital,
                        'current_capital': self.risk_manager.current_capital,
                        'total_return': total_return,
                        'total_return_percent': total_return * 100,
                        'daily_pnl': self.risk_manager.daily_pnl,
                        'trades_completed': self.risk_manager.trades_today
                    }
                })
            
            # Trading statistics
            trade_stats = self.trading_service.get_trading_statistics()
            summary['trading_stats'] = trade_stats
            
            # Recent trades
            summary['recent_trades'] = self.trading_service.get_recent_trades(limit=10)
            
            # AI model performance
            if self.ai_ensemble:
                summary['ai_performance'] = self.ai_ensemble.get_signal_summary()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting trading summary: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def update_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading configuration"""
        try:
            # Update through service
            result = self.trading_service.update_trading_configuration(new_config)
            
            if result.get('success'):
                self.config = self.trading_service.get_trading_configuration()
                
                # Update risk manager if initialized
                if self.risk_manager and 'capital_management' in new_config:
                    capital_config = new_config['capital_management']
                    if 'max_daily_loss' in capital_config:
                        self.risk_manager.max_daily_loss = capital_config['max_daily_loss']
                    if 'max_trades_per_day' in capital_config:
                        self.risk_manager.max_trades_per_day = capital_config['max_trades_per_day']
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return {
                'success': False,
                'message': 'Failed to update configuration',
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Check system health and component status"""
        try:
            health = {
                'timestamp': datetime.now(),
                'overall_status': 'HEALTHY',
                'components': {}
            }
            
            # Check each component
            components_status = []
            
            # Data fetcher
            if self.data_fetcher:
                try:
                    test_data = self.data_fetcher.get_comprehensive_market_data()
                    data_status = 'HEALTHY' if test_data else 'WARNING'
                except Exception as e:
                    data_status = 'ERROR'
                health['components']['data_fetcher'] = data_status
                components_status.append(data_status)
            else:
                health['components']['data_fetcher'] = 'NOT_INITIALIZED'
                components_status.append('NOT_INITIALIZED')
            
            # AI ensemble
            if self.ai_ensemble:
                health['components']['ai_ensemble'] = 'HEALTHY'
                components_status.append('HEALTHY')
            else:
                health['components']['ai_ensemble'] = 'NOT_INITIALIZED'
                components_status.append('NOT_INITIALIZED')
            
            # Risk manager
            if self.risk_manager:
                health['components']['risk_manager'] = 'HEALTHY'
                components_status.append('HEALTHY')
            else:
                health['components']['risk_manager'] = 'NOT_INITIALIZED'
                components_status.append('NOT_INITIALIZED')
            
            # Options trader
            if self.options_trader:
                health['components']['options_trader'] = 'HEALTHY'
                components_status.append('HEALTHY')
            else:
                health['components']['options_trader'] = 'NOT_INITIALIZED'
                components_status.append('NOT_INITIALIZED')
            
            # Determine overall status
            if 'ERROR' in components_status:
                health['overall_status'] = 'ERROR'
            elif 'WARNING' in components_status:
                health['overall_status'] = 'WARNING'
            elif 'NOT_INITIALIZED' in components_status:
                health['overall_status'] = 'PARTIAL'
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'timestamp': datetime.now(),
                'overall_status': 'ERROR',
                'error': str(e)
            }
    
    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all trading operations"""
        try:
            logger.warning("EMERGENCY STOP INITIATED")
            
            # Stop automated trading
            self.trading_active = False
            
            # Close all positions
            if self.options_trader:
                self.options_trader.close_all_positions()
            
            # Activate risk manager emergency stop
            if self.risk_manager:
                self.risk_manager.emergency_stop()
            
            return {
                'success': True,
                'message': 'Emergency stop completed',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return {
                'success': False,
                'message': 'Emergency stop failed',
                'error': str(e)
            }


if __name__ == '__main__':
    # Test the controller
    controller = TradingController()
    print("Trading Controller initialized")
    
    # Test system health
    health = controller.get_system_health()
    print(f"System Health: {health['overall_status']}")
    
    # Test signal generation (will fail without credentials)
    signal = controller.get_current_signal()
    print(f"Current Signal: {signal.get('signal', 'N/A')}")