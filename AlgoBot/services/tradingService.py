#!/usr/bin/env python3
"""
Trading Service
==============

Core business logic for trading operations and data management
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np

from config.config import trading_config, save_trading_config
from models.models import Trade, TradingSession, Portfolio

logger = logging.getLogger(__name__)

class TradingService:
    """
    Service class handling core trading business logic
    """
    
    def __init__(self):
        self.config = trading_config
        self.active_session = None
        self.trade_history = []
        self.current_positions = []
        logger.info("TradingService initialized")
    
    def get_trading_configuration(self) -> Dict[str, Any]:
        """Get current trading configuration"""
        return self.config.copy()
    
    def update_trading_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading configuration"""
        try:
            # Validate configuration
            validation_result = self._validate_config(new_config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'message': 'Configuration validation failed',
                    'errors': validation_result['errors']
                }
            
            # Update configuration
            self._deep_update(self.config, new_config)
            
            # Save to file
            success = save_trading_config(self.config)
            
            if success:
                logger.info("Trading configuration updated successfully")
                return {
                    'success': True,
                    'message': 'Configuration updated successfully',
                    'config': self.config
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save configuration'
                }
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return {
                'success': False,
                'message': 'Failed to update configuration',
                'error': str(e)
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        errors = []
        
        # Validate capital management
        if 'capital_management' in config:
            cm = config['capital_management']
            if 'initial_capital' in cm and cm['initial_capital'] <= 0:
                errors.append("Initial capital must be positive")
            if 'max_daily_loss' in cm and (cm['max_daily_loss'] < 0 or cm['max_daily_loss'] > 1):
                errors.append("Max daily loss must be between 0 and 1")
            if 'daily_profit_target' in cm and cm['daily_profit_target'] <= 0:
                errors.append("Daily profit target must be positive")
        
        # Validate risk management
        if 'risk_management' in config:
            rm = config['risk_management']
            if 'max_trades_per_day' in rm and rm['max_trades_per_day'] <= 0:
                errors.append("Max trades per day must be positive")
            if 'stop_loss_percentage' in rm and (rm['stop_loss_percentage'] < 0 or rm['stop_loss_percentage'] > 1):
                errors.append("Stop loss percentage must be between 0 and 1")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade"""
        try:
            # Create trade record
            trade = Trade(
                symbol=trade_data['symbol'],
                action=trade_data['action'],
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                timestamp=datetime.now(),
                strategy=trade_data.get('strategy', 'manual'),
                confidence=trade_data.get('confidence', 0.5)
            )
            
            # Calculate trade value and fees
            trade_value = trade.quantity * trade.price
            fees = self._calculate_trading_fees(trade_value, trade_data.get('option_type'))
            
            trade.fees = fees
            trade.net_value = trade_value - fees if trade.action == 'BUY' else trade_value + fees
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Update positions
            self._update_positions(trade)
            
            logger.info(f"Trade executed: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")
            
            return {
                'success': True,
                'message': 'Trade executed successfully',
                'trade_id': trade.trade_id,
                'trade_details': trade.to_dict(),
                'fees': fees,
                'net_value': trade.net_value
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'message': 'Failed to execute trade',
                'error': str(e)
            }
    
    def _calculate_trading_fees(self, trade_value: float, option_type: str = None) -> float:
        """Calculate trading fees including Indian taxes"""
        taxes = self.config.get('taxes', {})
        
        # STT (Securities Transaction Tax)
        if option_type:  # Options
            stt = trade_value * taxes.get('stt_rate', 0.00017)  # 0.017% for options
        else:  # Equity
            stt = trade_value * 0.00025  # 0.025% for equity
        
        # Exchange charges
        exchange_charges = trade_value * taxes.get('exchange_charges_rate', 0.000019)
        
        # Brokerage
        brokerage = taxes.get('brokerage_per_trade', 20)
        
        # GST on brokerage
        gst = brokerage * taxes.get('gst_rate', 0.18)
        
        total_fees = stt + exchange_charges + brokerage + gst
        return total_fees
    
    def _update_positions(self, trade: 'Trade'):
        """Update current positions based on trade"""
        # Find existing position
        existing_position = None
        for pos in self.current_positions:
            if pos['symbol'] == trade.symbol:
                existing_position = pos
                break
        
        if existing_position:
            # Update existing position
            if trade.action == 'BUY':
                new_quantity = existing_position['quantity'] + trade.quantity
                if new_quantity == 0:
                    self.current_positions.remove(existing_position)
                else:
                    existing_position['quantity'] = new_quantity
                    # Update average price
                    total_value = (existing_position['quantity'] * existing_position['avg_price'] + 
                                 trade.quantity * trade.price)
                    existing_position['avg_price'] = total_value / new_quantity
            else:  # SELL
                existing_position['quantity'] -= trade.quantity
                if existing_position['quantity'] <= 0:
                    self.current_positions.remove(existing_position)
        else:
            # Create new position
            if trade.action == 'BUY':
                self.current_positions.append({
                    'symbol': trade.symbol,
                    'quantity': trade.quantity,
                    'avg_price': trade.price,
                    'current_price': trade.price,
                    'pnl': 0,
                    'pnl_percent': 0,
                    'entry_time': trade.timestamp
                })
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions"""
        return self.current_positions.copy()
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades"""
        recent_trades = sorted(self.trade_history, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [trade.to_dict() for trade in recent_trades]
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl_per_trade': 0,
                    'total_fees': 0,
                    'net_pnl': 0
                }
            
            # Calculate statistics
            total_trades = len(self.trade_history)
            total_fees = sum(trade.fees for trade in self.trade_history)
            
            # Calculate P&L from closed positions
            closed_pnl = self._calculate_closed_positions_pnl()
            
            # Calculate open positions P&L
            open_pnl = sum(pos['pnl'] for pos in self.current_positions)
            
            total_pnl = closed_pnl + open_pnl
            net_pnl = total_pnl - total_fees
            
            # Win/loss statistics for closed positions
            winning_trades = sum(1 for trade in self.trade_history if self._get_trade_pnl(trade) > 0)
            losing_trades = sum(1 for trade in self.trade_history if self._get_trade_pnl(trade) < 0)
            win_rate = (winning_trades / max(1, total_trades)) * 100
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / max(1, total_trades),
                'total_fees': total_fees,
                'net_pnl': net_pnl,
                'open_positions': len(self.current_positions),
                'realized_pnl': closed_pnl,
                'unrealized_pnl': open_pnl
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_closed_positions_pnl(self) -> float:
        """Calculate P&L from closed positions"""
        # This is simplified - in a real system, you'd track position closing
        return sum(self._get_trade_pnl(trade) for trade in self.trade_history)
    
    def _get_trade_pnl(self, trade: 'Trade') -> float:
        """Get P&L for a specific trade (simplified)"""
        # This is a simplified calculation
        # In reality, you'd need to match buy/sell pairs
        return 0  # Placeholder
    
    def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        try:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value()
            initial_capital = self.config['capital_management']['initial_capital']
            
            # Calculate returns
            total_return = (portfolio_value - initial_capital) / initial_capital
            
            # Daily performance (simplified)
            today = datetime.now().date()
            today_trades = [t for t in self.trade_history if t.timestamp.date() == today]
            daily_pnl = sum(self._get_trade_pnl(trade) for trade in today_trades)
            daily_return = daily_pnl / initial_capital
            
            return {
                'portfolio_value': portfolio_value,
                'initial_capital': initial_capital,
                'total_return': total_return,
                'total_return_percent': total_return * 100,
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'daily_return_percent': daily_return * 100,
                'positions_count': len(self.current_positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        # Cash + position values
        cash = self.config['capital_management']['initial_capital']
        
        # Subtract used capital for current positions
        for pos in self.current_positions:
            cash -= pos['quantity'] * pos['avg_price']
        
        # Add current position values
        positions_value = sum(pos['quantity'] * pos['current_price'] for pos in self.current_positions)
        
        return cash + positions_value
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics for the portfolio"""
        try:
            if not self.trade_history:
                return {
                    'var_1d': 0,
                    'var_5d': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'volatility': 0
                }
            
            # Calculate daily returns
            daily_returns = self._calculate_daily_returns()
            
            if len(daily_returns) < 2:
                return {
                    'var_1d': 0,
                    'var_5d': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'volatility': np.std(daily_returns) if daily_returns else 0
                }
            
            # Value at Risk (95% confidence)
            var_1d = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
            var_5d = var_1d * np.sqrt(5)
            
            # Maximum Drawdown
            cumulative_returns = np.cumsum(daily_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Sharpe Ratio (simplified, assuming risk-free rate = 0)
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            return {
                'var_1d': var_1d,
                'var_5d': var_5d,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility': std_return,
                'avg_daily_return': avg_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns from trade history"""
        # Group trades by date and calculate daily P&L
        daily_pnl = {}
        
        for trade in self.trade_history:
            date = trade.timestamp.date()
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += self._get_trade_pnl(trade)
        
        # Convert to returns
        initial_capital = self.config['capital_management']['initial_capital']
        daily_returns = [pnl / initial_capital for pnl in daily_pnl.values()]
        
        return daily_returns
    
    def get_live_market_data(self) -> Dict[str, Any]:
        """Get live market data (placeholder)"""
        # This would integrate with actual market data feed
        return {
            'nifty': {
                'ltp': 25000,
                'change': 150,
                'change_percent': 0.6
            },
            'india_vix': {
                'vix_value': 18.5
            },
            'timestamp': datetime.now()
        }
    
    def get_current_trading_signal(self) -> Dict[str, Any]:
        """Get current trading signal (placeholder)"""
        # This would integrate with AI ensemble
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'option_type': None,
            'reasoning': 'Market analysis pending',
            'timestamp': datetime.now()
        }
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data"""
        return {
            'nifty_50': {
                'current': 25000,
                'change': 150,
                'change_percent': 0.6,
                'high': 25100,
                'low': 24850
            },
            'india_vix': {
                'current': 18.5,
                'change': -0.5,
                'change_percent': -2.6
            },
            'market_status': 'OPEN' if self._is_market_hours() else 'CLOSED',
            'last_updated': datetime.now()
        }
    
    def _is_market_hours(self) -> bool:
        """Check if currently in market hours"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() > 4:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_trading_system_status(self) -> Dict[str, Any]:
        """Get trading system status"""
        return {
            'status': 'ACTIVE',
            'market_hours': self._is_market_hours(),
            'total_positions': len(self.current_positions),
            'total_trades_today': len([t for t in self.trade_history 
                                     if t.timestamp.date() == datetime.now().date()]),
            'system_uptime': '24h 15m',  # Placeholder
            'last_signal_update': datetime.now(),
            'components': {
                'data_feed': 'CONNECTED',
                'ai_models': 'ACTIVE',
                'risk_manager': 'ACTIVE',
                'order_management': 'ACTIVE'
            }
        }
    
    def get_trades_history(self, page: int = 1, per_page: int = 50) -> Dict[str, Any]:
        """Get paginated trades history"""
        try:
            total_trades = len(self.trade_history)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            
            # Sort by timestamp (most recent first)
            sorted_trades = sorted(self.trade_history, key=lambda x: x.timestamp, reverse=True)
            page_trades = sorted_trades[start_idx:end_idx]
            
            return {
                'trades': [trade.to_dict() for trade in page_trades],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total_trades,
                    'pages': (total_trades + per_page - 1) // per_page
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trades history: {e}")
            return {'error': str(e)}
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for reporting"""
        try:
            stats = self.get_trading_statistics()
            performance = self.get_portfolio_performance()
            risk_metrics = self.get_risk_metrics()
            
            return {
                'statistics': stats,
                'performance': performance,
                'risk_metrics': risk_metrics,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {'error': str(e)}


if __name__ == '__main__':
    # Test the service
    service = TradingService()
    print("TradingService initialized")
    
    # Test configuration
    config = service.get_trading_configuration()
    print(f"Initial capital: Rs. {config['capital_management']['initial_capital']:,}")
    
    # Test statistics
    stats = service.get_trading_statistics()
    print(f"Total trades: {stats['total_trades']}")
    
    # Test market overview
    overview = service.get_market_overview()
    print(f"Market status: {overview['market_status']}")