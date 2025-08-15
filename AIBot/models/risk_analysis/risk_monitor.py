"""
Real-time Risk Monitoring System
===============================

Monitor portfolio risk in real-time and send alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring for trading portfolios
    """
    
    def __init__(self, max_portfolio_loss=0.10, max_daily_loss=0.05, 
                 max_drawdown=0.15, var_limit=0.08):
        self.max_portfolio_loss = max_portfolio_loss
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        
        self.portfolio_start_value = None
        self.daily_start_value = None
        self.portfolio_peak = None
        self.alerts = []
        self.risk_status = 'NORMAL'
        
        # Historical data for calculations
        self.portfolio_history = []
        self.daily_returns = []
        
    def initialize_monitoring(self, initial_portfolio_value):
        """
        Initialize the risk monitoring system
        """
        self.portfolio_start_value = initial_portfolio_value
        self.daily_start_value = initial_portfolio_value
        self.portfolio_peak = initial_portfolio_value
        
        print(f"Risk monitoring initialized with portfolio value: Rs.{initial_portfolio_value:,.2f}")
    
    def update_portfolio_value(self, current_value, timestamp=None):
        """
        Update portfolio value and check risk limits
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store historical data
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': current_value
        })
        
        # Update peak
        if current_value > self.portfolio_peak:
            self.portfolio_peak = current_value
        
        # Check all risk limits
        risk_checks = self.check_all_risk_limits(current_value, timestamp)
        
        # Update daily start value if new day
        if self._is_new_day(timestamp):
            self.daily_start_value = current_value
            self.daily_returns.append(self._calculate_daily_return())
        
        return risk_checks
    
    def check_all_risk_limits(self, current_value, timestamp):
        """
        Check all risk limits and generate alerts
        """
        checks = {}
        
        # Portfolio loss check
        portfolio_loss = (self.portfolio_start_value - current_value) / self.portfolio_start_value
        checks['portfolio_loss'] = {
            'current_loss': portfolio_loss,
            'limit': self.max_portfolio_loss,
            'status': 'BREACH' if portfolio_loss > self.max_portfolio_loss else 'OK'
        }
        
        # Daily loss check
        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        checks['daily_loss'] = {
            'current_loss': daily_loss,
            'limit': self.max_daily_loss,
            'status': 'BREACH' if daily_loss > self.max_daily_loss else 'OK'
        }
        
        # Drawdown check
        drawdown = (self.portfolio_peak - current_value) / self.portfolio_peak
        checks['drawdown'] = {
            'current_drawdown': drawdown,
            'limit': self.max_drawdown,
            'status': 'BREACH' if drawdown > self.max_drawdown else 'OK'
        }
        
        # VaR check (if enough historical data)
        if len(self.daily_returns) >= 30:
            var_check = self._check_var_limit()
            checks['var'] = var_check
        
        # Update overall status
        breaches = [check for check in checks.values() if check['status'] == 'BREACH']
        
        if breaches:
            self.risk_status = 'CRITICAL'
            self._generate_risk_alert(breaches, current_value, timestamp)
        elif any(check['current_loss'] > check['limit'] * 0.8 for check in [checks['portfolio_loss'], checks['daily_loss']] 
                if 'current_loss' in check):
            self.risk_status = 'WARNING'
        else:
            self.risk_status = 'NORMAL'
        
        return checks
    
    def _check_var_limit(self):
        """
        Check Value at Risk limit
        """
        if len(self.daily_returns) < 30:
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Calculate 1-day VaR at 95% confidence
        var_95 = np.percentile(self.daily_returns, 5)
        
        return {
            'current_var': abs(var_95),
            'limit': self.var_limit,
            'status': 'BREACH' if abs(var_95) > self.var_limit else 'OK'
        }
    
    def _generate_risk_alert(self, breaches, current_value, timestamp):
        """
        Generate risk alert
        """
        alert = {
            'timestamp': timestamp,
            'portfolio_value': current_value,
            'alert_type': 'RISK_BREACH',
            'breaches': breaches,
            'message': self._format_alert_message(breaches, current_value)
        }
        
        self.alerts.append(alert)
        print(f" RISK ALERT: {alert['message']}")
    
    def _format_alert_message(self, breaches, current_value):
        """
        Format alert message
        """
        messages = []
        
        for breach in breaches:
            if 'current_loss' in breach:
                messages.append(
                    f"Loss {breach['current_loss']*100:.2f}% exceeds limit {breach['limit']*100:.2f}%"
                )
            elif 'current_drawdown' in breach:
                messages.append(
                    f"Drawdown {breach['current_drawdown']*100:.2f}% exceeds limit {breach['limit']*100:.2f}%"
                )
            elif 'current_var' in breach:
                messages.append(
                    f"VaR {breach['current_var']*100:.2f}% exceeds limit {breach['limit']*100:.2f}%"
                )
        
        return f"Portfolio Rs.{current_value:,.2f} - " + "; ".join(messages)
    
    def get_risk_dashboard(self):
        """
        Get current risk dashboard
        """
        if not self.portfolio_history:
            return "No portfolio data available"
        
        current_value = self.portfolio_history[-1]['value']
        
        dashboard = {
            'timestamp': datetime.now(),
            'portfolio_value': current_value,
            'risk_status': self.risk_status,
            'portfolio_return': (current_value - self.portfolio_start_value) / self.portfolio_start_value,
            'daily_return': (current_value - self.daily_start_value) / self.daily_start_value,
            'max_drawdown': (self.portfolio_peak - current_value) / self.portfolio_peak,
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'risk_utilization': {
                'portfolio_loss': abs((current_value - self.portfolio_start_value) / self.portfolio_start_value) / self.max_portfolio_loss,
                'daily_loss': abs((current_value - self.daily_start_value) / self.daily_start_value) / self.max_daily_loss,
                'drawdown': abs((self.portfolio_peak - current_value) / self.portfolio_peak) / self.max_drawdown
            }
        }
        
        return dashboard
    
    def get_position_risk_limits(self, current_portfolio_value):
        """
        Get recommended position risk limits based on current state
        """
        # Current portfolio loss
        portfolio_loss = (self.portfolio_start_value - current_portfolio_value) / self.portfolio_start_value
        
        # Adjust position sizing based on current drawdown
        remaining_portfolio_risk = max(0, self.max_portfolio_loss - portfolio_loss)
        remaining_daily_risk = max(0, self.max_daily_loss - 
                                 (self.daily_start_value - current_portfolio_value) / self.daily_start_value)
        
        # Conservative position sizing when close to limits
        if remaining_portfolio_risk < 0.02 or remaining_daily_risk < 0.01:
            position_risk_limit = 0.005  # Very conservative
        elif remaining_portfolio_risk < 0.05 or remaining_daily_risk < 0.02:
            position_risk_limit = 0.01   # Conservative
        else:
            position_risk_limit = 0.02   # Normal
        
        return {
            'max_position_risk': position_risk_limit,
            'remaining_portfolio_risk': remaining_portfolio_risk,
            'remaining_daily_risk': remaining_daily_risk,
            'recommendation': self._get_risk_recommendation(remaining_portfolio_risk, remaining_daily_risk)
        }
    
    def _get_risk_recommendation(self, remaining_portfolio_risk, remaining_daily_risk):
        """
        Get risk management recommendation
        """
        if remaining_portfolio_risk < 0.01 or remaining_daily_risk < 0.005:
            return "STOP_TRADING - Close to risk limits"
        elif remaining_portfolio_risk < 0.03 or remaining_daily_risk < 0.01:
            return "REDUCE_EXPOSURE - Limit new positions"
        elif remaining_portfolio_risk < 0.05 or remaining_daily_risk < 0.02:
            return "CONSERVATIVE - Small position sizes only"
        else:
            return "NORMAL - Standard position sizing"
    
    def _calculate_daily_return(self):
        """
        Calculate daily return
        """
        if len(self.portfolio_history) < 2:
            return 0
        
        today_value = self.portfolio_history[-1]['value']
        yesterday_value = self.daily_start_value
        
        return (today_value - yesterday_value) / yesterday_value
    
    def _is_new_day(self, timestamp):
        """
        Check if it's a new trading day
        """
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return False
        
        last_timestamp = self.portfolio_history[-2]['timestamp']
        return timestamp.date() != last_timestamp.date()
    
    def export_risk_report(self):
        """
        Export comprehensive risk report
        """
        if not self.portfolio_history:
            return "No data available for report"
        
        current_value = self.portfolio_history[-1]['value']
        
        report = {
            'report_date': datetime.now(),
            'portfolio_summary': {
                'current_value': current_value,
                'initial_value': self.portfolio_start_value,
                'peak_value': self.portfolio_peak,
                'total_return': (current_value - self.portfolio_start_value) / self.portfolio_start_value,
                'max_drawdown': (self.portfolio_peak - current_value) / self.portfolio_peak
            },
            'risk_limits': {
                'max_portfolio_loss': self.max_portfolio_loss,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'var_limit': self.var_limit
            },
            'current_status': self.risk_status,
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'risk_metrics': self._calculate_risk_metrics()
        }
        
        return report
    
    def _calculate_risk_metrics(self):
        """
        Calculate comprehensive risk metrics
        """
        if len(self.daily_returns) < 10:
            return "Insufficient data for risk metrics"
        
        returns_array = np.array(self.daily_returns)
        
        metrics = {
            'volatility_annualized': np.std(returns_array) * np.sqrt(252),
            'var_95': np.percentile(returns_array, 5),
            'var_99': np.percentile(returns_array, 1),
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
            'max_daily_gain': np.max(returns_array),
            'max_daily_loss': np.min(returns_array),
            'positive_days': np.sum(returns_array > 0) / len(returns_array),
            'average_gain': np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0,
            'average_loss': np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
        }
        
        return metrics