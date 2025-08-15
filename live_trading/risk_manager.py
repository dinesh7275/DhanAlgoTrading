"""
Advanced Risk Management for Live Trading
========================================

Real-time risk monitoring and automatic risk controls
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from aiModels.riskAnalysisModel import (
    PortfolioRiskCalculator, PositionSizer, 
    RealTimeRiskMonitor, PortfolioStressTester
)


class LiveRiskManager:
    """
    Advanced risk management for live trading
    """
    
    def __init__(self, initial_capital, max_daily_loss=0.05, max_portfolio_loss=0.10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk limits
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_loss = max_portfolio_loss
        self.max_position_risk = 0.02
        self.max_correlation_exposure = 0.3
        self.max_single_position = 0.1
        
        # Risk monitoring components
        self.risk_calculator = PortfolioRiskCalculator()
        self.position_sizer = PositionSizer(initial_capital, max_daily_loss, max_portfolio_loss)
        self.risk_monitor = RealTimeRiskMonitor(max_portfolio_loss, max_daily_loss)
        self.stress_tester = PortfolioStressTester()
        
        # Initialize monitoring
        self.risk_monitor.initialize_monitoring(initial_capital)
        
        # Risk state
        self.risk_alerts = []
        self.circuit_breakers = {
            'daily_loss_breaker': False,
            'portfolio_loss_breaker': False,
            'volatility_breaker': False,
            'correlation_breaker': False
        }
        
        # Trading controls
        self.trading_allowed = True
        self.position_limits = {
            'max_positions': 5,
            'current_positions': 0,
            'max_sector_exposure': 0.4
        }
        
        print("🛡️ Advanced Risk Manager Initialized")
        print(f"💰 Capital: ₹{initial_capital:,}")
        print(f"⚠️  Max Daily Loss: {max_daily_loss*100:.1f}%")
        print(f"🚫 Max Portfolio Loss: {max_portfolio_loss*100:.1f}%")
    
    def update_portfolio_value(self, current_value, positions=None):
        """Update portfolio value and check risk limits"""
        self.current_capital = current_value
        
        # Update risk monitor
        risk_checks = self.risk_monitor.update_portfolio_value(current_value)
        
        # Check circuit breakers
        self._check_circuit_breakers(risk_checks)
        
        # Update position limits based on current positions
        if positions:
            self.position_limits['current_positions'] = len(positions)
            self._analyze_position_risk(positions)
        
        return risk_checks
    
    def _check_circuit_breakers(self, risk_checks):
        """Check and activate circuit breakers"""
        
        # Daily loss circuit breaker
        if risk_checks.get('daily_loss', {}).get('status') == 'BREACH':
            if not self.circuit_breakers['daily_loss_breaker']:
                self.circuit_breakers['daily_loss_breaker'] = True
                self.trading_allowed = False
                self._trigger_alert('CRITICAL', 'Daily loss limit breached - Trading halted')
        
        # Portfolio loss circuit breaker
        if risk_checks.get('portfolio_loss', {}).get('status') == 'BREACH':
            if not self.circuit_breakers['portfolio_loss_breaker']:
                self.circuit_breakers['portfolio_loss_breaker'] = True
                self.trading_allowed = False
                self._trigger_alert('CRITICAL', 'Portfolio loss limit breached - Trading halted')
        
        # Drawdown circuit breaker
        if risk_checks.get('drawdown', {}).get('status') == 'BREACH':
            if not self.circuit_breakers['volatility_breaker']:
                self.circuit_breakers['volatility_breaker'] = True
                self.trading_allowed = False
                self._trigger_alert('CRITICAL', 'Maximum drawdown breached - Trading halted')
    
    def _analyze_position_risk(self, positions):
        """Analyze risk from current positions"""
        if not positions:
            return
        
        total_exposure = 0
        sector_exposure = {}
        
        for position in positions:
            position_value = abs(position.get('quantity', 0) * position.get('current_price', 0))
            total_exposure += position_value
            
            # Track sector exposure (simplified)
            sector = self._get_position_sector(position)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        # Check concentration risk
        portfolio_value = self.current_capital
        
        for sector, exposure in sector_exposure.items():
            sector_ratio = exposure / portfolio_value
            if sector_ratio > self.max_correlation_exposure:
                self._trigger_alert('WARNING', f'High {sector} sector exposure: {sector_ratio*100:.1f}%')
        
        # Check total exposure
        exposure_ratio = total_exposure / portfolio_value
        if exposure_ratio > 0.8:
            self._trigger_alert('WARNING', f'High portfolio exposure: {exposure_ratio*100:.1f}%')
    
    def _get_position_sector(self, position):
        """Determine sector for position (simplified)"""
        symbol = position.get('symbol', '').upper()
        
        if 'BANK' in symbol or symbol in ['HDFC', 'ICICI', 'AXIS', 'SBI']:
            return 'BANKING'
        elif symbol in ['TCS', 'INFY', 'WIPRO', 'HCL']:
            return 'IT'
        elif 'NIFTY' in symbol:
            return 'INDEX'
        else:
            return 'OTHER'
    
    def validate_new_position(self, order_details, current_positions):
        """Validate if new position is allowed based on risk limits"""
        validation_result = {
            'allowed': True,
            'reasons': [],
            'adjusted_quantity': order_details.get('quantity', 0)
        }
        
        # Check if trading is allowed
        if not self.trading_allowed:
            validation_result['allowed'] = False
            validation_result['reasons'].append('Trading halted due to risk limits')
            return validation_result
        
        # Check maximum positions
        if len(current_positions) >= self.position_limits['max_positions']:
            validation_result['allowed'] = False
            validation_result['reasons'].append(f'Maximum positions limit reached: {self.position_limits["max_positions"]}')
            return validation_result
        
        # Calculate position size based on risk
        entry_price = order_details.get('price', 0)
        stop_loss = order_details.get('stop_loss', entry_price * 0.98)
        
        if entry_price <= 0:
            validation_result['allowed'] = False
            validation_result['reasons'].append('Invalid entry price')
            return validation_result
        
        # Calculate optimal position size
        optimal_size = self.position_sizer.calculate_nifty_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            method='fixed_fractional'
        )
        
        requested_quantity = order_details.get('quantity', 0)
        
        # Limit position size
        if requested_quantity > optimal_size:
            validation_result['adjusted_quantity'] = optimal_size
            validation_result['reasons'].append(f'Position size reduced from {requested_quantity} to {optimal_size} for risk management')
        
        # Check single position limit
        position_value = entry_price * validation_result['adjusted_quantity']
        position_ratio = position_value / self.current_capital
        
        if position_ratio > self.max_single_position:
            max_quantity = int(self.current_capital * self.max_single_position / entry_price)
            validation_result['adjusted_quantity'] = max_quantity
            validation_result['reasons'].append(f'Position size limited to {self.max_single_position*100:.1f}% of portfolio')
        
        # Final validation
        if validation_result['adjusted_quantity'] <= 0:
            validation_result['allowed'] = False
            validation_result['reasons'].append('Position size too small after risk adjustments')
        
        return validation_result
    
    def calculate_portfolio_risk_metrics(self, positions, market_data=None):
        """Calculate comprehensive portfolio risk metrics"""
        if not positions:
            return {'status': 'NO_POSITIONS', 'risk_score': 0}
        
        # Calculate returns for risk analysis
        portfolio_returns = []
        
        for position in positions:
            if 'pnl_percent' in position:
                portfolio_returns.append(position['pnl_percent'] / 100)
        
        if not portfolio_returns:
            return {'status': 'INSUFFICIENT_DATA', 'risk_score': 0.5}
        
        # Calculate risk metrics
        portfolio_returns = np.array(portfolio_returns)
        
        # Basic risk metrics
        risk_metrics = self.risk_calculator.calculate_comprehensive_risk_metrics(
            returns=portfolio_returns
        )
        
        # Add current market context
        if market_data:
            vix_level = market_data.get('india_vix', {}).get('vix_value', 20)
            market_volatility = market_data.get('nifty', {}).get('change_percent', 0)
            
            # Adjust risk score based on market conditions
            base_risk_score = min(abs(risk_metrics.get('var_5', {}).get('historical_var', 0.02)), 0.1)
            
            # Market condition adjustments
            if vix_level > 25:
                market_risk_multiplier = 1.3
            elif vix_level < 15:
                market_risk_multiplier = 0.8
            else:
                market_risk_multiplier = 1.0
            
            adjusted_risk_score = base_risk_score * market_risk_multiplier
            
            risk_metrics.update({
                'market_vix': vix_level,
                'market_volatility': market_volatility,
                'adjusted_risk_score': adjusted_risk_score,
                'risk_grade': self.risk_calculator.get_risk_grade(risk_metrics)
            })
        
        return risk_metrics
    
    def stress_test_portfolio(self, positions, scenarios=None):
        """Run stress tests on current portfolio"""
        if not positions:
            return {'status': 'NO_POSITIONS'}
        
        # Convert positions to format expected by stress tester
        portfolio_positions = {}
        current_prices = {}
        
        for position in positions:
            symbol = position.get('symbol', 'UNKNOWN')
            portfolio_positions[symbol] = position.get('quantity', 0)
            current_prices[symbol] = position.get('current_price', 0)
        
        if not portfolio_positions:
            return {'status': 'NO_VALID_POSITIONS'}
        
        # Run stress tests
        stress_results = self.stress_tester.stress_test_portfolio(
            portfolio_positions, current_prices
        )
        
        # Analyze results
        worst_case_loss = min([result['portfolio_return'] for result in stress_results.values()])
        
        stress_summary = {
            'worst_case_scenario': worst_case_loss,
            'scenarios_tested': len(stress_results),
            'high_risk_scenarios': sum(1 for result in stress_results.values() if result['portfolio_return'] < -0.1),
            'stress_results': stress_results
        }
        
        # Generate alerts if needed
        if worst_case_loss < -0.15:
            self._trigger_alert('WARNING', f'Stress test shows potential {worst_case_loss*100:.1f}% loss in worst case')
        
        return stress_summary
    
    def get_position_recommendations(self, signal_strength, market_volatility):
        """Get position sizing recommendations based on risk"""
        base_risk = self.max_position_risk
        
        # Adjust risk based on signal strength
        if signal_strength > 0.8:
            adjusted_risk = base_risk * 1.2
        elif signal_strength < 0.6:
            adjusted_risk = base_risk * 0.7
        else:
            adjusted_risk = base_risk
        
        # Adjust for market volatility
        if market_volatility > 0.25:  # High volatility
            adjusted_risk *= 0.7
        elif market_volatility < 0.15:  # Low volatility
            adjusted_risk *= 1.1
        
        # Final limits
        adjusted_risk = min(adjusted_risk, self.max_position_risk * 1.5)
        
        recommendations = {
            'max_position_risk': adjusted_risk,
            'recommended_positions': max(1, 5 - len(self.circuit_breakers)),
            'risk_adjustment_factor': adjusted_risk / base_risk,
            'market_condition': 'HIGH_VOL' if market_volatility > 0.25 else 'LOW_VOL' if market_volatility < 0.15 else 'NORMAL'
        }
        
        return recommendations
    
    def _trigger_alert(self, severity, message):
        """Trigger risk alert"""
        alert = {
            'timestamp': datetime.now(),
            'severity': severity,
            'message': message,
            'portfolio_value': self.current_capital
        }
        
        self.risk_alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.risk_alerts) > 50:
            self.risk_alerts = self.risk_alerts[-50:]
        
        # Print alert
        emoji = "🚨" if severity == 'CRITICAL' else "⚠️" if severity == 'WARNING' else "ℹ️"
        print(f"{emoji} {severity}: {message}")
    
    def reset_circuit_breakers(self):
        """Reset circuit breakers (use with caution)"""
        print("🔄 Resetting circuit breakers...")
        
        for breaker in self.circuit_breakers:
            self.circuit_breakers[breaker] = False
        
        self.trading_allowed = True
        self._trigger_alert('INFO', 'Circuit breakers reset - Trading resumed')
    
    def get_risk_dashboard(self):
        """Get comprehensive risk dashboard"""
        dashboard = {
            'timestamp': datetime.now(),
            'risk_status': 'NORMAL' if self.trading_allowed else 'HALTED',
            'portfolio_value': self.current_capital,
            'risk_limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_portfolio_loss': self.max_portfolio_loss,
                'max_position_risk': self.max_position_risk
            },
            'circuit_breakers': self.circuit_breakers,
            'position_limits': self.position_limits,
            'recent_alerts': self.risk_alerts[-10:] if self.risk_alerts else [],
            'risk_monitor_status': self.risk_monitor.get_risk_dashboard()
        }
        
        return dashboard
    
    def export_risk_report(self):
        """Export detailed risk report"""
        report = {
            'report_timestamp': datetime.now(),
            'portfolio_summary': {
                'current_value': self.current_capital,
                'initial_value': self.initial_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            },
            'risk_settings': {
                'max_daily_loss': self.max_daily_loss,
                'max_portfolio_loss': self.max_portfolio_loss,
                'max_position_risk': self.max_position_risk
            },
            'circuit_breaker_status': self.circuit_breakers,
            'all_alerts': self.risk_alerts,
            'position_limits': self.position_limits,
            'risk_monitor_report': self.risk_monitor.export_risk_report()
        }
        
        # Save to file
        filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📊 Risk report saved: {filename}")
        except Exception as e:
            print(f"Error saving risk report: {e}")
        
        return report
    
    def emergency_stop(self):
        """Emergency stop - halt all trading immediately"""
        print("🚨 EMERGENCY STOP ACTIVATED")
        
        self.trading_allowed = False
        
        # Activate all circuit breakers
        for breaker in self.circuit_breakers:
            self.circuit_breakers[breaker] = True
        
        self._trigger_alert('CRITICAL', 'EMERGENCY STOP - All trading halted')
        
        return True