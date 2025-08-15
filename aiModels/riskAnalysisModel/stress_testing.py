"""
Portfolio Stress Testing
=======================

Stress test portfolios under various market scenarios
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PortfolioStressTester:
    """
    Stress test portfolio under various market scenarios
    """
    
    def __init__(self):
        self.scenarios = {}
        self.stress_results = {}
    
    def define_market_scenarios(self):
        """
        Define various market stress scenarios for Indian markets
        """
        self.scenarios = {
            'nifty_crash_10': {
                'name': 'Nifty 10% Crash',
                'nifty_change': -0.10,
                'volatility_spike': 2.0,
                'probability': 0.05,
                'description': 'Sudden 10% drop in Nifty with volatility spike'
            },
            'nifty_crash_20': {
                'name': 'Nifty 20% Crash',
                'nifty_change': -0.20,
                'volatility_spike': 3.0,
                'probability': 0.01,
                'description': 'Major market crash with 20% Nifty drop'
            },
            'volatility_spike': {
                'name': 'Volatility Spike',
                'nifty_change': -0.05,
                'volatility_spike': 2.5,
                'probability': 0.10,
                'description': 'High volatility with moderate market decline'
            },
            'currency_crisis': {
                'name': 'Currency Crisis',
                'nifty_change': -0.15,
                'usdinr_change': 0.10,
                'volatility_spike': 2.0,
                'probability': 0.02,
                'description': 'INR weakening with market decline'
            },
            'banking_crisis': {
                'name': 'Banking Sector Crisis',
                'nifty_change': -0.12,
                'bank_nifty_change': -0.25,
                'volatility_spike': 2.2,
                'probability': 0.03,
                'description': 'Banking sector specific crisis'
            },
            'global_recession': {
                'name': 'Global Recession',
                'nifty_change': -0.30,
                'volatility_spike': 3.5,
                'probability': 0.005,
                'description': 'Global economic recession impact'
            },
            'oil_shock': {
                'name': 'Oil Price Shock',
                'nifty_change': -0.08,
                'volatility_spike': 1.8,
                'probability': 0.08,
                'description': 'Sudden oil price spike affecting markets'
            },
            'tech_bubble_burst': {
                'name': 'Tech Bubble Burst',
                'nifty_change': -0.15,
                'it_sector_change': -0.35,
                'volatility_spike': 2.5,
                'probability': 0.02,
                'description': 'Technology sector bubble burst'
            }
        }
    
    def stress_test_portfolio(self, portfolio_positions, current_prices, correlations=None):
        """
        Stress test portfolio under defined scenarios
        
        Parameters:
        portfolio_positions: dict with {symbol: quantity}
        current_prices: dict with {symbol: price}
        correlations: correlation matrix (optional)
        """
        if not self.scenarios:
            self.define_market_scenarios()
        
        # Calculate current portfolio value
        current_portfolio_value = sum(
            portfolio_positions[symbol] * current_prices[symbol] 
            for symbol in portfolio_positions
        )
        
        stress_results = {}
        
        for scenario_name, scenario in self.scenarios.items():
            scenario_pnl = self._calculate_scenario_pnl(
                scenario, portfolio_positions, current_prices, correlations
            )
            
            scenario_return = scenario_pnl / current_portfolio_value
            
            stress_results[scenario_name] = {
                'scenario': scenario,
                'portfolio_pnl': scenario_pnl,
                'portfolio_return': scenario_return,
                'portfolio_value_after': current_portfolio_value + scenario_pnl,
                'probability': scenario['probability'],
                'expected_loss': scenario_pnl * scenario['probability']
            }
        
        self.stress_results = stress_results
        return stress_results
    
    def _calculate_scenario_pnl(self, scenario, positions, prices, correlations):
        """
        Calculate P&L under a specific scenario
        """
        total_pnl = 0
        
        for symbol, quantity in positions.items():
            current_price = prices[symbol]
            
            # Determine price change based on symbol type and scenario
            price_change = self._get_price_change_for_symbol(symbol, scenario, correlations)
            
            new_price = current_price * (1 + price_change)
            position_pnl = quantity * (new_price - current_price)
            
            total_pnl += position_pnl
        
        return total_pnl
    
    def _get_price_change_for_symbol(self, symbol, scenario, correlations):
        """
        Get expected price change for a symbol under scenario
        """
        # Default to Nifty change
        base_change = scenario.get('nifty_change', 0)
        
        # Adjust based on symbol characteristics
        if 'BANK' in symbol.upper() or 'HDFC' in symbol.upper() or 'ICICI' in symbol.upper():
            # Banking stocks
            if 'bank_nifty_change' in scenario:
                base_change = scenario['bank_nifty_change']
            else:
                base_change *= 1.2  # Banks typically more volatile
        
        elif 'TCS' in symbol.upper() or 'INFY' in symbol.upper() or 'TECH' in symbol.upper():
            # IT stocks
            if 'it_sector_change' in scenario:
                base_change = scenario['it_sector_change']
            else:
                base_change *= 0.8  # IT stocks less correlated with domestic factors
        
        elif 'RELIANCE' in symbol.upper() or 'ONGC' in symbol.upper():
            # Oil/Energy stocks
            if scenario['name'] == 'Oil Price Shock':
                base_change *= 1.5  # More impact on energy stocks
        
        # Add random component based on volatility spike
        volatility_multiplier = scenario.get('volatility_spike', 1.0)
        random_component = np.random.normal(0, 0.02 * volatility_multiplier)
        
        return base_change + random_component
    
    def monte_carlo_stress_test(self, portfolio_positions, current_prices, 
                              num_simulations=1000, time_horizon_days=30):
        """
        Monte Carlo stress testing
        """
        portfolio_value = sum(
            portfolio_positions[symbol] * current_prices[symbol] 
            for symbol in portfolio_positions
        )
        
        # Historical volatility assumptions (daily)
        daily_volatilities = {symbol: 0.02 for symbol in portfolio_positions}  # 2% daily vol
        
        simulation_results = []
        
        for _ in range(num_simulations):
            simulated_returns = {}
            
            for symbol in portfolio_positions:
                # Generate correlated random returns
                daily_returns = np.random.normal(
                    0, daily_volatilities[symbol], time_horizon_days
                )
                cumulative_return = np.prod(1 + daily_returns) - 1
                simulated_returns[symbol] = cumulative_return
            
            # Calculate portfolio return
            portfolio_return = sum(
                (portfolio_positions[symbol] * current_prices[symbol] / portfolio_value) * simulated_returns[symbol]
                for symbol in portfolio_positions
            )
            
            simulation_results.append(portfolio_return)
        
        # Calculate stress metrics
        simulation_results = np.array(simulation_results)
        
        monte_carlo_metrics = {
            'var_95': np.percentile(simulation_results, 5),
            'var_99': np.percentile(simulation_results, 1),
            'cvar_95': np.mean(simulation_results[simulation_results <= np.percentile(simulation_results, 5)]),
            'cvar_99': np.mean(simulation_results[simulation_results <= np.percentile(simulation_results, 1)]),
            'worst_case': np.min(simulation_results),
            'best_case': np.max(simulation_results),
            'mean_return': np.mean(simulation_results),
            'volatility': np.std(simulation_results),
            'probability_loss': np.mean(simulation_results < 0),
            'probability_large_loss': np.mean(simulation_results < -0.10)
        }
        
        return monte_carlo_metrics, simulation_results
    
    def calculate_tail_risk_metrics(self):
        """
        Calculate tail risk metrics from stress test results
        """
        if not self.stress_results:
            return "No stress test results available"
        
        returns = [result['portfolio_return'] for result in self.stress_results.values()]
        probabilities = [result['probability'] for result in self.stress_results.values()]
        
        # Sort by returns (worst first)
        sorted_scenarios = sorted(
            self.stress_results.items(), 
            key=lambda x: x[1]['portfolio_return']
        )
        
        # Calculate expected shortfall
        worst_10_percent = returns[:max(1, len(returns) // 10)]
        expected_shortfall = np.mean(worst_10_percent) if worst_10_percent else 0
        
        # Probability-weighted expected loss
        expected_loss = sum(
            ret * prob for ret, prob in zip(returns, probabilities)
        )
        
        tail_metrics = {
            'worst_scenario': sorted_scenarios[0],
            'expected_shortfall_10pct': expected_shortfall,
            'probability_weighted_expected_loss': expected_loss,
            'number_negative_scenarios': sum(1 for ret in returns if ret < 0),
            'worst_case_loss': min(returns),
            'scenarios_exceeding_10pct_loss': sum(1 for ret in returns if ret < -0.10)
        }
        
        return tail_metrics
    
    def generate_stress_report(self, portfolio_positions=None, current_prices=None):
        """
        Generate comprehensive stress testing report
        """
        if not self.stress_results:
            return "No stress test results available. Run stress_test_portfolio first."
        
        # Calculate portfolio metrics
        if portfolio_positions and current_prices:
            portfolio_value = sum(
                portfolio_positions[symbol] * current_prices[symbol] 
                for symbol in portfolio_positions
            )
        else:
            portfolio_value = 1000000  # Default for demonstration
        
        tail_metrics = self.calculate_tail_risk_metrics()
        
        report = {
            'stress_test_summary': {
                'total_scenarios_tested': len(self.stress_results),
                'portfolio_value': portfolio_value,
                'worst_case_scenario': tail_metrics['worst_scenario'][0],
                'worst_case_loss': tail_metrics['worst_case_loss'],
                'expected_tail_loss': tail_metrics['expected_shortfall_10pct']
            },
            'scenario_results': self.stress_results,
            'tail_risk_metrics': tail_metrics,
            'risk_recommendations': self._generate_risk_recommendations()
        }
        
        return report
    
    def _generate_risk_recommendations(self):
        """
        Generate risk management recommendations based on stress test results
        """
        if not self.stress_results:
            return []
        
        recommendations = []
        
        # Check for severe scenarios
        severe_losses = [
            (name, result) for name, result in self.stress_results.items() 
            if result['portfolio_return'] < -0.15
        ]
        
        if severe_losses:
            recommendations.append(
                f"Portfolio vulnerable to {len(severe_losses)} scenarios with >15% losses. Consider diversification."
            )
        
        # Check concentration risk
        banking_exposure = any('banking' in scenario.lower() or 'bank' in scenario.lower() 
                             for scenario in self.stress_results.keys())
        if banking_exposure:
            banking_loss = self.stress_results.get('banking_crisis', {}).get('portfolio_return', 0)
            if banking_loss < -0.10:
                recommendations.append(
                    "High exposure to banking sector risk. Consider reducing banking positions."
                )
        
        # Check tail risk
        tail_metrics = self.calculate_tail_risk_metrics()
        if tail_metrics['worst_case_loss'] < -0.25:
            recommendations.append(
                "Extreme tail risk detected. Consider implementing tail risk hedging strategies."
            )
        
        # Check for positive expected loss scenarios
        high_prob_losses = [
            result for result in self.stress_results.values() 
            if result['probability'] > 0.05 and result['portfolio_return'] < -0.05
        ]
        
        if high_prob_losses:
            recommendations.append(
                "High probability scenarios show significant losses. Review position sizing and stop losses."
            )
        
        if not recommendations:
            recommendations.append("Portfolio shows reasonable resilience to tested stress scenarios.")
        
        return recommendations
    
    def backtest_stress_scenarios(self, historical_data, lookback_period=252):
        """
        Backtest how stress scenarios would have performed historically
        """
        # This would use historical data to see how often scenarios actually occurred
        # and validate the stress test assumptions
        
        backtest_results = {
            'historical_validation': 'This would validate scenario probabilities against historical data',
            'scenario_frequency': 'How often each scenario type occurred historically',
            'actual_vs_predicted': 'Comparison of predicted vs actual portfolio performance during historical stress events'
        }
        
        return backtest_results