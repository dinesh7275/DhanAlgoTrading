"""
Portfolio Risk Metrics Calculator
================================

Calculate various risk metrics for portfolio analysis
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PortfolioRiskCalculator:
    """
    Calculate portfolio risk metrics
    """
    
    def __init__(self):
        self.portfolio_data = None
        self.risk_metrics = {}
    
    def calculate_var(self, returns, confidence_level=0.05):
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        returns: Series of portfolio returns
        confidence_level: Confidence level for VaR (default 5%)
        """
        if len(returns) == 0:
            return 0
        
        # Historical VaR
        historical_var = np.percentile(returns, confidence_level * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_score = np.percentile(np.random.normal(0, 1, 10000), confidence_level * 100)
        parametric_var = mean_return + z_score * std_return
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'confidence_level': confidence_level
        }
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        """
        if len(returns) == 0:
            return 0
        
        var_threshold = np.percentile(returns, confidence_level * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return var_threshold
        
        cvar = np.mean(tail_losses)
        return cvar
    
    def calculate_maximum_drawdown(self, cumulative_returns):
        """
        Calculate maximum drawdown
        """
        if len(cumulative_returns) == 0:
            return 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Find the dates of peak and trough
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(running_max[:max_dd_idx + 1])
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'peak_index': peak_idx,
            'trough_index': max_dd_idx,
            'recovery_time': len(cumulative_returns) - max_dd_idx if max_dd_idx < len(cumulative_returns) - 1 else None
        }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.06):
        """
        Calculate Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns, target_return=0):
        """
        Calculate Sortino ratio (downside deviation)
        """
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return np.inf
        
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        return sortino_ratio
    
    def calculate_beta(self, portfolio_returns, market_returns):
        """
        Calculate portfolio beta against market
        """
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) == 0:
            return 0
        
        # Remove NaN values
        valid_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()
        
        if len(valid_data) < 2:
            return 0
        
        covariance = np.cov(valid_data['portfolio'], valid_data['market'])[0, 1]
        market_variance = np.var(valid_data['market'])
        
        if market_variance == 0:
            return 0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_tracking_error(self, portfolio_returns, benchmark_returns):
        """
        Calculate tracking error against benchmark
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        
        return tracking_error
    
    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """
        Calculate information ratio
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        if tracking_error == 0:
            return 0
        
        information_ratio = np.mean(excess_returns) * 252 / tracking_error
        return information_ratio
    
    def calculate_calmar_ratio(self, returns):
        """
        Calculate Calmar ratio (annual return / max drawdown)
        """
        if len(returns) == 0:
            return 0
        
        annual_return = np.mean(returns) * 252
        cumulative_returns = (1 + returns).cumprod()
        max_dd_info = self.calculate_maximum_drawdown(cumulative_returns)
        max_drawdown = abs(max_dd_info['max_drawdown'])
        
        if max_drawdown == 0:
            return np.inf
        
        calmar_ratio = annual_return / max_drawdown
        return calmar_ratio
    
    def calculate_comprehensive_risk_metrics(self, returns, market_returns=None, 
                                           benchmark_returns=None, risk_free_rate=0.06):
        """
        Calculate comprehensive risk metrics for a portfolio
        """
        if len(returns) == 0:
            return {}
        
        cumulative_returns = (1 + returns).cumprod()
        
        metrics = {
            # Basic statistics
            'total_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0,
            'annualized_return': np.mean(returns) * 252,
            'annualized_volatility': np.std(returns) * np.sqrt(252),
            
            # Risk metrics
            'var_5': self.calculate_var(returns, 0.05),
            'var_1': self.calculate_var(returns, 0.01),
            'cvar_5': self.calculate_cvar(returns, 0.05),
            'cvar_1': self.calculate_cvar(returns, 0.01),
            'max_drawdown': self.calculate_maximum_drawdown(cumulative_returns),
            
            # Risk-adjusted returns
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
        }
        
        # Market-related metrics
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, market_returns)
        
        # Benchmark-related metrics
        if benchmark_returns is not None:
            metrics['tracking_error'] = self.calculate_tracking_error(returns, benchmark_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
        
        self.risk_metrics = metrics
        return metrics
    
    def get_risk_grade(self, metrics=None):
        """
        Assign risk grade based on metrics
        """
        if metrics is None:
            metrics = self.risk_metrics
        
        if not metrics:
            return 'Unknown'
        
        # Risk scoring based on multiple factors
        score = 0
        
        # Sharpe ratio scoring
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 3
        elif sharpe > 1:
            score += 2
        elif sharpe > 0:
            score += 1
        
        # Maximum drawdown scoring
        max_dd = abs(metrics.get('max_drawdown', {}).get('max_drawdown', 0))
        if max_dd < 0.05:
            score += 3
        elif max_dd < 0.10:
            score += 2
        elif max_dd < 0.20:
            score += 1
        
        # VaR scoring
        var_5 = metrics.get('var_5', {}).get('historical_var', 0)
        if var_5 > -0.02:
            score += 3
        elif var_5 > -0.05:
            score += 2
        elif var_5 > -0.10:
            score += 1
        
        # Convert score to grade
        if score >= 7:
            return 'A'
        elif score >= 5:
            return 'B'
        elif score >= 3:
            return 'C'
        else:
            return 'D'