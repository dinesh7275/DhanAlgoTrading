"""
Arbitrage Opportunity Finder
===========================

Identify arbitrage opportunities in Nifty options
"""

import numpy as np
import pandas as pd
from .black_scholes import BlackScholesCalculator
import warnings
warnings.filterwarnings('ignore')


class ArbitrageOpportunityFinder:
    """
    Find arbitrage opportunities in Nifty options
    """
    
    def __init__(self, risk_free_rate=0.06, transaction_cost=0.001):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.bs_calculator = BlackScholesCalculator()
        self.opportunities = []
    
    def check_put_call_parity(self, calls_data, puts_data, spot_price, time_to_expiry):
        """
        Check for put-call parity violations
        Put-Call Parity: C - P = S - K * e^(-r*T)
        """
        arbitrage_opportunities = []
        
        # Merge calls and puts data on strike price
        merged_data = pd.merge(
            calls_data[['strike', 'market_price']].rename(columns={'market_price': 'call_price'}),
            puts_data[['strike', 'market_price']].rename(columns={'market_price': 'put_price'}),
            on='strike'
        )
        
        for _, row in merged_data.iterrows():
            strike = row['strike']
            call_price = row['call_price']
            put_price = row['put_price']
            
            # Calculate theoretical parity value
            pv_strike = strike * np.exp(-self.risk_free_rate * time_to_expiry)
            theoretical_diff = spot_price - pv_strike
            actual_diff = call_price - put_price
            
            # Check for violations (accounting for transaction costs)
            min_profit = 2 * self.transaction_cost * (call_price + put_price + spot_price + pv_strike)
            
            if abs(actual_diff - theoretical_diff) > min_profit:
                direction = "buy_call_sell_put" if actual_diff < theoretical_diff else "sell_call_buy_put"
                profit_potential = abs(actual_diff - theoretical_diff) - min_profit
                
                arbitrage_opportunities.append({
                    'type': 'put_call_parity',
                    'strike': strike,
                    'direction': direction,
                    'profit_potential': profit_potential,
                    'call_price': call_price,
                    'put_price': put_price,
                    'spot_price': spot_price,
                    'theoretical_diff': theoretical_diff,
                    'actual_diff': actual_diff
                })
        
        return arbitrage_opportunities
    
    def check_box_spread_arbitrage(self, calls_data, puts_data, time_to_expiry):
        """
        Check for box spread arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Get available strikes
        strikes = sorted(set(calls_data['strike'].unique()) & set(puts_data['strike'].unique()))
        
        for i in range(len(strikes) - 1):
            k1, k2 = strikes[i], strikes[i + 1]
            
            # Get option prices
            call_k1 = calls_data[calls_data['strike'] == k1]['market_price'].iloc[0]
            call_k2 = calls_data[calls_data['strike'] == k2]['market_price'].iloc[0]
            put_k1 = puts_data[puts_data['strike'] == k1]['market_price'].iloc[0]
            put_k2 = puts_data[puts_data['strike'] == k2]['market_price'].iloc[0]
            
            # Box spread cost
            box_cost = (call_k1 - call_k2) + (put_k2 - put_k1)
            
            # Theoretical box value
            theoretical_value = (k2 - k1) * np.exp(-self.risk_free_rate * time_to_expiry)
            
            # Transaction costs
            total_transaction_cost = self.transaction_cost * (call_k1 + call_k2 + put_k1 + put_k2)
            
            # Check for arbitrage
            if box_cost < theoretical_value - total_transaction_cost:
                profit = theoretical_value - box_cost - total_transaction_cost
                
                arbitrage_opportunities.append({
                    'type': 'box_spread',
                    'lower_strike': k1,
                    'upper_strike': k2,
                    'direction': 'buy_box',
                    'profit_potential': profit,
                    'box_cost': box_cost,
                    'theoretical_value': theoretical_value,
                    'profit_margin': profit / box_cost if box_cost > 0 else 0
                })
            
            elif box_cost > theoretical_value + total_transaction_cost:
                profit = box_cost - theoretical_value - total_transaction_cost
                
                arbitrage_opportunities.append({
                    'type': 'box_spread',
                    'lower_strike': k1,
                    'upper_strike': k2,
                    'direction': 'sell_box',
                    'profit_potential': profit,
                    'box_cost': box_cost,
                    'theoretical_value': theoretical_value,
                    'profit_margin': profit / theoretical_value if theoretical_value > 0 else 0
                })
        
        return arbitrage_opportunities
    
    def check_butterfly_arbitrage(self, options_data, option_type='call'):
        """
        Check for butterfly spread arbitrage
        """
        arbitrage_opportunities = []
        
        # Filter by option type
        data = options_data[options_data['option_type'] == option_type]
        strikes = sorted(data['strike'].unique())
        
        for i in range(1, len(strikes) - 1):
            k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
            
            # Check if strikes are equally spaced
            if k2 - k1 != k3 - k2:
                continue
            
            # Get option prices
            price_k1 = data[data['strike'] == k1]['market_price'].iloc[0]
            price_k2 = data[data['strike'] == k2]['market_price'].iloc[0]
            price_k3 = data[data['strike'] == k3]['market_price'].iloc[0]
            
            # Butterfly spread cost (buy 1 K1, sell 2 K2, buy 1 K3)
            butterfly_cost = price_k1 - 2 * price_k2 + price_k3
            
            # Transaction costs
            total_transaction_cost = self.transaction_cost * (price_k1 + 2 * price_k2 + price_k3)
            
            # Maximum profit should be (K2 - K1) for a butterfly
            max_theoretical_profit = k2 - k1
            
            # Check for arbitrage (butterfly should have non-negative cost)
            if butterfly_cost < -total_transaction_cost:
                profit = -butterfly_cost - total_transaction_cost
                
                arbitrage_opportunities.append({
                    'type': f'{option_type}_butterfly',
                    'strikes': [k1, k2, k3],
                    'direction': 'buy_butterfly',
                    'profit_potential': profit,
                    'butterfly_cost': butterfly_cost,
                    'max_profit': max_theoretical_profit,
                    'option_type': option_type
                })
        
        return arbitrage_opportunities
    
    def check_calendar_spread_arbitrage(self, near_expiry_data, far_expiry_data, 
                                      near_expiry_time, far_expiry_time):
        """
        Check for calendar spread arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Find common strikes
        common_strikes = set(near_expiry_data['strike']) & set(far_expiry_data['strike'])
        
        for strike in common_strikes:
            for option_type in ['call', 'put']:
                near_option = near_expiry_data[
                    (near_expiry_data['strike'] == strike) & 
                    (near_expiry_data['option_type'] == option_type)
                ]
                
                far_option = far_expiry_data[
                    (far_expiry_data['strike'] == strike) & 
                    (far_expiry_data['option_type'] == option_type)
                ]
                
                if len(near_option) == 0 or len(far_option) == 0:
                    continue
                
                near_price = near_option['market_price'].iloc[0]
                far_price = far_option['market_price'].iloc[0]
                
                # Calendar spread cost (sell near, buy far)
                calendar_cost = far_price - near_price
                
                # Transaction costs
                total_transaction_cost = self.transaction_cost * (near_price + far_price)
                
                # Far expiry should typically be more expensive
                if calendar_cost < -total_transaction_cost:
                    profit = -calendar_cost - total_transaction_cost
                    
                    arbitrage_opportunities.append({
                        'type': f'{option_type}_calendar',
                        'strike': strike,
                        'direction': 'reverse_calendar',
                        'profit_potential': profit,
                        'near_price': near_price,
                        'far_price': far_price,
                        'calendar_cost': calendar_cost,
                        'option_type': option_type
                    })
        
        return arbitrage_opportunities
    
    def find_all_arbitrage_opportunities(self, options_data, spot_price, 
                                       time_to_expiry, volatility=None):
        """
        Find all types of arbitrage opportunities
        """
        print("Searching for arbitrage opportunities...")
        
        all_opportunities = []
        
        # Separate calls and puts
        calls_data = options_data[options_data['option_type'] == 'call']
        puts_data = options_data[options_data['option_type'] == 'put']
        
        # Check put-call parity
        pcp_opportunities = self.check_put_call_parity(
            calls_data, puts_data, spot_price, time_to_expiry
        )
        all_opportunities.extend(pcp_opportunities)
        
        # Check box spreads
        box_opportunities = self.check_box_spread_arbitrage(
            calls_data, puts_data, time_to_expiry
        )
        all_opportunities.extend(box_opportunities)
        
        # Check butterfly spreads
        call_butterfly_opportunities = self.check_butterfly_arbitrage(options_data, 'call')
        put_butterfly_opportunities = self.check_butterfly_arbitrage(options_data, 'put')
        all_opportunities.extend(call_butterfly_opportunities)
        all_opportunities.extend(put_butterfly_opportunities)
        
        # If volatility is provided, check for theoretical vs market price discrepancies
        if volatility is not None:
            theoretical_opportunities = self.check_theoretical_price_discrepancies(
                options_data, spot_price, time_to_expiry, volatility
            )
            all_opportunities.extend(theoretical_opportunities)
        
        # Sort by profit potential
        all_opportunities = sorted(all_opportunities, key=lambda x: x['profit_potential'], reverse=True)
        
        self.opportunities = all_opportunities
        
        print(f"Found {len(all_opportunities)} potential arbitrage opportunities")
        return all_opportunities
    
    def check_theoretical_price_discrepancies(self, options_data, spot_price, 
                                            time_to_expiry, volatility):
        """
        Check for significant discrepancies between theoretical and market prices
        """
        discrepancies = []
        
        for _, option in options_data.iterrows():
            strike = option['strike']
            market_price = option['market_price']
            option_type = option['option_type']
            
            # Calculate theoretical price
            theoretical_price = self.bs_calculator.calculate_option_price(
                spot_price, strike, time_to_expiry, self.risk_free_rate, volatility, option_type
            )
            
            # Calculate percentage difference
            price_diff = abs(market_price - theoretical_price)
            price_diff_pct = price_diff / theoretical_price if theoretical_price > 0 else 0
            
            # Check if difference is significant (accounting for transaction costs)
            min_threshold = self.transaction_cost * 2
            
            if price_diff_pct > min_threshold and price_diff > 1:  # Minimum ₹1 difference
                direction = "buy" if market_price < theoretical_price else "sell"
                profit_potential = price_diff - (self.transaction_cost * market_price * 2)
                
                if profit_potential > 0:
                    discrepancies.append({
                        'type': 'theoretical_discrepancy',
                        'strike': strike,
                        'option_type': option_type,
                        'direction': direction,
                        'profit_potential': profit_potential,
                        'market_price': market_price,
                        'theoretical_price': theoretical_price,
                        'price_diff_pct': price_diff_pct
                    })
        
        return discrepancies
    
    def get_opportunity_summary(self):
        """Get summary of found opportunities"""
        if not self.opportunities:
            return "No arbitrage opportunities found"
        
        summary = {
            'total_opportunities': len(self.opportunities),
            'total_potential_profit': sum(op['profit_potential'] for op in self.opportunities),
            'opportunity_types': {},
            'top_opportunities': self.opportunities[:5]
        }
        
        # Count by type
        for op in self.opportunities:
            op_type = op['type']
            if op_type not in summary['opportunity_types']:
                summary['opportunity_types'][op_type] = 0
            summary['opportunity_types'][op_type] += 1
        
        return summary