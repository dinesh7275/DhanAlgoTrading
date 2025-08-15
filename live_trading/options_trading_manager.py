"""
Options Trading Manager for Indian Markets
==========================================

Specialized module for options buying strategies (CE/PE only)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import logging
import yfinance as yf
from dhanhq import dhanhq
import warnings
warnings.filterwarnings('ignore')

from .dhan_data_fetcher import DhanLiveDataFetcher
from .ai_ensemble import TradingSignalEnsemble

class OptionsPosition:
    """Individual options position tracking"""
    
    def __init__(self, strike, option_type, premium, quantity, expiry_date):
        self.strike = strike
        self.option_type = option_type  # 'CE' or 'PE'
        self.premium = premium
        self.quantity = quantity
        self.expiry_date = expiry_date
        self.entry_time = datetime.now()
        self.current_premium = premium
        self.pnl = 0
        self.pnl_percent = 0
        
    def update_premium(self, current_premium):
        """Update current premium and calculate P&L"""
        self.current_premium = current_premium
        gross_pnl = (current_premium - self.premium) * self.quantity
        
        # Apply Indian taxes
        taxes = self.calculate_taxes(gross_pnl)
        self.pnl = gross_pnl - taxes
        self.pnl_percent = (self.pnl / (self.premium * self.quantity)) * 100
        
    def calculate_taxes(self, gross_pnl):
        """Calculate Indian taxes on options trading"""
        if gross_pnl <= 0:
            return 0
            
        # STT on options (0.017% on premium)
        stt = self.current_premium * self.quantity * 0.00017
        
        # Exchange charges (approx 0.0019%)
        exchange_charges = self.current_premium * self.quantity * 0.000019
        
        # Brokerage (assume ₹20 per trade)
        brokerage = 20
        
        # GST on brokerage (18%)
        gst = brokerage * 0.18
        
        total_taxes = stt + exchange_charges + brokerage + gst
        return total_taxes


class OptionsChainAnalyzer:
    """Analyze options chain for best trading opportunities"""
    
    def __init__(self):
        self.min_profit_points = 12  # Minimum points for profitability
        self.max_profit_points = 15  # Target profit points
        
    def get_nifty_options_chain(self, spot_price, days_to_expiry=7):
        """Get or simulate Nifty options chain"""
        # For now, simulate options chain - replace with real NSE data
        strikes = self.generate_strike_prices(spot_price)
        options_data = []
        
        for strike in strikes:
            ce_data = self.calculate_option_premium(spot_price, strike, 'CE', days_to_expiry)
            pe_data = self.calculate_option_premium(spot_price, strike, 'PE', days_to_expiry)
            
            options_data.extend([ce_data, pe_data])
            
        return pd.DataFrame(options_data)
    
    def generate_strike_prices(self, spot_price):
        """Generate relevant strike prices around spot"""
        base_strike = round(spot_price / 50) * 50  # Round to nearest 50
        strikes = []
        
        # Generate strikes from -500 to +500 points
        for offset in range(-500, 550, 50):
            strikes.append(base_strike + offset)
            
        return strikes
    
    def calculate_option_premium(self, spot_price, strike, option_type, days_to_expiry):
        """Calculate theoretical option premium (simplified)"""
        time_to_expiry = days_to_expiry / 365
        volatility = 0.20  # Assume 20% volatility
        
        if option_type == 'CE':
            intrinsic = max(0, spot_price - strike)
            moneyness = spot_price / strike
        else:
            intrinsic = max(0, strike - spot_price)
            moneyness = strike / spot_price
        
        # Simplified time value calculation
        if moneyness > 0.98 and moneyness < 1.02:  # ATM
            time_value = spot_price * volatility * np.sqrt(time_to_expiry) * 0.4
        elif moneyness > 1.02 or moneyness < 0.98:  # OTM/ITM
            time_value = spot_price * volatility * np.sqrt(time_to_expiry) * 0.2
        else:
            time_value = 5
            
        premium = intrinsic + time_value + np.random.uniform(-2, 2)
        premium = max(0.5, premium)
        
        return {
            'strike': strike,
            'option_type': option_type,
            'premium': round(premium, 2),
            'intrinsic': intrinsic,
            'time_value': time_value,
            'moneyness': moneyness,
            'days_to_expiry': days_to_expiry
        }
    
    def find_profitable_strikes(self, options_chain, signal, spot_price):
        """Find strikes with 12-15 point profit probability"""
        profitable_options = []
        
        for _, option in options_chain.iterrows():
            profit_potential = self.calculate_profit_potential(
                option, signal, spot_price
            )
            
            if self.min_profit_points <= profit_potential <= self.max_profit_points:
                profitable_options.append({
                    'strike': option['strike'],
                    'option_type': option['option_type'],
                    'premium': option['premium'],
                    'profit_potential': profit_potential,
                    'risk_reward': profit_potential / option['premium']
                })
                
        # Sort by risk-reward ratio
        return sorted(profitable_options, key=lambda x: x['risk_reward'], reverse=True)
    
    def calculate_profit_potential(self, option, signal, spot_price):
        """Calculate potential profit points based on signal"""
        strike = option['strike']
        option_type = option['option_type']
        
        if signal == 'BUY' and option_type == 'CE':
            # Bullish with Call options
            if spot_price < strike:  # OTM Call
                return max(0, (spot_price + 50) - strike)  # Assume 50 point move
            else:  # ITM Call
                return 50  # Direct premium gain
                
        elif signal == 'SELL' and option_type == 'PE':
            # Bearish with Put options  
            if spot_price > strike:  # OTM Put
                return max(0, strike - (spot_price - 50))  # Assume 50 point move
            else:  # ITM Put
                return 50  # Direct premium gain
                
        return 0


class OptionsLotSizeCalculator:
    """Calculate optimal lot sizes for ₹10,000 capital"""
    
    def __init__(self, capital=10000):
        self.capital = capital
        self.nifty_lot_size = 50  # Standard Nifty lot size
        self.max_risk_per_trade = 0.3  # 30% max risk per trade for small capital
        
    def calculate_position_size(self, premium, target_return=0.1):
        """Calculate position size based on capital and risk"""
        max_loss_amount = self.capital * self.max_risk_per_trade
        
        # Calculate lots based on premium and risk
        premium_per_lot = premium * self.nifty_lot_size
        
        if premium_per_lot > max_loss_amount:
            # Can't afford even 1 lot
            return {
                'lots': 0,
                'quantity': 0,
                'capital_required': 0,
                'max_loss': 0,
                'capital_utilization': 0
            }
            
        # Calculate how many lots we can buy
        lots_affordable = int(self.capital / premium_per_lot)
        lots_by_risk = int(max_loss_amount / premium_per_lot)
        
        # Take minimum of affordable and risk-based lots
        optimal_lots = min(lots_affordable, lots_by_risk, 2)  # Max 2 lots
        
        return {
            'lots': optimal_lots,
            'quantity': optimal_lots * self.nifty_lot_size,
            'capital_required': optimal_lots * premium_per_lot,
            'max_loss': optimal_lots * premium_per_lot,
            'capital_utilization': (optimal_lots * premium_per_lot) / self.capital
        }


class OptionsRiskManager:
    """Enhanced risk management for options trading"""
    
    def __init__(self, daily_profit_target=0.1, max_trades_per_day=10):
        self.daily_profit_target = daily_profit_target  # 10% daily target
        self.max_trades_per_day = max_trades_per_day
        self.trades_today = 0
        self.daily_pnl = 0
        self.last_reset_date = datetime.now().date()
        
    def reset_daily_counters(self):
        """Reset daily counters at market open"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.trades_today = 0
            self.daily_pnl = 0
            self.last_reset_date = today
            
    def can_place_trade(self, initial_capital):
        """Check if we can place a new trade"""
        self.reset_daily_counters()
        
        # Check trade count limit
        if self.trades_today >= self.max_trades_per_day:
            return False, "Maximum trades per day reached"
            
        # Check if daily target achieved
        daily_return = self.daily_pnl / initial_capital
        if daily_return >= self.daily_profit_target:
            return False, "Daily profit target achieved - stopping trading"
            
        return True, "OK"
        
    def update_daily_pnl(self, trade_pnl):
        """Update daily P&L"""
        self.daily_pnl += trade_pnl
        self.trades_today += 1


class IndianOptionsTrader:
    """Main options trading manager for Indian markets"""
    
    def __init__(self, client_id, access_token, initial_capital=10000):
        self.client_id = client_id
        self.access_token = access_token
        self.dhan = dhanhq(client_id, access_token)
        
        # Core parameters for Indian options trading
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_capital = initial_capital  # For daily compounding
        
        # Components
        self.data_fetcher = DhanLiveDataFetcher(client_id, access_token)
        self.ai_ensemble = TradingSignalEnsemble()
        self.options_analyzer = OptionsChainAnalyzer()
        self.lot_calculator = OptionsLotSizeCalculator(initial_capital)
        self.risk_manager = OptionsRiskManager()
        
        # Trading state
        self.active_positions = {}
        self.trade_history = []
        self.paper_trading = True  # Set to False for live trading
        
        # Setup logging
        self._setup_logging()
        
        print(f"Indian Options Trader Initialized")
        print(f"Capital: ₹{initial_capital:,}")
        print(f"Trading Mode: {'Paper' if self.paper_trading else 'LIVE'}")
        
    def _setup_logging(self):
        """Setup trading logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'options_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def implement_daily_compounding(self):
        """Implement daily compounding of capital"""
        today = datetime.now().date()
        
        # Calculate daily return
        daily_return = (self.current_capital - self.daily_capital) / self.daily_capital
        
        if daily_return != 0:
            print(f"Daily Return: {daily_return:.2%}")
            
            # Update daily capital for tomorrow
            self.daily_capital = self.current_capital
            self.lot_calculator.capital = self.current_capital
            
            self.logger.info(f"Daily compounding: New capital = ₹{self.current_capital:,}")
            
    def get_market_signal(self):
        """Get AI signal for options trading"""
        try:
            # Get current market data
            market_data = self.data_fetcher.get_comprehensive_market_data()
            if not market_data or not market_data.get('nifty'):
                return None
                
            # Get AI ensemble signal
            signal_result = self.ai_ensemble.generate_ensemble_signal(
                market_data=market_data,
                portfolio_value=self.current_capital,
                current_positions=list(self.active_positions.values())
            )
            
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error getting market signal: {e}")
            return None
            
    def find_trading_opportunity(self, signal_result, spot_price):
        """Find the best options trading opportunity"""
        signal = signal_result['final_signal']
        confidence = signal_result['confidence']
        
        if signal == 'HOLD' or confidence < 0.65:
            return None
            
        # Get options chain
        options_chain = self.options_analyzer.get_nifty_options_chain(spot_price)
        
        # Find profitable strikes
        profitable_options = self.options_analyzer.find_profitable_strikes(
            options_chain, signal, spot_price
        )
        
        if not profitable_options:
            return None
            
        # Select best option (highest risk-reward)
        best_option = profitable_options[0]
        
        # Calculate position size
        position_info = self.lot_calculator.calculate_position_size(
            best_option['premium']
        )
        
        if position_info['lots'] == 0:
            return None
            
        return {
            'action': 'BUY',
            'strike': best_option['strike'],
            'option_type': best_option['option_type'],
            'premium': best_option['premium'],
            'lots': position_info['lots'],
            'quantity': position_info['quantity'],
            'capital_required': position_info['capital_required'],
            'expected_profit': best_option['profit_potential'],
            'signal_confidence': confidence
        }
        
    def execute_options_trade(self, opportunity):
        """Execute options trade"""
        try:
            # Check risk management
            can_trade, reason = self.risk_manager.can_place_trade(self.current_capital)
            if not can_trade:
                self.logger.info(f"Trade blocked: {reason}")
                return False
                
            if self.paper_trading:
                return self._execute_paper_options_trade(opportunity)
            else:
                return self._execute_live_options_trade(opportunity)
                
        except Exception as e:
            self.logger.error(f"Error executing options trade: {e}")
            return False
            
    def _execute_paper_options_trade(self, opportunity):
        """Execute paper options trade"""
        position_id = f"OPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create position
        position = OptionsPosition(
            strike=opportunity['strike'],
            option_type=opportunity['option_type'],
            premium=opportunity['premium'],
            quantity=opportunity['quantity'],
            expiry_date=datetime.now() + timedelta(days=7)
        )
        
        self.active_positions[position_id] = position
        
        # Update capital
        cost = opportunity['capital_required']
        self.current_capital -= cost
        
        # Update risk manager
        self.risk_manager.update_daily_pnl(-cost)  # Initially a loss
        
        self.logger.info(
            f"PAPER TRADE: BUY {opportunity['quantity']} "
            f"{opportunity['strike']}{opportunity['option_type']} @ ₹{opportunity['premium']}"
        )
        
        return True
        
    def update_positions(self, current_spot_price):
        """Update all active positions"""
        for position_id, position in list(self.active_positions.items()):
            # Calculate current premium (simplified)
            current_premium = self.estimate_current_premium(position, current_spot_price)
            position.update_premium(current_premium)
            
            # Check exit conditions
            if self.should_exit_position(position):
                self.close_position(position_id, "Exit condition met")
                
    def estimate_current_premium(self, position, spot_price):
        """Estimate current option premium based on spot price"""
        intrinsic = 0
        
        if position.option_type == 'CE':
            intrinsic = max(0, spot_price - position.strike)
        else:
            intrinsic = max(0, position.strike - spot_price)
            
        # Add time value (simplified)
        time_value = max(1, position.premium * 0.3)  # Assume some time value remains
        
        return intrinsic + time_value
        
    def should_exit_position(self, position):
        """Determine if position should be exited"""
        # Exit if profit > 15 points or loss > 10 points
        return position.pnl_percent > 50 or position.pnl_percent < -80
        
    def close_position(self, position_id, reason):
        """Close an options position"""
        if position_id not in self.active_positions:
            return
            
        position = self.active_positions[position_id]
        
        # Calculate final P&L
        final_pnl = position.pnl
        
        # Update capital
        proceeds = position.current_premium * position.quantity
        self.current_capital += proceeds
        
        # Update risk manager
        self.risk_manager.update_daily_pnl(final_pnl)
        
        # Add to trade history
        trade_record = {
            'position_id': position_id,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'strike': position.strike,
            'option_type': position.option_type,
            'quantity': position.quantity,
            'entry_premium': position.premium,
            'exit_premium': position.current_premium,
            'pnl': final_pnl,
            'pnl_percent': position.pnl_percent,
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        
        # Remove from active positions
        del self.active_positions[position_id]
        
        self.logger.info(
            f"POSITION CLOSED: {position.strike}{position.option_type} - "
            f"P&L: ₹{final_pnl:.2f} ({position.pnl_percent:.1f}%) - {reason}"
        )
        
    def start_options_trading(self, update_interval=30):
        """Start options trading loop"""
        self.logger.info("Starting Indian Options Trading")
        self.logger.info(f"Capital: ₹{self.current_capital:,}")
        self.logger.info(f"Daily Target: 10% (₹{self.current_capital * 0.1:,.0f})")
        
        try:
            while True:
                # Check if market is open (9:15 AM - 3:30 PM IST)
                if not self.is_market_open():
                    print("Market closed - waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                    
                # Implement daily compounding
                self.implement_daily_compounding()
                
                # Get current market data
                market_data = self.data_fetcher.get_comprehensive_market_data()
                if not market_data:
                    time.sleep(update_interval)
                    continue
                    
                spot_price = market_data.get('nifty', {}).get('ltp', 0)
                if spot_price == 0:
                    time.sleep(update_interval)
                    continue
                
                # Update existing positions
                self.update_positions(spot_price)
                
                # Look for new trading opportunities
                signal_result = self.get_market_signal()
                if signal_result:
                    opportunity = self.find_trading_opportunity(signal_result, spot_price)
                    if opportunity:
                        success = self.execute_options_trade(opportunity)
                        if success:
                            self.logger.info("New position opened successfully")
                
                # Print status
                self.print_status_update()
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Trading error: {e}")
        finally:
            self.close_all_positions()
            
    def is_market_open(self):
        """Check if Indian market is open"""
        now = datetime.now()
        
        # Check if weekday
        if now.weekday() > 4:  # Weekend
            return False
            
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    def print_status_update(self):
        """Print current trading status"""
        daily_pnl = self.current_capital - self.daily_capital
        daily_return = daily_pnl / self.daily_capital * 100
        
        print(f"\n{'='*60}")
        print(f"OPTIONS TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Current Capital: ₹{self.current_capital:,.2f}")
        print(f"Daily P&L: ₹{daily_pnl:,.2f} ({daily_return:+.2f}%)")
        print(f"Active Positions: {len(self.active_positions)}")
        print(f"Trades Today: {self.risk_manager.trades_today}/{self.risk_manager.max_trades_per_day}")
        
        if self.active_positions:
            print(f"\nActive Options Positions:")
            for pos_id, position in self.active_positions.items():
                print(f"  {position.strike}{position.option_type}: ₹{position.pnl:.2f} ({position.pnl_percent:.1f}%)")
                
    def close_all_positions(self):
        """Close all open positions"""
        print("Closing all positions...")
        for position_id in list(self.active_positions.keys()):
            self.close_position(position_id, "End of trading session")
            
        # Print final summary
        self.print_final_summary()
        
    def print_final_summary(self):
        """Print final trading summary"""
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        total_trades = len(self.trade_history)
        win_rate = (winning_trades / max(1, total_trades)) * 100
        
        print(f"\n{'='*60}")
        print(f"FINAL OPTIONS TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Capital: ₹{self.initial_capital:,}")
        print(f"Final Capital: ₹{self.current_capital:,}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Profit Target (10%): {'✓ ACHIEVED' if total_return >= 10 else '✗ NOT ACHIEVED'}")


# Quick start function
def start_indian_options_trading(client_id, access_token, capital=10000, paper_trading=True):
    """Quick start Indian options trading"""
    print("Starting Indian Options Trading Bot...")
    
    trader = IndianOptionsTrader(client_id, access_token, capital)
    trader.paper_trading = paper_trading
    
    try:
        trader.start_options_trading()
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"Trading error: {e}")
    
    return trader