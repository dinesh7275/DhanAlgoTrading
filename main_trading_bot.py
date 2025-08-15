"""
Main AI Trading Bot
==================

Complete AI-powered trading bot with Dhan API integration
"""

import sys
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading import (
    LiveTradingManager, TradingDashboard, ConsoleMonitor,
    LiveRiskManager, setup_trading_environment, 
    create_dashboard_template, quick_config_update
)


class CompleteTradingBot:
    """
    Complete AI Trading Bot with all components integrated
    """
    
    def __init__(self, config_manager=None, credentials=None):
        self.config_manager = config_manager
        self.credentials = credentials
        
        # Components
        self.trading_manager = None
        self.risk_manager = None
        self.dashboard = None
        self.console_monitor = None
        
        print("🤖 Complete AI Trading Bot Initialized")
    
    def setup_components(self):
        """Setup all trading components"""
        if not self.credentials:
            print("❌ No credentials provided")
            return False
        
        try:
            # Get configuration
            trading_config = self.config_manager.get_trading_config()
            risk_config = self.config_manager.get_risk_config()
            
            # Initialize trading manager
            self.trading_manager = LiveTradingManager(
                client_id=self.credentials['client_id'],
                access_token=self.credentials['access_token'],
                initial_capital=trading_config.get('initial_capital', 1000000),
                max_risk_per_trade=trading_config.get('max_risk_per_trade', 0.02)
            )
            
            # Set trading parameters
            self.trading_manager.paper_trading = trading_config.get('paper_trading', True)
            self.trading_manager.market_hours_only = trading_config.get('market_hours_only', True)
            self.trading_manager.max_positions = trading_config.get('max_positions', 5)
            
            # Initialize risk manager
            self.risk_manager = LiveRiskManager(
                initial_capital=trading_config.get('initial_capital', 1000000),
                max_daily_loss=trading_config.get('max_daily_loss', 0.05),
                max_portfolio_loss=trading_config.get('max_portfolio_loss', 0.10)
            )
            
            # Integrate risk manager with trading manager
            self.trading_manager.risk_manager = self.risk_manager
            
            # Initialize monitoring
            self.dashboard = TradingDashboard(self.trading_manager)
            self.console_monitor = ConsoleMonitor(self.trading_manager)
            
            print("✅ All components setup successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up components: {e}")
            return False
    
    def start_trading(self, mode='console'):
        """Start trading with specified monitoring mode"""
        if not self.trading_manager:
            print("❌ Trading manager not initialized")
            return
        
        # Validate setup
        if not self.trading_manager.validate_trading_setup():
            print("❌ Trading setup validation failed")
            return
        
        print(f"\n🚀 Starting AI Trading Bot")
        print(f"📊 Mode: {mode}")
        print(f"💰 Capital: ₹{self.trading_manager.current_capital:,}")
        print(f"📄 Paper Trading: {self.trading_manager.paper_trading}")
        print(f"⏱️ Update Interval: {self.config_manager.get_trading_config().get('update_interval', 30)}s")
        
        if mode == 'dashboard':
            # Create dashboard template
            create_dashboard_template()
            
            # Start web dashboard
            print("🌐 Starting web dashboard...")
            self.dashboard.start_web_server()
            
        elif mode == 'console':
            # Start console monitoring
            print("📊 Starting console monitoring...")
            self.console_monitor.start_console_monitoring(
                update_interval=self.config_manager.get_trading_config().get('update_interval', 30)
            )
            
        elif mode == 'headless':
            # Start headless trading
            print("🔇 Starting headless trading...")
            self.trading_manager.start_live_trading(
                update_interval=self.config_manager.get_trading_config().get('update_interval', 30)
            )
        
        else:
            print(f"❌ Unknown mode: {mode}")
    
    def stop_trading(self):
        """Stop all trading activities"""
        print("🛑 Stopping trading bot...")
        
        if self.trading_manager:
            self.trading_manager.stop_trading()
        
        if self.console_monitor:
            self.console_monitor.stop_monitoring()
        
        if self.dashboard:
            self.dashboard.stop_data_updates()
        
        print("✅ Trading bot stopped")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AI Trading Bot with Dhan API')
    parser.add_argument('--mode', choices=['console', 'dashboard', 'headless'], 
                       default='console', help='Monitoring mode')
    parser.add_argument('--config', action='store_true', 
                       help='Run configuration setup')
    parser.add_argument('--setup', action='store_true', 
                       help='Run complete setup (credentials + config)')
    parser.add_argument('--paper', action='store_true', 
                       help='Force paper trading mode')
    parser.add_argument('--live', action='store_true', 
                       help='Force live trading mode (DANGER!)')
    
    args = parser.parse_args()
    
    print(f"🤖 AI Trading Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Configuration setup
    if args.config:
        quick_config_update()
        return
    
    # Complete setup
    if args.setup:
        config_manager, credentials = setup_trading_environment()
        if not config_manager or not credentials:
            print("❌ Setup failed")
            return
        print("✅ Setup completed successfully")
        return
    
    # Load existing configuration
    try:
        config_manager, credentials = setup_trading_environment()
        if not config_manager or not credentials:
            print("❌ Configuration or credentials missing")
            print("💡 Run with --setup to configure the bot")
            return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Override trading mode if specified
    trading_config = config_manager.get_trading_config()
    if args.paper:
        trading_config['paper_trading'] = True
        print("📄 Paper trading mode forced via command line")
    elif args.live:
        confirm = input("⚠️ LIVE TRADING MODE - This will use real money! Type 'CONFIRM' to proceed: ")
        if confirm == 'CONFIRM':
            trading_config['paper_trading'] = False
            print("🚨 LIVE TRADING MODE ACTIVATED")
        else:
            print("❌ Live trading not confirmed - using paper trading")
            trading_config['paper_trading'] = True
    
    # Create and start trading bot
    try:
        bot = CompleteTradingBot(config_manager, credentials)
        
        if not bot.setup_components():
            print("❌ Failed to setup trading components")
            return
        
        # Start trading
        bot.start_trading(mode=args.mode)
        
    except KeyboardInterrupt:
        print("\n👋 Trading stopped by user")
    except Exception as e:
        print(f"❌ Trading error: {e}")
    finally:
        if 'bot' in locals():
            bot.stop_trading()


# Quick start functions for different use cases
def start_paper_trading():
    """Quick start paper trading with console monitoring"""
    print("🚀 Quick Start: Paper Trading")
    
    config_manager, credentials = setup_trading_environment()
    if not config_manager or not credentials:
        return
    
    # Force paper trading
    trading_config = config_manager.get_trading_config()
    trading_config['paper_trading'] = True
    
    bot = CompleteTradingBot(config_manager, credentials)
    if bot.setup_components():
        bot.start_trading(mode='console')


def start_web_dashboard():
    """Quick start with web dashboard"""
    print("🚀 Quick Start: Web Dashboard")
    
    config_manager, credentials = setup_trading_environment()
    if not config_manager or not credentials:
        return
    
    bot = CompleteTradingBot(config_manager, credentials)
    if bot.setup_components():
        bot.start_trading(mode='dashboard')


def start_live_trading_confirmed():
    """Quick start live trading (with confirmation)"""
    print("🚀 Quick Start: Live Trading")
    print("⚠️ WARNING: This will trade with real money!")
    
    confirm = input("Type 'I UNDERSTAND THE RISKS' to proceed: ")
    if confirm != 'I UNDERSTAND THE RISKS':
        print("❌ Live trading not confirmed")
        return
    
    config_manager, credentials = setup_trading_environment()
    if not config_manager or not credentials:
        return
    
    # Force live trading
    trading_config = config_manager.get_trading_config()
    trading_config['paper_trading'] = False
    
    bot = CompleteTradingBot(config_manager, credentials)
    if bot.setup_components():
        bot.start_trading(mode='console')


if __name__ == "__main__":
    main()