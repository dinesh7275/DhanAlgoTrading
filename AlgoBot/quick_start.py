"""
Quick Start Script for AI Trading Bot
====================================

Easy setup and launch for the AI trading bot
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'tensorflow', 'pandas', 'numpy', 'scikit-learn',
        'dhanhq', 'flask', 'plotly', 'cryptography',
        'yfinance', 'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print(f"\n📦 Or install all requirements:")
        print(f"   pip install -r requirements_live_trading.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def main_menu():
    """Display main menu and handle user choice"""
    while True:
        print("\n🤖 AI Trading Bot - Quick Start")
        print("=" * 40)
        print("1. 📊 Setup Trading Bot (First Time)")
        print("2. 📄 Start Paper Trading (Safe)")
        print("3. 🌐 Start Web Dashboard")
        print("4. ⚙️  Configuration Menu")
        print("5. 🚨 Start Live Trading (Real Money)")
        print("6. 📖 View Documentation")
        print("7. ❌ Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            setup_bot()
        elif choice == '2':
            start_paper_trading()
        elif choice == '3':
            start_web_dashboard()
        elif choice == '4':
            config_menu()
        elif choice == '5':
            start_live_trading()
        elif choice == '6':
            show_documentation()
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-7.")


def setup_bot():
    """Setup the trading bot"""
    print("\n🔧 Setting up AI Trading Bot...")
    
    try:
        from live_trading import setup_trading_environment
        
        config_manager, credentials = setup_trading_environment()
        if config_manager and credentials:
            print("✅ Bot setup completed successfully!")
            print("\n💡 Next steps:")
            print("   - Review your configuration")
            print("   - Start with paper trading to test")
            print("   - Monitor performance before going live")
        else:
            print("❌ Setup incomplete")
    
    except ImportError:
        print("❌ Trading modules not found. Check installation.")
    except Exception as e:
        print(f"❌ Setup error: {e}")


def start_paper_trading():
    """Start paper trading"""
    print("\n📄 Starting Paper Trading...")
    print("💡 This is safe - no real money will be used")
    
    try:
        from main_trading_bot import start_paper_trading
        start_paper_trading()
    except ImportError:
        print("❌ Trading modules not found")
    except Exception as e:
        print(f"❌ Error: {e}")


def start_web_dashboard():
    """Start web dashboard"""
    print("\n🌐 Starting Web Dashboard...")
    
    try:
        from main_trading_bot import start_web_dashboard
        start_web_dashboard()
    except ImportError:
        print("❌ Trading modules not found")
    except Exception as e:
        print(f"❌ Error: {e}")


def config_menu():
    """Configuration menu"""
    print("\n⚙️ Configuration Menu")
    
    try:
        from live_trading import quick_config_update
        quick_config_update()
    except ImportError:
        print("❌ Configuration modules not found")
    except Exception as e:
        print(f"❌ Error: {e}")


def start_live_trading():
    """Start live trading with warnings"""
    print("\n🚨 LIVE TRADING MODE")
    print("=" * 30)
    print("⚠️  WARNING: This will trade with REAL MONEY!")
    print("⚠️  Make sure you have:")
    print("   ✓ Tested with paper trading")
    print("   ✓ Verified your configuration")
    print("   ✓ Set appropriate risk limits")
    print("   ✓ Have sufficient account balance")
    
    print("\n🔒 Safety Checks:")
    confirm1 = input("Have you tested with paper trading? (yes/no): ").lower()
    if confirm1 != 'yes':
        print("❌ Please test with paper trading first")
        return
    
    confirm2 = input("Do you understand the risks? (yes/no): ").lower()
    if confirm2 != 'yes':
        print("❌ Live trading cancelled")
        return
    
    confirm3 = input("Type 'START LIVE TRADING' to confirm: ")
    if confirm3 != 'START LIVE TRADING':
        print("❌ Live trading not confirmed")
        return
    
    try:
        from main_trading_bot import start_live_trading_confirmed
        start_live_trading_confirmed()
    except ImportError:
        print("❌ Trading modules not found")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_documentation():
    """Show documentation"""
    print("\n📖 AI Trading Bot Documentation")
    print("=" * 40)
    
    docs = """
🤖 AI TRADING BOT FEATURES:
- Volatility prediction using LSTM models
- Price movement prediction with CNN/LSTM
- Options anomaly detection
- Risk management and position sizing
- Real-time monitoring dashboard
- Paper trading for safe testing

🔧 SETUP PROCESS:
1. Install dependencies: pip install -r requirements_live_trading.txt
2. Run setup: python quick_start.py -> Option 1
3. Enter your Dhan API credentials
4. Configure trading parameters
5. Start with paper trading

📊 TRADING MODES:
- Paper Trading: Simulated trading (recommended for testing)
- Live Trading: Real money trading (use with caution)
- Console Mode: Text-based monitoring
- Web Dashboard: Browser-based monitoring

⚠️  RISK MANAGEMENT:
- Built-in stop losses and position sizing
- Daily and portfolio loss limits
- Circuit breakers for extreme scenarios
- Real-time risk monitoring

🔐 SECURITY:
- Encrypted credential storage
- Secure API communication
- No credential logging

📈 AI MODELS:
- Volatility Prediction: LSTM for IV forecasting
- Price Movement: CNN+LSTM for direction prediction
- Anomaly Detection: Autoencoder for unusual patterns
- Risk Assessment: Portfolio risk scoring

🌐 WEB DASHBOARD:
- Real-time portfolio metrics
- Live position monitoring
- AI signal tracking
- Performance charts
- Emergency controls

💡 BEST PRACTICES:
1. Always start with paper trading
2. Test extensively before going live
3. Set conservative risk limits
4. Monitor performance regularly
5. Keep credentials secure
6. Review AI signals manually

🆘 SUPPORT:
- Check logs for detailed information
- Use paper trading to debug issues
- Review configuration settings
- Ensure API credentials are correct

⚡ QUICK COMMANDS:
- Paper Trading: python main_trading_bot.py --paper
- Web Dashboard: python main_trading_bot.py --mode dashboard
- Configuration: python main_trading_bot.py --config
- Full Setup: python main_trading_bot.py --setup
"""
    
    print(docs)
    input("\nPress Enter to continue...")


def check_dhan_credentials():
    """Check if Dhan credentials are configured"""
    try:
        from live_trading import ConfigManager
        config_manager = ConfigManager()
        credentials = config_manager.load_credentials()
        return credentials is not None
    except:
        return False


if __name__ == "__main__":
    print("🤖 AI Trading Bot - Quick Start")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install required packages first")
        sys.exit(1)
    
    # Check if bot is already configured
    if check_dhan_credentials():
        print("✅ Bot is configured and ready to use")
    else:
        print("⚙️ Bot needs initial setup")
        print("💡 Select option 1 to configure credentials and settings")
    
    # Show main menu
    main_menu()