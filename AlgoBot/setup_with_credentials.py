"""
Setup Script with Your Dhan Credentials
======================================

Replace YOUR_CLIENT_ID and YOUR_ACCESS_TOKEN with your actual Dhan API credentials
"""

from live_trading import ConfigManager

def setup_bot_with_credentials():
    """Setup bot with your Dhan credentials"""
    
    # ============================================================================
    # ENTER YOUR DHAN API CREDENTIALS HERE:
    # ============================================================================
    
    # Replace these with your actual Dhan API credentials
    CLIENT_ID = "1107321060"  # Replace with your Dhan Client ID
    ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU3MTM4NzgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNzMyMTA2MCJ9.n_2HhEW9ePhAfi63KoxQskzohVPi4N8F_RWn-a9rqTbne5GX7DHRTF9NpU4LEyf1dC8J-M32Fuk-EbXlOYOWOA"  # Replace with your Dhan Access Token
    
    # ============================================================================
    
    if CLIENT_ID == "YOUR_CLIENT_ID" or ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("Please edit this file and enter your Dhan API credentials")
        print("Replace YOUR_CLIENT_ID and YOUR_ACCESS_TOKEN with your actual credentials")
        print("You can find these in your Dhan API dashboard")
        return False
    
    print("Setting up Dhan API credentials...")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Save credentials
    credentials = {
        'client_id': CLIENT_ID,
        'access_token': ACCESS_TOKEN,
        'created_at': '2025-01-14T23:20:00'
    }
    
    if config_manager.save_credentials(credentials):
        print("Credentials saved successfully!")
        
        # Save default configuration
        config_manager.save_config()
        print("Configuration saved!")
        
        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print("1. Test with paper trading: python main_trading_bot.py --paper")
        print("2. Open web dashboard: python main_trading_bot.py --mode dashboard")
        print("3. When ready for live trading: python main_trading_bot.py --live")
        
        return True
    else:
        print("Failed to save credentials")
        return False

if __name__ == "__main__":
    print("AI Trading Bot - Credential Setup")
    print("=" * 50)
    
    if setup_bot_with_credentials():
        print("\nYour AI trading bot is ready!")
        
        # Test the setup
        try:
            from live_trading import DhanLiveDataFetcher
            print("\nTesting API connection...")
            
            # Load saved credentials for testing
            config_manager = ConfigManager()
            saved_creds = config_manager.load_credentials()
            
            if saved_creds:
                data_fetcher = DhanLiveDataFetcher(
                    saved_creds['client_id'], 
                    saved_creds['access_token']
                )
                
                if data_fetcher.validate_connection():
                    print("Dhan API connection successful!")
                else:
                    print("Dhan API connection failed - please check your credentials")
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            print("You can still proceed with paper trading to test the system")
    
    else:
        print("\nSetup incomplete")
        print("Please edit this file and enter your Dhan API credentials")