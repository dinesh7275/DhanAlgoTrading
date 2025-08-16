#!/usr/bin/env python3
"""
Test Dhan Option Chain APIs
===========================

Test script to verify the option chain API calls are working correctly.
"""

import sys
import os
sys.path.append('AIBot')

from dotenv import load_dotenv
from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_option_chain_apis():
    """Test Dhan option chain APIs"""
    try:
        # Load environment variables
        load_dotenv()
        
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            print("Missing credentials in .env file")
            return False
        
        # Create credentials
        credentials = DhanCredentials(
            client_id=client_id,
            access_token=access_token
        )
        
        # Initialize client
        print("Initializing Dhan API client...")
        client = DhanAPIClient(credentials)
        
        # Test authentication
        if not client.authenticate():
            print("Authentication failed")
            return False
        
        print("Authentication successful!")
        
        # Test option chain expiry list
        print("\n=== Testing Option Chain Expiry List ===")
        expiry_list = client.get_option_chain_expiry_list(
            underlying_scrip=13,  # NIFTY
            underlying_seg="IDX_I"
        )
        
        if expiry_list:
            print(f"Found {len(expiry_list)} expiries:")
            for expiry in expiry_list[:5]:  # Show first 5
                print(f"  - {expiry}")
        else:
            print("No expiries found")
        
        # Test option chain for first expiry
        if expiry_list:
            print(f"\n=== Testing Option Chain for {expiry_list[0]} ===")
            option_chain = client.get_option_chain(
                underlying_scrip=13,  # NIFTY
                underlying_seg="IDX_I",
                expiry=expiry_list[0]
            )
            
            if option_chain:
                print(f"Found {len(option_chain)} option contracts:")
                
                # Show some sample contracts
                ce_contracts = [opt for opt in option_chain if opt.option_type == 'CE'][:3]
                pe_contracts = [opt for opt in option_chain if opt.option_type == 'PE'][:3]
                
                print("\nSample CE contracts:")
                for opt in ce_contracts:
                    print(f"  {opt.symbol}: Strike {opt.strike_price}, LTP {opt.ltp}, IV {opt.iv}")
                
                print("\nSample PE contracts:")
                for opt in pe_contracts:
                    print(f"  {opt.symbol}: Strike {opt.strike_price}, LTP {opt.ltp}, IV {opt.iv}")
            else:
                print("No option chain data found")
        
        return True
    
    except Exception as e:
        print(f"Error testing option chain APIs: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Dhan Option Chain APIs")
    print("=" * 40)
    
    success = test_option_chain_apis()
    
    if success:
        print("\nOption chain API tests completed!")
    else:
        print("\nOption chain API tests failed!")