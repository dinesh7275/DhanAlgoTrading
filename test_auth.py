#!/usr/bin/env python3
"""
Test Dhan API Authentication
============================

Test script to verify Dhan API authentication is working correctly.
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

def test_dhan_auth():
    """Test Dhan API authentication"""
    try:
        # Load environment variables
        load_dotenv()
        
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or client_id == 'your_client_id_here':
            print("DHAN_CLIENT_ID not set in .env file")
            print("Please update .env file with your actual Dhan client ID")
            return False
        
        if not access_token or access_token == 'your_access_token_here':
            print("DHAN_ACCESS_TOKEN not set in .env file")
            print("Please update .env file with your actual Dhan access token")
            return False
        
        print(f"Found credentials:")
        print(f"   Client ID: {client_id[:8]}...")
        print(f"   Access Token: {access_token[:16]}...")
        
        # Create credentials
        credentials = DhanCredentials(
            client_id=client_id,
            access_token=access_token
        )
        
        # Initialize client
        print("\nInitializing Dhan API client...")
        client = DhanAPIClient(credentials)
        
        # Test authentication
        print("Testing authentication...")
        if client.authenticate():
            print("Authentication successful!")
            
            # Get profile
            print("\nFetching profile...")
            profile = client.get_profile()
            if profile:
                print(f"Profile loaded: {profile.get('clientId', 'Unknown')}")
            
            # Get fund limit
            print("\nFetching fund limit...")
            fund_limit = client.get_fund_limit()
            if fund_limit:
                print(f"Available Balance: Rs.{fund_limit.get('available_balance', 0):,.2f}")
            
            # Get positions
            print("\nFetching positions...")
            positions = client.get_positions()
            print(f"Current positions: {len(positions)}")
            
            return True
        else:
            print("Authentication failed!")
            print("Please check your credentials and try again")
            return False
    
    except Exception as e:
        print(f"Error testing authentication: {e}")
        return False

if __name__ == "__main__":
    print("Testing Dhan API Authentication")
    print("=" * 40)
    
    success = test_dhan_auth()
    
    if success:
        print("\nAll tests passed! Your Dhan API is properly configured.")
    else:
        print("\nNext steps:")
        print("1. Get your Dhan API credentials from https://dhanhq.co/")
        print("2. Update the .env file with your actual credentials")
        print("3. Run this script again to verify")