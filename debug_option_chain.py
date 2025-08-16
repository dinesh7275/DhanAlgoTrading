#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Option Chain Data
=======================

Test script to see exactly what option chain data is being returned
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent / "AIBot"))

from dotenv import load_dotenv
from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials

load_dotenv()

def main():
    print("=" * 60)
    print("DEBUGGING OPTION CHAIN DATA")
    print("=" * 60)
    
    try:
        # Initialize client
        credentials = DhanCredentials(
            client_id=os.getenv('DHAN_CLIENT_ID', '1107321060'),
            access_token=os.getenv('DHAN_ACCESS_TOKEN')
        )
        
        client = DhanAPIClient(credentials)
        
        if not client.authenticate():
            print("Authentication failed")
            return
        
        print("Authentication successful")
        
        # Get option chain
        print("\nFetching option chain...")
        option_chain = client.get_option_chain(underlying_scrip=13)
        
        print(f"Total contracts fetched: {len(option_chain)}")
        
        if option_chain:
            print("\nSample of first 5 contracts:")
            for i, option in enumerate(option_chain[:5]):
                print(f"\nContract {i+1}:")
                print(f"  Symbol: {option.symbol}")
                print(f"  Strike: {option.strike_price}")
                print(f"  Type: {option.option_type}")
                print(f"  LTP: {option.ltp}")
                print(f"  Volume: {option.volume}")
                print(f"  OI: {option.open_interest}")
                print(f"  IV: {option.iv}")
                print(f"  Bid: {option.bid_price}")
                print(f"  Ask: {option.ask_price}")
                print(f"  Delta: {option.delta}")
            
            # Test the processing function
            print("\nTesting data processing...")
            processed = process_option_chain_debug(option_chain, 24616)
            
            print(f"Processed contracts: {len(processed)}")
            
            if processed:
                print("\nSample of processed data:")
                for i, contract in enumerate(processed[:3]):
                    print(f"\nProcessed Contract {i+1}:")
                    for key, value in contract.items():
                        print(f"  {key}: {value}")
        
        else:
            print("No option chain data received")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def process_option_chain_debug(raw_data, nifty_price):
    """Debug version of option chain processing"""
    print(f"\nProcessing {len(raw_data)} contracts for NIFTY price {nifty_price}")
    
    processed = []
    
    try:
        # Group options by strike price
        strike_groups = {}
        
        for option in raw_data:
            strike = option.strike_price
            if strike not in strike_groups:
                strike_groups[strike] = {'CE': None, 'PE': None}
            
            if option.option_type == 'CE':
                strike_groups[strike]['CE'] = option
            elif option.option_type == 'PE':
                strike_groups[strike]['PE'] = option
        
        print(f"Found {len(strike_groups)} unique strikes")
        
        # Find strikes around current NIFTY price
        target_strikes = []
        for strike in sorted(strike_groups.keys()):
            if abs(strike - nifty_price) <= 500:  # Within 500 points
                target_strikes.append(strike)
        
        print(f"Target strikes (within 500 points): {len(target_strikes)}")
        print(f"   Strikes: {sorted(target_strikes)}")
        
        # Create combined rows for display
        for strike in sorted(target_strikes):
            ce_option = strike_groups[strike].get('CE')
            pe_option = strike_groups[strike].get('PE')
            
            if ce_option or pe_option:
                contract = {
                    'strike_price': strike,
                    'option_type': 'COMBINED',
                    'ltp': ce_option.ltp if ce_option and ce_option.ltp > 0 else (pe_option.ltp if pe_option else 0),
                    'volume': ce_option.volume if ce_option else (pe_option.volume if pe_option else 0),
                    'open_interest': ce_option.open_interest if ce_option else (pe_option.open_interest if pe_option else 0),
                    'iv': ce_option.iv if ce_option else (pe_option.iv if pe_option else 0),
                    'ce_ltp': ce_option.ltp if ce_option else 0,
                    'ce_volume': ce_option.volume if ce_option else 0,
                    'ce_oi': ce_option.open_interest if ce_option else 0,
                    'pe_ltp': pe_option.ltp if pe_option else 0,
                    'pe_volume': pe_option.volume if pe_option else 0,
                    'pe_oi': pe_option.open_interest if pe_option else 0
                }
                
                processed.append(contract)
                
                print(f"Strike {strike}: CE_LTP={contract['ce_ltp']}, PE_LTP={contract['pe_ltp']}, CE_Vol={contract['ce_volume']}, PE_Vol={contract['pe_volume']}")
        
        return processed[:15]  # Top 15 strikes
    
    except Exception as e:
        print(f"Error processing option chain: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()