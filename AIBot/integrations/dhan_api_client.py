#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dhan API Integration
===================

Complete integration with Dhan API for live Indian market data,
order execution, and portfolio management for options trading.
"""

import requests
import json
import time
import logging
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DhanCredentials:
    """Dhan API credentials"""
    client_id: str
    access_token: str
    base_url: str = "https://api.dhan.co"

@dataclass
class OptionChainData:
    """Option chain data structure"""
    symbol: str
    expiry_date: str
    strike_price: float
    option_type: str  # CE or PE
    ltp: float
    bid_price: float
    ask_price: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

@dataclass
class MarketQuote:
    """Market quote data structure"""
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime

@dataclass
class OrderDetails:
    """Order details structure"""
    order_id: str
    symbol: str
    transaction_type: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, SL, SL-M
    quantity: int
    price: float
    trigger_price: float
    status: str
    filled_quantity: int
    pending_quantity: int
    timestamp: datetime

class DhanAPIClient:
    """
    Comprehensive Dhan API client for live trading
    """
    
    def __init__(self, credentials: DhanCredentials):
        self.credentials = credentials
        self.session = requests.Session()
        # Disable SSL verification for certificate issues
        self.session.verify = False
        self.ws_connection = None
        self.is_connected = False
        self.market_data_callbacks = []
        self.order_update_callbacks = []
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache for market data
        self.market_data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 5  # 5 seconds
        
        # Order tracking
        self.active_orders = {}
        self.order_history = []
        
        # Setup session headers for Dhan v2 API
        self.session.headers.update({
            'access-token': credentials.access_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info("DhanAPIClient initialized")
    
    def _rate_limit_check(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        if time_diff < self.min_request_interval:
            time.sleep(self.min_request_interval - time_diff)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None) -> Dict[str, Any]:
        """
        Make API request with error handling and rate limiting
        """
        self._rate_limit_check()
        
        url = f"{self.credentials.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise
    
    def authenticate(self) -> bool:
        """
        Authenticate with Dhan API
        """
        try:
            # Test authentication with profile endpoint
            response = self._make_request('GET', '/profile')
            
            if response.get('status') == 'success':
                self.is_connected = True
                logger.info("Successfully authenticated with Dhan API")
                return True
            else:
                logger.error("Authentication failed")
                return False
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_fund_limit(self) -> Dict[str, Any]:
        """
        Get fund limit and account balance from Dhan API
        """
        try:
            response = self._make_request('GET', '/fundlimit')
            return {
                'available_balance': response.get('availabelBalance', 0),  # Note: typo in API response
                'sod_limit': response.get('sodLimit', 0),
                'utilized_margin': response.get('utilizedAmount', 0),
                'withdrawable_balance': response.get('withdrawableBalance', 0),
                'collateral_amount': response.get('collateralAmount', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching fund limit: {e}")
            return {}
    
    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile information
        """
        try:
            response = self._make_request('GET', '/profile')
            return response.get('data', {})
        
        except Exception as e:
            logger.error(f"Error fetching profile: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get current holdings
        """
        try:
            response = self._make_request('GET', '/holdings')
            return response.get('data', [])
        
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        """
        try:
            response = self._make_request('GET', '/positions')
            return response.get('data', [])
        
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_funds(self) -> Dict[str, Any]:
        """
        Get available funds
        """
        try:
            response = self._make_request('GET', '/funds')
            return response.get('data', {})
        
        except Exception as e:
            logger.error(f"Error fetching funds: {e}")
            return {}
    
    def get_market_quote(self, security_ids: List[int], exchange_segment: str = "NSE_FNO") -> Dict[str, Any]:
        """
        Get real-time market quote using Dhan v2 API
        """
        try:
            # Dhan v2 Market Quote API format
            request_data = {
                exchange_segment: security_ids
            }
            
            response = self._make_request('POST', '/v2/marketfeed/quote', data=request_data)
            
            if not response or 'data' not in response:
                logger.warning("No market quote data received")
                return {}
            
            quotes = {}
            for security_id, quote_data in response['data'].items():
                quotes[security_id] = {
                    'ltp': quote_data.get('LTP', 0.0),
                    'open': quote_data.get('open', 0.0),
                    'high': quote_data.get('high', 0.0),
                    'low': quote_data.get('low', 0.0),
                    'close': quote_data.get('close', 0.0),
                    'volume': quote_data.get('volume', 0),
                    'change': quote_data.get('change', 0.0),
                    'change_percent': quote_data.get('changePercent', 0.0),
                    'open_interest': quote_data.get('openInterest', 0),
                    'timestamp': datetime.now()
                }
            
            return quotes
        
        except Exception as e:
            logger.error(f"Error fetching market quotes: {e}")
            return {}
    
    def get_historical_data(self, security_id: str, exchange_segment: str = "NSE_EQ", 
                           instrument: str = "EQUITY", from_date: str = None, 
                           to_date: str = None) -> pd.DataFrame:
        """
        Get historical market data
        """
        try:
            # Default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Dhan v2 Historical Data API format
            request_data = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = self._make_request('POST', '/v2/charts/historical', data=request_data)
            
            if not response or 'data' not in response:
                logger.warning("No historical data received")
                return pd.DataFrame()
            
            data = response['data']
            
            if data:
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                # Ensure proper column names match Dhan response
                column_mapping = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                return df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, underlying_scrip: int = 13, expiry: str = None) -> List[OptionChainData]:
        """
        Get complete option chain using Dhan v2 API
        """
        try:
            if expiry is None:
                # Get next weekly expiry (Thursday)
                today = datetime.now()
                days_ahead = 3 - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                expiry = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            # Dhan v2 Option Chain API format
            request_data = {
                "UnderlyingScrip": underlying_scrip,  # 13 for NIFTY
                "UnderlyingSeg": "IDX_I",  # Index segment
                "Expiry": expiry
            }
            
            response = self._make_request('POST', '/v2/optionchain', data=request_data)
            
            if not response or 'data' not in response:
                logger.warning("No option chain data received")
                return []
            
            option_chain = []
            data = response['data']
            
            # Parse Dhan option chain response
            for strike_data in data.get('optionChainDetails', []):
                strike_price = strike_data.get('strikePrice', 0)
                
                # Call (CE) option
                if 'CE' in strike_data:
                    ce_data = strike_data['CE']
                    option_chain.append(OptionChainData(
                        symbol=ce_data.get('tradingSymbol', ''),
                        expiry_date=expiry,
                        strike_price=strike_price,
                        option_type='CE',
                        ltp=ce_data.get('lastPrice', 0),
                        bid_price=ce_data.get('bidPrice', 0),
                        ask_price=ce_data.get('askPrice', 0),
                        volume=ce_data.get('volume', 0),
                        open_interest=ce_data.get('openInterest', 0),
                        iv=ce_data.get('impliedVolatility', 0),
                        delta=ce_data.get('delta', 0),
                        gamma=ce_data.get('gamma', 0),
                        theta=ce_data.get('theta', 0),
                        vega=ce_data.get('vega', 0)
                    ))
                
                # Put (PE) option
                if 'PE' in strike_data:
                    pe_data = strike_data['PE']
                    option_chain.append(OptionChainData(
                        symbol=pe_data.get('tradingSymbol', ''),
                        expiry_date=expiry,
                        strike_price=strike_price,
                        option_type='PE',
                        ltp=pe_data.get('lastPrice', 0),
                        bid_price=pe_data.get('bidPrice', 0),
                        ask_price=pe_data.get('askPrice', 0),
                        volume=pe_data.get('volume', 0),
                        open_interest=pe_data.get('openInterest', 0),
                        iv=pe_data.get('impliedVolatility', 0),
                        delta=pe_data.get('delta', 0),
                        gamma=pe_data.get('gamma', 0),
                        theta=pe_data.get('theta', 0),
                        vega=pe_data.get('vega', 0)
                    ))
            
            logger.info(f"Fetched option chain with {len(option_chain)} contracts")
            return option_chain
        
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return []
    
    def _get_security_id(self, symbol: str) -> str:
        """
        Get Dhan security ID for the given symbol
        """
        try:
            # Parse symbol to extract components
            # Format: "NIFTY 21 AUG 25000 CALL"
            parts = symbol.split()
            if len(parts) >= 5:
                underlying = parts[0]  # NIFTY
                day = parts[1]         # 21
                month = parts[2]       # AUG
                strike = parts[3]      # 25000
                option_type = parts[4] # CALL/PUT
                
                # Convert month name to number
                month_map = {
                    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
                }
                
                month_num = month_map.get(month, '01')
                year = datetime.now().year
                
                # For options, use a mapping or API call to get security ID
                # This is a simplified approach - in reality, you'd fetch from instruments API
                if underlying == "NIFTY":
                    # NIFTY security IDs range - this would come from instruments API
                    base_id = 50000  # Base for NIFTY options
                    strike_offset = int(int(strike) / 50)  # Strike price offset
                    type_offset = 1 if option_type == "CALL" else 0
                    
                    security_id = str(base_id + strike_offset + type_offset)
                    logger.info(f"Generated security ID {security_id} for {symbol}")
                    return security_id
            
            # Fallback for other symbols or incorrect format
            logger.warning(f"Could not parse symbol {symbol}, using default security ID")
            return "13"  # Default NIFTY index security ID
            
        except Exception as e:
            logger.error(f"Error getting security ID for {symbol}: {e}")
            return "13"  # Default fallback
    
    def get_instruments(self, exchange_segment: str = "NSE_FO") -> pd.DataFrame:
        """
        Get instruments list from Dhan API
        """
        try:
            response = self._make_request('GET', f'/v2/instrument/{exchange_segment}')
            
            if response and 'data' in response:
                instruments_data = response['data']
                df = pd.DataFrame(instruments_data)
                return df
            else:
                logger.warning("No instruments data received")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int,
                   order_type: str = "MARKET", price: float = 0.0,
                   trigger_price: float = 0.0, product_type: str = "INTRADAY",
                   security_id: str = None) -> str:
        """
        Place order using Dhan API v2 format
        """
        try:
            # Dhan v2 API requires specific format
            order_data = {
                "dhanClientId": self.credentials.client_id,
                "transactionType": transaction_type,  # BUY or SELL
                "exchangeSegment": "NSE_FO",  # For options
                "productType": product_type.upper(),  # INTRADAY, CNC, etc.
                "orderType": order_type,  # MARKET, LIMIT
                "validity": "DAY",
                "securityId": security_id or self._get_security_id(symbol),
                "quantity": str(quantity),
                "price": str(price) if price > 0 else "",
                "triggerPrice": str(trigger_price) if trigger_price > 0 else ""
            }
            
            # Add correlation ID for tracking
            order_data["correlationId"] = f"trade_{int(time.time())}"
            
            response = self._make_request('POST', '/v2/orders', data=order_data)
            
            if response.get('orderId'):
                order_id = response.get('orderId')
                
                # Track the order
                self.active_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'transaction_type': transaction_type,
                    'order_type': order_type,
                    'quantity': quantity,
                    'price': price,
                    'status': 'PENDING',
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Order placement failed: {error_msg}")
                raise Exception(f"Order placement failed: {error_msg}")
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def place_super_order(self, symbol: str, transaction_type: str, quantity: int,
                         price: float, target_price: float, stop_loss_price: float,
                         order_type: str = "LIMIT", product_type: str = "INTRADAY",
                         security_id: str = None) -> str:
        """
        Place super order with automatic target and stop-loss using Dhan API v2
        """
        try:
            # Dhan v2 Super Order API format
            order_data = {
                "dhanClientId": self.credentials.client_id,
                "transactionType": transaction_type,  # BUY or SELL
                "exchangeSegment": "NSE_FO",  # For options
                "productType": product_type.upper(),  # INTRADAY, CNC, etc.
                "orderType": order_type,  # LIMIT, MARKET
                "securityId": security_id or self._get_security_id(symbol),
                "quantity": quantity,
                "price": price,
                "targetPrice": target_price,
                "stopLossPrice": stop_loss_price,
                "trailingJump": 0  # Optional trailing stop
            }
            
            # Add correlation ID for tracking
            order_data["correlationId"] = f"super_trade_{int(time.time())}"
            
            response = self._make_request('POST', '/v2/super/orders', data=order_data)
            
            if response.get('orderId'):
                order_id = response.get('orderId')
                
                # Track the super order
                self.active_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'transaction_type': transaction_type,
                    'quantity': quantity,
                    'price': price,
                    'target_price': target_price,
                    'stop_loss_price': stop_loss_price,
                    'order_type': 'SUPER',
                    'timestamp': datetime.now(),
                    'status': 'PENDING'
                }
                
                logger.info(f"Super order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"Failed to place super order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing super order: {e}")
            return None
    
    def modify_order(self, order_id: str, quantity: int = None, 
                    price: float = None, order_type: str = None) -> bool:
        """
        Modify an existing order
        """
        try:
            modify_data = {'order_id': order_id}
            
            if quantity is not None:
                modify_data['quantity'] = quantity
            if price is not None:
                modify_data['price'] = price
            if order_type is not None:
                modify_data['order_type'] = order_type
            
            response = self._make_request('PUT', f'/orders/{order_id}', data=modify_data)
            
            if response.get('status') == 'success':
                logger.info(f"Order {order_id} modified successfully")
                return True
            else:
                logger.error(f"Order modification failed: {response.get('message')}")
                return False
        
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        """
        try:
            response = self._make_request('DELETE', f'/orders/{order_id}')
            
            if response.get('status') == 'success':
                # Update local tracking
                if order_id in self.active_orders:
                    self.active_orders[order_id]['status'] = 'CANCELLED'
                
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"Order cancellation failed: {response.get('message')}")
                return False
        
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order
        """
        try:
            response = self._make_request('GET', f'/orders/{order_id}')
            
            if response.get('status') == 'success':
                order_data = response.get('data', {})
                
                # Update local tracking
                if order_id in self.active_orders:
                    self.active_orders[order_id].update(order_data)
                
                return order_data
            else:
                logger.error(f"Failed to get order status: {response.get('message')}")
                return {}
        
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {}
    
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day
        """
        try:
            response = self._make_request('GET', '/orders')
            
            if response.get('status') == 'success':
                orders = response.get('data', [])
                
                # Update local tracking
                for order in orders:
                    order_id = order.get('order_id')
                    if order_id:
                        self.active_orders[order_id] = order
                
                return orders
            else:
                logger.error(f"Failed to get orders: {response.get('message')}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get all executed trades for the day
        """
        try:
            response = self._make_request('GET', '/trades')
            
            if response.get('status') == 'success':
                return response.get('data', [])
            else:
                logger.error(f"Failed to get trades: {response.get('message')}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def calculate_margin_required(self, symbol: str, transaction_type: str, 
                                 quantity: int, price: float) -> Dict[str, float]:
        """
        Calculate margin required for a trade
        """
        try:
            margin_data = {
                'symbol': symbol,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'price': price
            }
            
            response = self._make_request('POST', '/margin/calculate', data=margin_data)
            
            if response.get('status') == 'success':
                return response.get('data', {})
            else:
                logger.error(f"Failed to calculate margin: {response.get('message')}")
                return {}
        
        except Exception as e:
            logger.error(f"Error calculating margin: {e}")
            return {}
    
    def place_options_strategy_order(self, strategy_name: str, legs: List[Dict[str, Any]]) -> List[str]:
        """
        Place multi-leg options strategy order
        """
        order_ids = []
        
        try:
            for i, leg in enumerate(legs):
                try:
                    order_id = self.place_order(
                        symbol=leg['symbol'],
                        transaction_type=leg['transaction_type'],
                        quantity=leg['quantity'],
                        order_type=leg.get('order_type', 'MARKET'),
                        price=leg.get('price', 0.0),
                        trigger_price=leg.get('trigger_price', 0.0)
                    )
                    
                    order_ids.append(order_id)
                    logger.info(f"Strategy {strategy_name} leg {i+1} placed: {order_id}")
                    
                    # Small delay between legs
                    time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Failed to place leg {i+1} of strategy {strategy_name}: {e}")
                    
                    # Cancel previously placed orders on failure
                    for existing_order_id in order_ids:
                        self.cancel_order(existing_order_id)
                    
                    return []
            
            logger.info(f"Strategy {strategy_name} placed with {len(order_ids)} legs")
            return order_ids
        
        except Exception as e:
            logger.error(f"Error placing strategy order: {e}")
            return []
    
    def get_option_greeks(self, symbol: str) -> Dict[str, float]:
        """
        Get option Greeks for a specific option contract
        """
        try:
            params = {'symbol': symbol}
            response = self._make_request('GET', '/marketdata/greeks', params=params)
            
            if response.get('status') == 'success':
                return response.get('data', {})
            else:
                logger.error(f"Failed to get Greeks: {response.get('message')}")
                return {}
        
        except Exception as e:
            logger.error(f"Error getting option Greeks: {e}")
            return {}
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status
        """
        try:
            response = self._make_request('GET', '/marketdata/status')
            
            if response.get('status') == 'success':
                return response.get('data', {})
            else:
                logger.error(f"Failed to get market status: {response.get('message')}")
                return {}
        
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}
    
    def start_websocket_feed(self, symbols: List[str], callback_func=None):
        """
        Start real-time WebSocket feed for market data
        """
        # This would implement WebSocket connection for real-time data
        # Implementation depends on Dhan's WebSocket API specification
        logger.info(f"Starting WebSocket feed for {len(symbols)} symbols")
        
        # Placeholder implementation
        def mock_websocket_feed():
            while self.is_connected:
                for symbol in symbols:
                    try:
                        quote = self.get_market_quote(symbol)
                        if callback_func:
                            callback_func(quote)
                        
                        # Call registered callbacks
                        for callback in self.market_data_callbacks:
                            callback(quote)
                        
                    except Exception as e:
                        logger.error(f"Error in WebSocket feed for {symbol}: {e}")
                
                time.sleep(1)  # Update every second
        
        # Start in background thread
        feed_thread = threading.Thread(target=mock_websocket_feed, daemon=True)
        feed_thread.start()
        
        logger.info("WebSocket feed started")
    
    def add_market_data_callback(self, callback_func):
        """
        Add callback function for market data updates
        """
        self.market_data_callbacks.append(callback_func)
    
    def add_order_update_callback(self, callback_func):
        """
        Add callback function for order updates
        """
        self.order_update_callbacks.append(callback_func)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary
        """
        try:
            holdings = self.get_holdings()
            positions = self.get_positions()
            funds = self.get_funds()
            
            # Calculate totals
            total_holdings_value = sum(holding.get('market_value', 0) for holding in holdings)
            total_pnl = sum(position.get('pnl', 0) for position in positions)
            available_cash = funds.get('available_balance', 0)
            
            return {
                'total_portfolio_value': total_holdings_value + available_cash,
                'total_holdings_value': total_holdings_value,
                'available_cash': available_cash,
                'total_pnl': total_pnl,
                'day_pnl': sum(position.get('day_pnl', 0) for position in positions),
                'holdings_count': len(holdings),
                'positions_count': len(positions),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def export_trading_data(self, filename: str = None) -> str:
        """
        Export trading data to JSON file
        """
        if not filename:
            filename = f"dhan_trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            trading_data = {
                'timestamp': datetime.now().isoformat(),
                'profile': self.get_profile(),
                'holdings': self.get_holdings(),
                'positions': self.get_positions(),
                'funds': self.get_funds(),
                'orders': self.get_all_orders(),
                'trades': self.get_trades(),
                'portfolio_summary': self.get_portfolio_summary(),
                'active_orders': self.active_orders,
                'order_history': self.order_history
            }
            
            # Save to file
            data_dir = Path("data/dhan_exports")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            with open(data_dir / filename, 'w') as f:
                json.dump(trading_data, f, indent=2, default=str)
            
            logger.info(f"Trading data exported to {filename}")
            return str(data_dir / filename)
        
        except Exception as e:
            logger.error(f"Error exporting trading data: {e}")
            return ""
    
    def close_all_positions(self) -> List[str]:
        """
        Close all open positions
        """
        order_ids = []
        
        try:
            positions = self.get_positions()
            
            for position in positions:
                if position.get('quantity', 0) != 0:
                    # Determine transaction type to close position
                    current_qty = position.get('quantity', 0)
                    transaction_type = 'SELL' if current_qty > 0 else 'BUY'
                    close_qty = abs(current_qty)
                    
                    try:
                        order_id = self.place_order(
                            symbol=position.get('symbol', ''),
                            transaction_type=transaction_type,
                            quantity=close_qty,
                            order_type='MARKET'
                        )
                        
                        order_ids.append(order_id)
                        logger.info(f"Position close order placed: {order_id}")
                    
                    except Exception as e:
                        logger.error(f"Failed to close position {position.get('symbol')}: {e}")
            
            logger.info(f"Attempted to close {len(order_ids)} positions")
            return order_ids
        
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return []
    
    def disconnect(self):
        """
        Disconnect from Dhan API and cleanup
        """
        self.is_connected = False
        
        if self.ws_connection:
            # Close WebSocket connection
            pass
        
        # Clear caches
        self.market_data_cache.clear()
        self.cache_expiry.clear()
        
        logger.info("Disconnected from Dhan API")

# Utility functions for Dhan integration

def create_dhan_client_from_config() -> DhanAPIClient:
    """
    Create Dhan client from configuration
    """
    try:
        # Try to import config
        from config.config import API_KEYS
        
        credentials = DhanCredentials(
            client_id=API_KEYS.get('dhan_client_id', ''),
            access_token=API_KEYS.get('dhan_access_token', '')
        )
        
        client = DhanAPIClient(credentials)
        
        if client.authenticate():
            return client
        else:
            raise Exception("Authentication failed")
    
    except Exception as e:
        logger.error(f"Error creating Dhan client: {e}")
        raise

def get_nifty_option_symbols(expiry: str, strikes: List[float]) -> List[str]:
    """
    Generate NIFTY option symbols for given strikes and expiry
    """
    symbols = []
    
    for strike in strikes:
        # CE symbols
        ce_symbol = f"NIFTY{expiry}{int(strike)}CE"
        symbols.append(ce_symbol)
        
        # PE symbols
        pe_symbol = f"NIFTY{expiry}{int(strike)}PE"
        symbols.append(pe_symbol)
    
    return symbols

def calculate_option_premium_with_greeks(spot_price: float, strike_price: float, 
                                        time_to_expiry: float, risk_free_rate: float,
                                        volatility: float, option_type: str = 'CE') -> Dict[str, float]:
    """
    Calculate option premium and Greeks using Black-Scholes model
    """
    from scipy.stats import norm
    import math
    
    try:
        # Convert time to expiry to years
        T = time_to_expiry / 365.0
        
        # Calculate d1 and d2
        d1 = (math.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * T) / (volatility * math.sqrt(T))
        d2 = d1 - volatility * math.sqrt(T)
        
        # Calculate option premium
        if option_type.upper() == 'CE':
            premium = spot_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * T) * norm.cdf(d2)
        else:  # PE
            premium = strike_price * math.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        # Calculate Greeks
        delta = norm.cdf(d1) if option_type.upper() == 'CE' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (spot_price * volatility * math.sqrt(T))
        theta = (-spot_price * norm.pdf(d1) * volatility / (2 * math.sqrt(T)) - 
                risk_free_rate * strike_price * math.exp(-risk_free_rate * T) * 
                (norm.cdf(d2) if option_type.upper() == 'CE' else norm.cdf(-d2)))
        vega = spot_price * norm.pdf(d1) * math.sqrt(T)
        
        return {
            'premium': premium,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility change
        }
    
    except Exception as e:
        logger.error(f"Error calculating option premium: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    try:
        # Create credentials (replace with actual values)
        credentials = DhanCredentials(
            client_id="your_client_id",
            access_token="your_access_token"
        )
        
        # Initialize client
        client = DhanAPIClient(credentials)
        
        # Authenticate
        if client.authenticate():
            print("Successfully connected to Dhan API")
            
            # Get profile
            profile = client.get_profile()
            print(f"Profile: {profile}")
            
            # Get NIFTY quote
            nifty_quote = client.get_market_quote("NIFTY 50")
            print(f"NIFTY Quote: {nifty_quote}")
            
            # Get option chain
            option_chain = client.get_option_chain("NIFTY")
            print(f"Option chain has {len(option_chain)} contracts")
            
            # Export data
            export_file = client.export_trading_data()
            print(f"Data exported to: {export_file}")
        
        else:
            print("Failed to authenticate")
    
    except Exception as e:
        print(f"Error: {e}")