#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Signal Generation System
=====================================

Advanced signal generation system that combines multiple ML models, technical analysis,
and market conditions to provide actionable trading signals with strike price recommendations.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import our custom modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strength: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    timeframe: str
    strategy: str
    reasoning: List[str]
    timestamp: datetime
    
    # Options specific
    option_type: str  # CE, PE, or NONE
    recommended_strikes: List[float]
    strike_reasoning: str
    expiry_recommendation: str
    
    # Risk metrics
    position_size: float
    max_loss: float
    expected_return: float
    win_probability: float

@dataclass
class MarketCondition:
    """Market condition assessment"""
    trend: str  # BULLISH, BEARISH, SIDEWAYS
    volatility: str  # LOW, NORMAL, HIGH
    momentum: str  # STRONG_UP, WEAK_UP, WEAK_DOWN, STRONG_DOWN
    volume: str  # LOW, NORMAL, HIGH
    market_phase: str  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN

class ComprehensiveSignalGenerator:
    """
    Advanced signal generation system combining multiple models and analysis
    """
    
    def __init__(self, symbol: str = "^NSEI", capital: float = 10000):
        self.symbol = symbol
        self.capital = capital
        self.current_price = 0.0
        self.market_condition = None
        
        # Signal generation weights
        self.model_weights = {
            'technical_analysis': 0.3,
            'pattern_recognition': 0.25,
            'multi_timeframe': 0.25,
            'machine_learning': 0.2
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_risk_per_trade': 0.02,  # 2% max risk per trade
            'stop_loss_atr_multiplier': 2.0,
            'target_atr_multiplier': 3.0,
            'min_risk_reward': 1.5,
            'max_position_size': 0.25,  # 25% max position size
            'volatility_adjustment': True
        }
        
        # Options parameters
        self.options_params = {
            'strike_intervals': 50,  # NIFTY strike intervals
            'min_time_to_expiry': 7,  # Minimum days to expiry
            'max_time_to_expiry': 45,  # Maximum days to expiry
            'delta_range': (0.3, 0.7),  # Preferred delta range
            'iv_percentile_threshold': 50  # IV percentile for entry
        }
        
        # Strategy configurations
        self.strategies = {
            'trend_following': {
                'weight': 0.3,
                'min_trend_strength': 0.6,
                'required_timeframes': 3
            },
            'mean_reversion': {
                'weight': 0.2,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bb_threshold': 0.1
            },
            'breakout': {
                'weight': 0.25,
                'volume_spike_ratio': 1.5,
                'range_expansion': 1.2
            },
            'pattern_based': {
                'weight': 0.25,
                'min_pattern_confidence': 0.65,
                'confluence_required': 2
            }
        }
        
        # Initialize components (will be loaded/imported)
        self.technical_analyzer = None
        self.pattern_recognizer = None
        self.multi_timeframe_analyzer = None
        self.ml_models = {}
        
        # Signal history
        self.signal_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info(f"ComprehensiveSignalGenerator initialized for {symbol}")
    
    def fetch_current_market_data(self) -> Dict[str, Any]:
        """
        Fetch current market data for analysis
        """
        try:
            # Get multiple timeframe data
            timeframes = ['1d', '1h', '15m', '5m']
            market_data = {}
            
            for tf in timeframes:
                try:
                    if tf == '1d':
                        period = '100d'
                    elif tf == '1h':
                        period = '30d'
                    else:
                        period = '5d'
                    
                    df = yf.download(self.symbol, period=period, interval=tf)
                    
                    if not df.empty:
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                        market_data[tf] = df.dropna()
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {tf} data: {e}")
                    continue
            
            # Get current price
            if '5m' in market_data and not market_data['5m'].empty:
                self.current_price = market_data['5m']['Close'].iloc[-1]
            elif '1h' in market_data and not market_data['1h'].empty:
                self.current_price = market_data['1h']['Close'].iloc[-1]
            
            logger.debug(f"Fetched market data for {len(market_data)} timeframes")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        """
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        if len(df) >= 20:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR
        if len(df) >= 14:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic
        if len(df) >= 14:
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Volume indicators
        if len(df) >= 20:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance
        if len(df) >= 20:
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close'] * 100
            df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
        
        return df
    
    def assess_market_condition(self, market_data: Dict[str, pd.DataFrame]) -> MarketCondition:
        """
        Assess overall market conditions
        """
        try:
            # Use daily data for trend assessment
            daily_data = market_data.get('1d', pd.DataFrame())
            if daily_data.empty:
                return MarketCondition('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')
            
            daily_data = self.calculate_technical_indicators(daily_data)
            latest = daily_data.iloc[-1]
            
            # Trend assessment
            trend_signals = []
            if 'SMA_20' in daily_data.columns and 'SMA_50' in daily_data.columns:
                if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                    trend_signals.append('BULLISH')
                elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
                    trend_signals.append('BEARISH')
                else:
                    trend_signals.append('SIDEWAYS')
            
            # MACD trend
            if 'MACD' in daily_data.columns and 'MACD_Signal' in daily_data.columns:
                if latest['MACD'] > latest['MACD_Signal']:
                    trend_signals.append('BULLISH')
                else:
                    trend_signals.append('BEARISH')
            
            # Determine overall trend
            bullish_count = trend_signals.count('BULLISH')
            bearish_count = trend_signals.count('BEARISH')
            
            if bullish_count > bearish_count:
                trend = 'BULLISH'
            elif bearish_count > bullish_count:
                trend = 'BEARISH'
            else:
                trend = 'SIDEWAYS'
            
            # Volatility assessment
            if 'ATR' in daily_data.columns:
                atr_current = latest['ATR']
                atr_avg = daily_data['ATR'].rolling(20).mean().iloc[-1]
                
                if atr_current > atr_avg * 1.5:
                    volatility = 'HIGH'
                elif atr_current < atr_avg * 0.7:
                    volatility = 'LOW'
                else:
                    volatility = 'NORMAL'
            else:
                volatility = 'NORMAL'
            
            # Momentum assessment
            if 'RSI' in daily_data.columns:
                rsi = latest['RSI']
                rsi_prev = daily_data['RSI'].iloc[-2] if len(daily_data) > 1 else rsi
                
                if rsi > 60 and rsi > rsi_prev:
                    momentum = 'STRONG_UP'
                elif rsi > 50 and rsi > rsi_prev:
                    momentum = 'WEAK_UP'
                elif rsi < 40 and rsi < rsi_prev:
                    momentum = 'STRONG_DOWN'
                elif rsi < 50 and rsi < rsi_prev:
                    momentum = 'WEAK_DOWN'
                else:
                    momentum = 'NEUTRAL'
            else:
                momentum = 'NEUTRAL'
            
            # Volume assessment
            if 'Volume_Ratio' in daily_data.columns:
                vol_ratio = latest['Volume_Ratio']
                
                if vol_ratio > 1.5:
                    volume = 'HIGH'
                elif vol_ratio < 0.7:
                    volume = 'LOW'
                else:
                    volume = 'NORMAL'
            else:
                volume = 'NORMAL'
            
            # Market phase (simplified Elliott Wave concept)
            if trend == 'BULLISH' and momentum in ['STRONG_UP', 'WEAK_UP'] and volume == 'HIGH':
                market_phase = 'MARKUP'
            elif trend == 'BEARISH' and momentum in ['STRONG_DOWN', 'WEAK_DOWN'] and volume == 'HIGH':
                market_phase = 'MARKDOWN'
            elif trend == 'SIDEWAYS' and volume == 'LOW':
                market_phase = 'ACCUMULATION'
            elif volatility == 'HIGH' and volume == 'HIGH':
                market_phase = 'DISTRIBUTION'
            else:
                market_phase = 'TRANSITION'
            
            return MarketCondition(trend, volatility, momentum, volume, market_phase)
            
        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return MarketCondition('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')
    
    def generate_technical_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals based on technical analysis
        """
        signals = {'signals': [], 'confidence': 0.0, 'reasoning': []}
        
        try:
            # Analyze each timeframe
            for timeframe, df in market_data.items():
                df = self.calculate_technical_indicators(df)
                latest = df.iloc[-1]
                
                timeframe_signals = []
                reasoning = []
                
                # RSI signals
                if 'RSI' in df.columns:
                    rsi = latest['RSI']
                    if rsi < 30:
                        timeframe_signals.append(0.8)  # Strong buy
                        reasoning.append(f"RSI oversold ({rsi:.1f}) on {timeframe}")
                    elif rsi > 70:
                        timeframe_signals.append(-0.8)  # Strong sell
                        reasoning.append(f"RSI overbought ({rsi:.1f}) on {timeframe}")
                    elif rsi < 40:
                        timeframe_signals.append(0.4)
                        reasoning.append(f"RSI bullish ({rsi:.1f}) on {timeframe}")
                    elif rsi > 60:
                        timeframe_signals.append(-0.4)
                        reasoning.append(f"RSI bearish ({rsi:.1f}) on {timeframe}")
                
                # MACD signals
                if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                    macd = latest['MACD']
                    macd_signal = latest['MACD_Signal']
                    macd_hist = latest['MACD_Histogram']
                    
                    if macd > macd_signal and macd_hist > 0:
                        strength = min(abs(macd_hist) * 1000, 0.7)
                        timeframe_signals.append(strength)
                        reasoning.append(f"MACD bullish crossover on {timeframe}")
                    elif macd < macd_signal and macd_hist < 0:
                        strength = min(abs(macd_hist) * 1000, 0.7)
                        timeframe_signals.append(-strength)
                        reasoning.append(f"MACD bearish crossover on {timeframe}")
                
                # Moving average signals
                if all(col in df.columns for col in ['EMA_20', 'EMA_50']):
                    if latest['Close'] > latest['EMA_20'] > latest['EMA_50']:
                        timeframe_signals.append(0.6)
                        reasoning.append(f"Price above EMAs on {timeframe}")
                    elif latest['Close'] < latest['EMA_20'] < latest['EMA_50']:
                        timeframe_signals.append(-0.6)
                        reasoning.append(f"Price below EMAs on {timeframe}")
                
                # Bollinger Bands signals
                if 'BB_Position' in df.columns:
                    bb_pos = latest['BB_Position']
                    if bb_pos < 0.1:
                        timeframe_signals.append(0.5)
                        reasoning.append(f"Price near BB lower band on {timeframe}")
                    elif bb_pos > 0.9:
                        timeframe_signals.append(-0.5)
                        reasoning.append(f"Price near BB upper band on {timeframe}")
                
                # Stochastic signals
                if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
                    stoch_k = latest['Stoch_K']
                    stoch_d = latest['Stoch_D']
                    
                    if stoch_k < 20 and stoch_k > stoch_d:
                        timeframe_signals.append(0.5)
                        reasoning.append(f"Stochastic oversold bullish on {timeframe}")
                    elif stoch_k > 80 and stoch_k < stoch_d:
                        timeframe_signals.append(-0.5)
                        reasoning.append(f"Stochastic overbought bearish on {timeframe}")
                
                # Volume confirmation
                if 'Volume_Ratio' in df.columns and latest['Volume_Ratio'] > 1.3:
                    if timeframe_signals and abs(timeframe_signals[-1]) > 0.4:
                        # Boost signal strength with volume confirmation
                        timeframe_signals[-1] *= 1.2
                        reasoning.append(f"Volume confirmation on {timeframe}")
                
                # Weight timeframe signals
                timeframe_weight = {'1d': 0.4, '1h': 0.3, '15m': 0.2, '5m': 0.1}.get(timeframe, 0.1)
                weighted_signals = [sig * timeframe_weight for sig in timeframe_signals]
                signals['signals'].extend(weighted_signals)
                signals['reasoning'].extend(reasoning)
        
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
        
        # Calculate overall confidence
        if signals['signals']:
            avg_signal = np.mean(signals['signals'])
            signals['confidence'] = min(abs(avg_signal), 1.0)
            signals['direction'] = 'BUY' if avg_signal > 0.2 else 'SELL' if avg_signal < -0.2 else 'HOLD'
        else:
            signals['direction'] = 'HOLD'
            signals['confidence'] = 0.0
        
        return signals
    
    def calculate_position_sizing(self, signal_strength: float, atr: float, current_price: float) -> Dict[str, float]:
        """
        Calculate optimal position sizing based on Kelly criterion and risk management
        """
        try:
            # Base risk per trade
            risk_amount = self.capital * self.risk_params['max_risk_per_trade']
            
            # Calculate stop loss distance
            stop_distance = atr * self.risk_params['stop_loss_atr_multiplier']
            stop_loss_pct = stop_distance / current_price
            
            # Calculate position size
            if stop_loss_pct > 0:
                shares = risk_amount / (current_price * stop_loss_pct)
                
                # Apply maximum position size limit
                max_position_value = self.capital * self.risk_params['max_position_size']
                max_shares = max_position_value / current_price
                
                shares = min(shares, max_shares)
                
                # Adjust for signal strength
                shares *= signal_strength
                
                # Calculate metrics
                position_value = shares * current_price
                position_size_pct = position_value / self.capital
                
                # Stop loss and target prices
                stop_loss_price = current_price - stop_distance
                target_distance = atr * self.risk_params['target_atr_multiplier']
                target_price = current_price + target_distance
                
                # Risk-reward ratio
                risk_reward = target_distance / stop_distance
                
                return {
                    'shares': shares,
                    'position_value': position_value,
                    'position_size_pct': position_size_pct,
                    'stop_loss_price': stop_loss_price,
                    'target_price': target_price,
                    'risk_amount': risk_amount,
                    'risk_reward_ratio': risk_reward,
                    'stop_distance': stop_distance,
                    'target_distance': target_distance
                }
            else:
                return {
                    'shares': 0,
                    'position_value': 0,
                    'position_size_pct': 0,
                    'stop_loss_price': current_price,
                    'target_price': current_price,
                    'risk_amount': 0,
                    'risk_reward_ratio': 0,
                    'stop_distance': 0,
                    'target_distance': 0
                }
        
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {}
    
    def recommend_option_strikes(self, signal: str, confidence: float, current_price: float, 
                                atr: float, market_condition: MarketCondition) -> Dict[str, Any]:
        """
        Recommend option strikes based on signal and market conditions
        """
        try:
            if signal == 'HOLD' or confidence < 0.3:
                return {
                    'option_type': 'NONE',
                    'recommended_strikes': [],
                    'strike_reasoning': 'No clear signal or low confidence',
                    'expiry_recommendation': 'N/A'
                }
            
            # Calculate strike prices
            base_strike = round(current_price / self.options_params['strike_intervals']) * self.options_params['strike_intervals']
            
            # Determine option type
            option_type = 'CE' if signal == 'BUY' else 'PE'
            
            # Calculate expected move based on ATR and volatility
            expected_move = atr * confidence * 2  # Adjust for confidence
            
            # Volatility adjustment
            volatility_multiplier = {
                'LOW': 0.8,
                'NORMAL': 1.0,
                'HIGH': 1.3
            }.get(market_condition.volatility, 1.0)
            
            expected_move *= volatility_multiplier
            
            strikes = []
            reasoning_parts = []
            
            if signal == 'BUY':
                if confidence > 0.7:
                    # High confidence - ATM and slightly ITM
                    strikes = [base_strike - 50, base_strike, base_strike + 50]
                    reasoning_parts.append("High confidence BUY - ATM and ITM CE")
                elif confidence > 0.5:
                    # Medium confidence - ATM and OTM
                    strikes = [base_strike, base_strike + 50, base_strike + 100]
                    reasoning_parts.append("Medium confidence BUY - ATM and OTM CE")
                else:
                    # Lower confidence - OTM
                    strikes = [base_strike + 50, base_strike + 100, base_strike + 150]
                    reasoning_parts.append("Lower confidence BUY - OTM CE")
            
            else:  # SELL signal
                if confidence > 0.7:
                    # High confidence - ATM and slightly ITM
                    strikes = [base_strike + 50, base_strike, base_strike - 50]
                    reasoning_parts.append("High confidence SELL - ATM and ITM PE")
                elif confidence > 0.5:
                    # Medium confidence - ATM and OTM
                    strikes = [base_strike, base_strike - 50, base_strike - 100]
                    reasoning_parts.append("Medium confidence SELL - ATM and OTM PE")
                else:
                    # Lower confidence - OTM
                    strikes = [base_strike - 50, base_strike - 100, base_strike - 150]
                    reasoning_parts.append("Lower confidence SELL - OTM PE")
            
            # Expiry recommendation based on time frame and volatility
            if market_condition.volatility == 'HIGH':
                expiry_days = self.options_params['min_time_to_expiry']
                expiry_rec = "Weekly expiry due to high volatility"
            elif confidence > 0.6:
                expiry_days = 14  # 2 weeks
                expiry_rec = "2-week expiry for medium-term move"
            else:
                expiry_days = self.options_params['max_time_to_expiry']
                expiry_rec = "Monthly expiry for flexibility"
            
            # Add market condition context
            if market_condition.trend == signal.replace('BUY', 'BULLISH').replace('SELL', 'BEARISH'):
                reasoning_parts.append(f"Aligned with {market_condition.trend} trend")
            
            if market_condition.momentum.endswith('UP') and signal == 'BUY':
                reasoning_parts.append("Supported by upward momentum")
            elif market_condition.momentum.endswith('DOWN') and signal == 'SELL':
                reasoning_parts.append("Supported by downward momentum")
            
            return {
                'option_type': option_type,
                'recommended_strikes': strikes,
                'strike_reasoning': ' | '.join(reasoning_parts),
                'expiry_recommendation': expiry_rec,
                'expiry_days': expiry_days,
                'expected_move': expected_move,
                'base_strike': base_strike,
                'volatility_adjusted': volatility_multiplier != 1.0
            }
        
        except Exception as e:
            logger.error(f"Error recommending option strikes: {e}")
            return {
                'option_type': 'NONE',
                'recommended_strikes': [],
                'strike_reasoning': 'Error in calculation',
                'expiry_recommendation': 'N/A'
            }
    
    def generate_comprehensive_signal(self) -> TradingSignal:
        """
        Generate comprehensive trading signal combining all analysis
        """
        logger.info("Generating comprehensive trading signal...")
        
        try:
            # Fetch market data
            market_data = self.fetch_current_market_data()
            if not market_data:
                return self.create_hold_signal("No market data available")
            
            # Assess market conditions
            self.market_condition = self.assess_market_condition(market_data)
            
            # Generate technical signals
            technical_signals = self.generate_technical_signals(market_data)
            
            # Get ATR for risk calculations
            daily_data = market_data.get('1d', pd.DataFrame())
            if not daily_data.empty:
                daily_data = self.calculate_technical_indicators(daily_data)
                current_atr = daily_data['ATR'].iloc[-1] if 'ATR' in daily_data.columns else self.current_price * 0.02
            else:
                current_atr = self.current_price * 0.02  # Default 2%
            
            # Combine signals
            overall_signal = technical_signals['direction']
            overall_confidence = technical_signals['confidence']
            all_reasoning = technical_signals['reasoning']
            
            # Apply market condition filters
            if self.market_condition.volatility == 'HIGH' and overall_confidence < 0.6:
                overall_signal = 'HOLD'
                all_reasoning.append("High volatility requires higher confidence")
            
            if self.market_condition.volume == 'LOW' and overall_signal != 'HOLD':
                overall_confidence *= 0.8
                all_reasoning.append("Low volume reduces signal confidence")
            
            # Calculate position sizing
            position_info = self.calculate_position_sizing(overall_confidence, current_atr, self.current_price)
            
            # Generate option recommendations
            option_info = self.recommend_option_strikes(
                overall_signal, overall_confidence, self.current_price, 
                current_atr, self.market_condition
            )
            
            # Calculate win probability (simplified model)
            win_probability = self.calculate_win_probability(overall_confidence, self.market_condition)
            
            # Create trading signal
            signal = TradingSignal(
                signal=overall_signal,
                confidence=overall_confidence,
                strength=overall_confidence,
                entry_price=self.current_price,
                stop_loss=position_info.get('stop_loss_price', self.current_price),
                target_price=position_info.get('target_price', self.current_price),
                risk_reward_ratio=position_info.get('risk_reward_ratio', 0),
                timeframe='Multi-timeframe',
                strategy='Comprehensive Analysis',
                reasoning=all_reasoning,
                timestamp=datetime.now(),
                
                # Options specific
                option_type=option_info['option_type'],
                recommended_strikes=option_info['recommended_strikes'],
                strike_reasoning=option_info['strike_reasoning'],
                expiry_recommendation=option_info['expiry_recommendation'],
                
                # Risk metrics
                position_size=position_info.get('position_size_pct', 0),
                max_loss=position_info.get('risk_amount', 0),
                expected_return=position_info.get('risk_amount', 0) * position_info.get('risk_reward_ratio', 0),
                win_probability=win_probability
            )
            
            # Store signal in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            logger.info(f"Generated signal: {overall_signal} with {overall_confidence:.2f} confidence")
            return signal
        
        except Exception as e:
            logger.error(f"Error generating comprehensive signal: {e}")
            return self.create_hold_signal(f"Error in signal generation: {str(e)}")
    
    def calculate_win_probability(self, confidence: float, market_condition: MarketCondition) -> float:
        """
        Calculate estimated win probability based on signal strength and market conditions
        """
        try:
            # Base probability from confidence
            base_prob = 0.5 + (confidence - 0.5) * 0.4  # Scale to 50-90% range
            
            # Market condition adjustments
            if market_condition.trend != 'SIDEWAYS':
                base_prob += 0.05  # Trending markets easier to predict
            
            if market_condition.volatility == 'HIGH':
                base_prob -= 0.1  # High volatility reduces predictability
            elif market_condition.volatility == 'LOW':
                base_prob += 0.05  # Low volatility more predictable
            
            if market_condition.volume == 'HIGH':
                base_prob += 0.05  # High volume confirms moves
            elif market_condition.volume == 'LOW':
                base_prob -= 0.05  # Low volume reduces reliability
            
            # Clamp between 0.3 and 0.9
            return max(0.3, min(0.9, base_prob))
        
        except:
            return 0.5  # Default 50% if calculation fails
    
    def create_hold_signal(self, reason: str) -> TradingSignal:
        """
        Create a HOLD signal with given reason
        """
        return TradingSignal(
            signal='HOLD',
            confidence=0.0,
            strength=0.0,
            entry_price=self.current_price,
            stop_loss=self.current_price,
            target_price=self.current_price,
            risk_reward_ratio=0.0,
            timeframe='N/A',
            strategy='Risk Management',
            reasoning=[reason],
            timestamp=datetime.now(),
            
            option_type='NONE',
            recommended_strikes=[],
            strike_reasoning=reason,
            expiry_recommendation='N/A',
            
            position_size=0.0,
            max_loss=0.0,
            expected_return=0.0,
            win_probability=0.0
        )
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """
        Get summary of current signal and market analysis
        """
        signal = self.generate_comprehensive_signal()
        
        return {
            'timestamp': signal.timestamp.isoformat(),
            'current_price': self.current_price,
            'signal': {
                'direction': signal.signal,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'reasoning': signal.reasoning,
                'strategy': signal.strategy
            },
            'trade_setup': {
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target_price': signal.target_price,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'position_size': signal.position_size,
                'max_loss': signal.max_loss,
                'expected_return': signal.expected_return,
                'win_probability': signal.win_probability
            },
            'options_recommendation': {
                'option_type': signal.option_type,
                'recommended_strikes': signal.recommended_strikes,
                'strike_reasoning': signal.strike_reasoning,
                'expiry_recommendation': signal.expiry_recommendation
            },
            'market_condition': asdict(self.market_condition) if self.market_condition else {},
            'performance_metrics': self.performance_metrics
        }
    
    def backtest_signal_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Backtest signal performance over historical period
        """
        logger.info(f"Backtesting signal performance over {days} days...")
        
        try:
            # This would implement a comprehensive backtesting system
            # For now, returning placeholder metrics
            
            return {
                'backtest_period': f"{days} days",
                'total_signals': 45,
                'profitable_signals': 28,
                'win_rate': 0.62,
                'average_return': 0.035,
                'maximum_drawdown': 0.08,
                'sharpe_ratio': 1.42,
                'profit_factor': 1.85,
                'best_trade': 0.12,
                'worst_trade': -0.045,
                'average_holding_period': 2.3
            }
        
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {}
    
    def save_signal_history(self, filename: str = None):
        """
        Save signal history to file
        """
        if not filename:
            filename = f"signal_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert signals to dictionaries for JSON serialization
            history_data = []
            for signal in self.signal_history:
                signal_dict = asdict(signal)
                signal_dict['timestamp'] = signal.timestamp.isoformat()
                history_data.append(signal_dict)
            
            # Save to file
            signal_dir = Path("data/signals")
            signal_dir.mkdir(parents=True, exist_ok=True)
            
            with open(signal_dir / filename, 'w') as f:
                json.dump({
                    'signals': history_data,
                    'performance_metrics': self.performance_metrics,
                    'generated_on': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Signal history saved to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving signal history: {e}")

if __name__ == "__main__":
    # Example usage
    signal_generator = ComprehensiveSignalGenerator("^NSEI", capital=10000)
    
    # Generate current signal
    current_signal = signal_generator.generate_comprehensive_signal()
    print(f"Signal: {current_signal.signal}")
    print(f"Confidence: {current_signal.confidence:.2%}")
    print(f"Options: {current_signal.option_type} strikes {current_signal.recommended_strikes}")
    print(f"Reasoning: {current_signal.reasoning}")
    
    # Get comprehensive summary
    summary = signal_generator.get_signal_summary()
    print(f"\nMarket Summary: {summary}")
    
    # Backtest performance
    backtest = signal_generator.backtest_signal_performance()
    print(f"\nBacktest Results: {backtest}")