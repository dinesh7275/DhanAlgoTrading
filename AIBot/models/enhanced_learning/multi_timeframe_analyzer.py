#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Analysis System
===============================

Advanced system for analyzing multiple timeframes simultaneously to generate
comprehensive trading signals with confluence analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TimeframeSignal:
    """Data class for timeframe-specific signals"""
    timeframe: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    indicators: Dict[str, float]
    strength: float
    timestamp: datetime

class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to provide confluence-based trading signals
    """
    
    def __init__(self, symbol: str = "^NSEI"):
        self.symbol = symbol
        self.timeframes = {
            '1m': {'interval': '1m', 'period': '1d', 'weight': 0.1},
            '5m': {'interval': '5m', 'period': '5d', 'weight': 0.15},
            '15m': {'interval': '15m', 'period': '5d', 'weight': 0.2},
            '1h': {'interval': '1h', 'period': '30d', 'weight': 0.25},
            '1d': {'interval': '1d', 'period': '90d', 'weight': 0.3}
        }
        
        self.current_signals = {}
        self.signal_history = []
        self.indicator_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_strength': 0.001,
            'bb_squeeze_threshold': 0.02,
            'volume_spike_ratio': 2.0
        }
        
        self.confluence_weights = {
            'trend_alignment': 0.3,
            'momentum_confluence': 0.25,
            'volume_confirmation': 0.2,
            'support_resistance': 0.15,
            'pattern_strength': 0.1
        }
        
        logger.info(f"MultiTimeframeAnalyzer initialized for {symbol}")
    
    def fetch_timeframe_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific timeframe
        """
        try:
            config = self.timeframes[timeframe]
            
            # Adjust period for intraday timeframes due to Yahoo Finance limitations
            if timeframe in ['1m', '5m']:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  # Last 7 days for intraday
                df = yf.download(self.symbol, start=start_date, end=end_date, 
                               interval=config['interval'])
            else:
                df = yf.download(self.symbol, period=config['period'], 
                               interval=config['interval'])
            
            if not df.empty:
                df = self.clean_and_prepare_data(df)
                logger.debug(f"Fetched {len(df)} records for {timeframe}")
                return df
            else:
                logger.warning(f"No data for {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data: {e}")
            return None
    
    def clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data with comprehensive technical indicators
        """
        # Ensure proper column names
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df.dropna()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Moving averages with multiple periods
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            if len(df) >= period:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI with multiple periods
        for period in [9, 14, 21]:
            if len(df) >= period * 2:
                df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
        
        # MACD with different configurations
        df = self.calculate_macd(df)
        
        # Bollinger Bands
        if len(df) >= 20:
            df = self.calculate_bollinger_bands(df)
        
        # Volume indicators
        df = self.calculate_volume_indicators(df)
        
        # Support and Resistance levels
        df = self.calculate_support_resistance(df)
        
        # Trend strength indicators
        df = self.calculate_trend_indicators(df)
        
        # Momentum indicators
        df = self.calculate_momentum_indicators(df)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD with multiple configurations"""
        # Standard MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Fast MACD (5, 13, 5) for shorter timeframes
        fast_exp1 = df['Close'].ewm(span=5).mean()
        fast_exp2 = df['Close'].ewm(span=13).mean()
        df['MACD_Fast'] = fast_exp1 - fast_exp2
        df['MACD_Fast_Signal'] = df['MACD_Fast'].ewm(span=5).mean()
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Bollinger Band squeeze
        df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(20).mean() * 0.8
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        # Volume moving averages
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period).mean()
        
        # Volume ratio and spikes
        if 'Volume_MA_20' in df.columns:
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['Volume_Spike'] = df['Volume_Ratio'] > self.indicator_thresholds['volume_spike_ratio']
        
        # On-Balance Volume
        df['OBV'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, 
                                           np.where(df['Close'] < df['Close'].shift(1), -1, 0))).cumsum()
        
        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        if len(df) < lookback:
            return df
        
        # Rolling highs and lows
        df['Resistance'] = df['High'].rolling(window=lookback).max()
        df['Support'] = df['Low'].rolling(window=lookback).min()
        
        # Distance from support/resistance
        df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
        df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close'] * 100
        
        # Pivot points (simplified)
        if len(df) >= 3:
            df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
            df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        
        return df
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend strength indicators"""
        # ADX (Average Directional Index) - simplified version
        if len(df) >= 14:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(14).mean()
        
        # Trend direction based on multiple MAs
        if all(f'EMA_{period}' in df.columns for period in [10, 20, 50]):
            df['Trend_Short'] = np.where(df['EMA_10'] > df['EMA_20'], 1, -1)
            df['Trend_Medium'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
            df['Trend_Alignment'] = df['Trend_Short'] + df['Trend_Medium']
        
        # Price position relative to key MAs
        for period in [20, 50, 200]:
            if f'SMA_{period}' in df.columns:
                df[f'Price_Above_SMA_{period}'] = df['Close'] > df[f'SMA_{period}']
        
        return df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # Rate of Change
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
        
        # Stochastic Oscillator
        if len(df) >= 14:
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        if len(df) >= 14:
            df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                              (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
        
        return df
    
    def analyze_timeframe_signals(self, timeframe: str, df: pd.DataFrame) -> TimeframeSignal:
        """
        Analyze signals for a specific timeframe
        """
        if df is None or df.empty:
            return TimeframeSignal(
                timeframe=timeframe,
                signal='HOLD',
                confidence=0.0,
                indicators={},
                strength=0.0,
                timestamp=datetime.now()
            )
        
        latest = df.iloc[-1]
        indicators = {}
        signal_scores = []
        
        # RSI Analysis
        if 'RSI_14' in df.columns and not pd.isna(latest['RSI_14']):
            rsi = latest['RSI_14']
            indicators['RSI'] = rsi
            
            if rsi < self.indicator_thresholds['rsi_oversold']:
                signal_scores.append(1.0)  # Bullish
            elif rsi > self.indicator_thresholds['rsi_overbought']:
                signal_scores.append(-1.0)  # Bearish
            else:
                signal_scores.append(0.0)  # Neutral
        
        # MACD Analysis
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            macd_hist = latest['MACD_Histogram']
            
            indicators['MACD'] = macd
            indicators['MACD_Signal'] = macd_signal
            indicators['MACD_Histogram'] = macd_hist
            
            # MACD crossover and histogram analysis
            if macd > macd_signal and macd_hist > 0:
                signal_scores.append(1.0)  # Bullish
            elif macd < macd_signal and macd_hist < 0:
                signal_scores.append(-1.0)  # Bearish
            else:
                signal_scores.append(0.0)
        
        # Moving Average Analysis
        ma_signals = []
        for short, long in [(10, 20), (20, 50)]:
            short_ma = f'EMA_{short}'
            long_ma = f'EMA_{long}'
            
            if short_ma in df.columns and long_ma in df.columns:
                if latest[short_ma] > latest[long_ma]:
                    ma_signals.append(1.0)
                else:
                    ma_signals.append(-1.0)
        
        if ma_signals:
            signal_scores.append(np.mean(ma_signals))
        
        # Bollinger Bands Analysis
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
            bb_pos = latest['BB_Position']
            indicators['BB_Position'] = bb_pos
            
            if bb_pos < 0.2:  # Near lower band
                signal_scores.append(0.8)  # Bullish
            elif bb_pos > 0.8:  # Near upper band
                signal_scores.append(-0.8)  # Bearish
            else:
                signal_scores.append(0.0)
        
        # Volume Analysis
        if 'Volume_Ratio' in df.columns and not pd.isna(latest['Volume_Ratio']):
            vol_ratio = latest['Volume_Ratio']
            indicators['Volume_Ratio'] = vol_ratio
            
            # High volume supports the signal
            volume_multiplier = min(vol_ratio / 2.0, 1.5)  # Cap at 1.5x
        else:
            volume_multiplier = 1.0
        
        # Support/Resistance Analysis
        if all(col in df.columns for col in ['Support_Distance', 'Resistance_Distance']):
            support_dist = latest['Support_Distance']
            resistance_dist = latest['Resistance_Distance']
            
            indicators['Support_Distance'] = support_dist
            indicators['Resistance_Distance'] = resistance_dist
            
            # Near support = bullish, near resistance = bearish
            if support_dist < 2.0:  # Within 2% of support
                signal_scores.append(0.6)
            elif resistance_dist < 2.0:  # Within 2% of resistance
                signal_scores.append(-0.6)
        
        # Calculate final signal
        if signal_scores:
            avg_score = np.mean(signal_scores) * volume_multiplier
            confidence = min(abs(avg_score), 1.0)
            
            if avg_score > 0.3:
                signal = 'BUY'
            elif avg_score < -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
        else:
            signal = 'HOLD'
            avg_score = 0.0
            confidence = 0.0
        
        return TimeframeSignal(
            timeframe=timeframe,
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            strength=abs(avg_score),
            timestamp=datetime.now()
        )
    
    def analyze_all_timeframes(self) -> Dict[str, TimeframeSignal]:
        """
        Analyze all timeframes concurrently
        """
        logger.info("Analyzing all timeframes...")
        
        signals = {}
        
        # Use ThreadPoolExecutor for concurrent data fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.timeframes)) as executor:
            # Submit data fetching tasks
            future_to_timeframe = {
                executor.submit(self.fetch_timeframe_data, tf): tf 
                for tf in self.timeframes.keys()
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_timeframe):
                timeframe = future_to_timeframe[future]
                try:
                    data = future.result()
                    signal = self.analyze_timeframe_signals(timeframe, data)
                    signals[timeframe] = signal
                    
                    logger.debug(f"{timeframe}: {signal.signal} (confidence: {signal.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {timeframe}: {e}")
                    signals[timeframe] = TimeframeSignal(
                        timeframe=timeframe,
                        signal='HOLD',
                        confidence=0.0,
                        indicators={},
                        strength=0.0,
                        timestamp=datetime.now()
                    )
        
        self.current_signals = signals
        return signals
    
    def get_confluence_signal(self, signals: Dict[str, TimeframeSignal]) -> Dict[str, Any]:
        """
        Generate confluence-based signal from multiple timeframes
        """
        if not signals:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'strength': 0.0,
                'confluence_score': 0.0,
                'timeframe_breakdown': {},
                'recommendation': 'No data available'
            }
        
        # Weight signals by timeframe importance
        weighted_scores = []
        signal_breakdown = {}
        
        for timeframe, signal_obj in signals.items():
            weight = self.timeframes[timeframe]['weight']
            
            # Convert signal to numeric score
            signal_score = 0.0
            if signal_obj.signal == 'BUY':
                signal_score = signal_obj.strength
            elif signal_obj.signal == 'SELL':
                signal_score = -signal_obj.strength
            
            weighted_score = signal_score * weight * signal_obj.confidence
            weighted_scores.append(weighted_score)
            
            signal_breakdown[timeframe] = {
                'signal': signal_obj.signal,
                'confidence': signal_obj.confidence,
                'strength': signal_obj.strength,
                'weight': weight,
                'weighted_score': weighted_score,
                'indicators': signal_obj.indicators
            }
        
        # Calculate confluence
        final_score = sum(weighted_scores)
        confluence_strength = abs(final_score)
        
        # Determine final signal
        if final_score > 0.2:
            final_signal = 'BUY'
        elif final_score < -0.2:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Calculate confluence score (how well timeframes align)
        buy_signals = sum(1 for s in signals.values() if s.signal == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s.signal == 'SELL')
        total_signals = len(signals)
        
        confluence_score = max(buy_signals, sell_signals) / total_signals if total_signals > 0 else 0
        
        # Generate recommendation
        recommendation = self.generate_recommendation(final_signal, confluence_strength, confluence_score, signal_breakdown)
        
        return {
            'signal': final_signal,
            'confidence': confluence_strength,
            'strength': confluence_strength,
            'confluence_score': confluence_score,
            'timeframe_breakdown': signal_breakdown,
            'recommendation': recommendation,
            'timestamp': datetime.now()
        }
    
    def generate_recommendation(self, signal: str, strength: float, confluence: float, breakdown: Dict) -> str:
        """
        Generate detailed trading recommendation
        """
        recommendations = []
        
        # Signal strength assessment
        if strength > 0.7:
            strength_desc = "Very Strong"
        elif strength > 0.5:
            strength_desc = "Strong"
        elif strength > 0.3:
            strength_desc = "Moderate"
        else:
            strength_desc = "Weak"
        
        # Confluence assessment
        if confluence > 0.8:
            confluence_desc = "Excellent alignment"
        elif confluence > 0.6:
            confluence_desc = "Good alignment"
        elif confluence > 0.4:
            confluence_desc = "Moderate alignment"
        else:
            confluence_desc = "Poor alignment"
        
        recommendations.append(f"{strength_desc} {signal} signal with {confluence_desc} across timeframes")
        
        # Timeframe-specific insights
        higher_tf_signals = [breakdown[tf]['signal'] for tf in ['1d', '1h'] if tf in breakdown]
        lower_tf_signals = [breakdown[tf]['signal'] for tf in ['15m', '5m', '1m'] if tf in breakdown]
        
        if len(set(higher_tf_signals)) == 1 and higher_tf_signals[0] != 'HOLD':
            recommendations.append(f"Higher timeframes strongly favor {higher_tf_signals[0]}")
        
        if len(set(lower_tf_signals)) == 1 and lower_tf_signals[0] != 'HOLD':
            recommendations.append(f"Lower timeframes show {lower_tf_signals[0]} momentum")
        
        # Risk assessment
        if confluence < 0.5:
            recommendations.append("⚠️ Low timeframe alignment - exercise caution")
        
        if strength < 0.3:
            recommendations.append("⚠️ Weak signal strength - consider waiting for better setup")
        
        return " | ".join(recommendations)
    
    def get_strike_price_recommendation(self, signal: str, current_price: float, confidence: float) -> Dict[str, Any]:
        """
        Recommend strike prices for options trading
        """
        if signal == 'HOLD' or confidence < 0.3:
            return {
                'recommended_strikes': [],
                'option_type': 'NONE',
                'reasoning': 'No clear signal or low confidence'
            }
        
        # Calculate strike prices based on signal strength and current price
        strikes = []
        option_type = 'CE' if signal == 'BUY' else 'PE'
        
        # Round current price to nearest 50 (typical NIFTY strike interval)
        base_strike = round(current_price / 50) * 50
        
        if signal == 'BUY':
            # For bullish signals, recommend CE strikes
            if confidence > 0.7:
                # High confidence - ATM and slightly ITM
                strikes = [base_strike - 50, base_strike, base_strike + 50]
                reasoning = "High confidence BUY - ATM and slightly ITM CE options"
            else:
                # Moderate confidence - ATM and OTM
                strikes = [base_strike, base_strike + 50, base_strike + 100]
                reasoning = "Moderate confidence BUY - ATM and OTM CE options"
        
        else:  # SELL signal
            # For bearish signals, recommend PE strikes
            if confidence > 0.7:
                # High confidence - ATM and slightly ITM
                strikes = [base_strike + 50, base_strike, base_strike - 50]
                reasoning = "High confidence SELL - ATM and slightly ITM PE options"
            else:
                # Moderate confidence - ATM and OTM
                strikes = [base_strike, base_strike - 50, base_strike - 100]
                reasoning = "Moderate confidence SELL - ATM and OTM PE options"
        
        return {
            'recommended_strikes': strikes,
            'option_type': option_type,
            'reasoning': reasoning,
            'current_price': current_price,
            'base_strike': base_strike
        }
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get complete multi-timeframe analysis with recommendations
        """
        logger.info("Performing comprehensive multi-timeframe analysis...")
        
        # Analyze all timeframes
        timeframe_signals = self.analyze_all_timeframes()
        
        # Get confluence signal
        confluence_analysis = self.get_confluence_signal(timeframe_signals)
        
        # Get current NIFTY price for strike recommendations
        try:
            current_data = yf.download(self.symbol, period='1d', interval='1m')
            current_price = current_data['Close'].iloc[-1] if not current_data.empty else 25000
        except:
            current_price = 25000  # Default fallback
        
        # Get strike price recommendations
        strike_recommendations = self.get_strike_price_recommendation(
            confluence_analysis['signal'],
            current_price,
            confluence_analysis['confidence']
        )
        
        # Compile comprehensive result
        result = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'confluence_analysis': confluence_analysis,
            'strike_recommendations': strike_recommendations,
            'timeframe_details': {tf: {
                'signal': sig.signal,
                'confidence': sig.confidence,
                'strength': sig.strength,
                'indicators': sig.indicators
            } for tf, sig in timeframe_signals.items()},
            'market_summary': self.generate_market_summary(timeframe_signals, confluence_analysis)
        }
        
        # Store in history
        self.signal_history.append(result)
        if len(self.signal_history) > 100:  # Keep last 100 analyses
            self.signal_history = self.signal_history[-100:]
        
        logger.info(f"Analysis complete: {confluence_analysis['signal']} with {confluence_analysis['confidence']:.2f} confidence")
        
        return result
    
    def generate_market_summary(self, timeframe_signals: Dict[str, TimeframeSignal], confluence: Dict[str, Any]) -> str:
        """
        Generate human-readable market summary
        """
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for signal_obj in timeframe_signals.values():
            signal_counts[signal_obj.signal] += 1
        
        total_signals = len(timeframe_signals)
        
        summary_parts = []
        summary_parts.append(f"Market Analysis: {confluence['signal']} signal")
        summary_parts.append(f"Confidence: {confluence['confidence']:.1%}")
        summary_parts.append(f"Timeframe alignment: {signal_counts['BUY']}/{total_signals} BUY, {signal_counts['SELL']}/{total_signals} SELL")
        
        if confluence['confidence'] > 0.6:
            summary_parts.append("✅ High confidence setup")
        elif confluence['confidence'] > 0.3:
            summary_parts.append("⚠️ Moderate confidence setup")
        else:
            summary_parts.append("❌ Low confidence - wait for better setup")
        
        return " | ".join(summary_parts)

if __name__ == "__main__":
    # Example usage
    analyzer = MultiTimeframeAnalyzer("^NSEI")
    analysis = analyzer.get_comprehensive_analysis()
    
    print("Multi-Timeframe Analysis Results:")
    print(f"Signal: {analysis['confluence_analysis']['signal']}")
    print(f"Confidence: {analysis['confluence_analysis']['confidence']:.2%}")
    print(f"Recommendation: {analysis['confluence_analysis']['recommendation']}")
    print(f"Strike Prices: {analysis['strike_recommendations']['recommended_strikes']}")
    print(f"Market Summary: {analysis['market_summary']}")