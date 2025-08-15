#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Indicator Monitoring System
=====================================

Advanced monitoring system that tracks technical indicators across multiple timeframes
in real-time and provides alerts when significant indicator signals occur.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import time
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import websocket
from enum import Enum
import talib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class IndicatorType(Enum):
    MOMENTUM = "MOMENTUM"
    TREND = "TREND"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"

@dataclass
class IndicatorAlert:
    """Data class for indicator alerts"""
    timestamp: datetime
    symbol: str
    timeframe: str
    indicator_name: str
    indicator_type: IndicatorType
    current_value: float
    previous_value: float
    threshold: float
    severity: AlertSeverity
    signal_type: str  # BUY, SELL, WARNING, INFO
    message: str
    confidence: float
    action_required: bool
    metadata: Dict[str, Any]

@dataclass
class IndicatorReading:
    """Current indicator reading"""
    timestamp: datetime
    symbol: str
    timeframe: str
    indicators: Dict[str, float]
    price_data: Dict[str, float]
    volume: int

@dataclass
class MonitoringRule:
    """Rule for monitoring indicators"""
    indicator_name: str
    indicator_type: IndicatorType
    timeframes: List[str]
    thresholds: Dict[str, float]
    alert_conditions: List[str]
    severity: AlertSeverity
    callback_function: Optional[Callable] = None
    enabled: bool = True

class RealTimeIndicatorMonitor:
    """
    Real-time monitoring system for technical indicators across multiple timeframes
    """
    
    def __init__(self, symbols: List[str] = None, update_interval: int = 30):
        self.symbols = symbols or ["^NSEI", "^NSEBANK"]
        self.update_interval = update_interval
        self.is_running = False
        self.is_market_hours = True
        
        # Data storage
        self.current_data = {}
        self.indicator_history = {}
        self.alerts = []
        self.alert_queue = queue.Queue()
        
        # Monitoring rules
        self.monitoring_rules = []
        self.timeframes = ['1m', '5m', '15m', '1h', '1d']
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Threading
        self.monitor_thread = None
        self.websocket_thread = None
        
        # Data directory
        self.data_dir = Path("data/monitoring")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default monitoring rules
        self._setup_default_monitoring_rules()
        
        logger.info(f"RealTimeIndicatorMonitor initialized for {len(self.symbols)} symbols")
    
    def _setup_default_monitoring_rules(self):
        """Setup default monitoring rules for common indicators"""
        
        # RSI Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="RSI",
            indicator_type=IndicatorType.MOMENTUM,
            timeframes=['5m', '15m', '1h', '1d'],
            thresholds={'oversold': 30, 'overbought': 70, 'extreme_oversold': 20, 'extreme_overbought': 80},
            alert_conditions=['oversold_crossover', 'overbought_crossover', 'divergence'],
            severity=AlertSeverity.HIGH
        ))
        
        # MACD Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="MACD",
            indicator_type=IndicatorType.MOMENTUM,
            timeframes=['15m', '1h', '1d'],
            thresholds={'signal_crossover': 0.0, 'histogram_reversal': 0.0},
            alert_conditions=['bullish_crossover', 'bearish_crossover', 'histogram_divergence'],
            severity=AlertSeverity.MEDIUM
        ))
        
        # Moving Average Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="MA_CROSSOVER",
            indicator_type=IndicatorType.TREND,
            timeframes=['1h', '1d'],
            thresholds={'ema_20_50': 0.0, 'sma_50_200': 0.0},
            alert_conditions=['golden_cross', 'death_cross', 'ema_crossover'],
            severity=AlertSeverity.MEDIUM
        ))
        
        # Bollinger Band Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="BOLLINGER_BANDS",
            indicator_type=IndicatorType.VOLATILITY,
            timeframes=['5m', '15m', '1h'],
            thresholds={'squeeze': 0.02, 'expansion': 0.05, 'band_touch': 0.01},
            alert_conditions=['squeeze_release', 'band_bounce', 'band_break'],
            severity=AlertSeverity.MEDIUM
        ))
        
        # Volume Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="VOLUME",
            indicator_type=IndicatorType.VOLUME,
            timeframes=['5m', '15m', '1h'],
            thresholds={'volume_spike': 2.0, 'volume_dry_up': 0.5},
            alert_conditions=['unusual_volume', 'volume_confirmation'],
            severity=AlertSeverity.LOW
        ))
        
        # Support/Resistance Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="SUPPORT_RESISTANCE",
            indicator_type=IndicatorType.SUPPORT_RESISTANCE,
            timeframes=['15m', '1h', '1d'],
            thresholds={'proximity': 0.005, 'break_strength': 0.01},
            alert_conditions=['level_test', 'level_break', 'false_break'],
            severity=AlertSeverity.HIGH
        ))
        
        # Stochastic Rules
        self.add_monitoring_rule(MonitoringRule(
            indicator_name="STOCHASTIC",
            indicator_type=IndicatorType.MOMENTUM,
            timeframes=['5m', '15m', '1h'],
            thresholds={'oversold': 20, 'overbought': 80},
            alert_conditions=['oversold_bullish', 'overbought_bearish'],
            severity=AlertSeverity.MEDIUM
        ))
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a new monitoring rule"""
        self.monitoring_rules.append(rule)
        logger.debug(f"Added monitoring rule for {rule.indicator_name}")
    
    def remove_monitoring_rule(self, indicator_name: str, timeframe: str = None):
        """Remove monitoring rule(s)"""
        self.monitoring_rules = [
            rule for rule in self.monitoring_rules 
            if not (rule.indicator_name == indicator_name and 
                   (timeframe is None or timeframe in rule.timeframes))
        ]
        logger.debug(f"Removed monitoring rule for {indicator_name}")
    
    def add_alert_callback(self, callback: Callable[[IndicatorAlert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def fetch_timeframe_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Fetch data for specific timeframe"""
        try:
            # Determine period string based on timeframe
            if timeframe == '1m':
                period = '5d'
                periods = min(periods, 300)  # 5 days max for 1m data
            elif timeframe == '5m':
                period = '30d'
                periods = min(periods, 500)
            elif timeframe == '15m':
                period = '60d'
            elif timeframe == '1h':
                period = '100d'
            else:  # 1d
                period = '2y'
            
            df = yf.download(symbol, period=period, interval=timeframe, progress=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # Clean and prepare data
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df.dropna()
            
            # Return last N periods
            return df.tail(periods) if len(df) > periods else df
            
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_comprehensive_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive set of technical indicators"""
        if df.empty or len(df) < 20:
            return {}
        
        indicators = {}
        
        try:
            # Basic price data
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            # RSI
            if len(close) >= 14:
                rsi = talib.RSI(close, timeperiod=14)
                indicators['RSI'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
                indicators['RSI_prev'] = rsi[-2] if len(rsi) > 1 and not np.isnan(rsi[-2]) else 50.0
            
            # MACD
            if len(close) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['MACD'] = macd[-1] if not np.isnan(macd[-1]) else 0.0
                indicators['MACD_Signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
                indicators['MACD_Histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
                
                if len(macd_hist) > 1:
                    indicators['MACD_Histogram_prev'] = macd_hist[-2] if not np.isnan(macd_hist[-2]) else 0.0
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(close) >= period:
                    sma = talib.SMA(close, timeperiod=period)
                    ema = talib.EMA(close, timeperiod=period)
                    indicators[f'SMA_{period}'] = sma[-1] if not np.isnan(sma[-1]) else close[-1]
                    indicators[f'EMA_{period}'] = ema[-1] if not np.isnan(ema[-1]) else close[-1]
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                indicators['BB_Upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1]
                indicators['BB_Middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[-1]
                indicators['BB_Lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1]
                
                # BB position and width
                bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] > 0 else 0
                bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5
                
                indicators['BB_Width'] = bb_width
                indicators['BB_Position'] = bb_position
            
            # ATR
            if len(close) >= 14:
                atr = talib.ATR(high, low, close, timeperiod=14)
                indicators['ATR'] = atr[-1] if not np.isnan(atr[-1]) else close[-1] * 0.02
            
            # Stochastic
            if len(close) >= 14:
                stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['Stoch_K'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50.0
                indicators['Stoch_D'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50.0
                
                if len(stoch_k) > 1:
                    indicators['Stoch_K_prev'] = stoch_k[-2] if not np.isnan(stoch_k[-2]) else 50.0
            
            # Williams %R
            if len(close) >= 14:
                willr = talib.WILLR(high, low, close, timeperiod=14)
                indicators['Williams_R'] = willr[-1] if not np.isnan(willr[-1]) else -50.0
            
            # CCI
            if len(close) >= 14:
                cci = talib.CCI(high, low, close, timeperiod=14)
                indicators['CCI'] = cci[-1] if not np.isnan(cci[-1]) else 0.0
            
            # ADX
            if len(close) >= 14:
                adx = talib.ADX(high, low, close, timeperiod=14)
                indicators['ADX'] = adx[-1] if not np.isnan(adx[-1]) else 25.0
            
            # Volume indicators
            if len(volume) >= 20:
                volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
                indicators['Volume_SMA'] = volume_sma[-1] if not np.isnan(volume_sma[-1]) else volume[-1]
                indicators['Volume_Ratio'] = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0
            
            # On Balance Volume
            if len(close) >= 20:
                obv = talib.OBV(close, volume.astype(float))
                indicators['OBV'] = obv[-1] if not np.isnan(obv[-1]) else 0.0
            
            # Support and Resistance levels
            if len(high) >= 20:
                resistance = np.max(high[-20:])
                support = np.min(low[-20:])
                indicators['Resistance_20'] = resistance
                indicators['Support_20'] = support
                indicators['Support_Distance'] = (close[-1] - support) / close[-1] * 100
                indicators['Resistance_Distance'] = (resistance - close[-1]) / close[-1] * 100
            
            # Current price data
            indicators['Current_Price'] = close[-1]
            indicators['Previous_Close'] = close[-2] if len(close) > 1 else close[-1]
            indicators['High_Today'] = high[-1]
            indicators['Low_Today'] = low[-1]
            indicators['Volume_Today'] = volume[-1]
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def evaluate_monitoring_rules(self, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate all monitoring rules for given data"""
        alerts = []
        
        for rule in self.monitoring_rules:
            if not rule.enabled or timeframe not in rule.timeframes:
                continue
            
            try:
                rule_alerts = self._evaluate_single_rule(rule, symbol, timeframe, indicators)
                alerts.extend(rule_alerts)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.indicator_name}: {e}")
        
        return alerts
    
    def _evaluate_single_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate a single monitoring rule"""
        alerts = []
        
        try:
            if rule.indicator_name == "RSI":
                alerts.extend(self._evaluate_rsi_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "MACD":
                alerts.extend(self._evaluate_macd_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "MA_CROSSOVER":
                alerts.extend(self._evaluate_ma_crossover_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "BOLLINGER_BANDS":
                alerts.extend(self._evaluate_bollinger_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "VOLUME":
                alerts.extend(self._evaluate_volume_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "SUPPORT_RESISTANCE":
                alerts.extend(self._evaluate_support_resistance_rule(rule, symbol, timeframe, indicators))
            elif rule.indicator_name == "STOCHASTIC":
                alerts.extend(self._evaluate_stochastic_rule(rule, symbol, timeframe, indicators))
        
        except Exception as e:
            logger.error(f"Error in rule evaluation for {rule.indicator_name}: {e}")
        
        return alerts
    
    def _evaluate_rsi_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate RSI-specific rules"""
        alerts = []
        
        if 'RSI' not in indicators or 'RSI_prev' not in indicators:
            return alerts
        
        rsi = indicators['RSI']
        rsi_prev = indicators['RSI_prev']
        
        # Oversold crossover
        if 'oversold_crossover' in rule.alert_conditions:
            if rsi_prev <= rule.thresholds['oversold'] and rsi > rule.thresholds['oversold']:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="RSI",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=rsi,
                    previous_value=rsi_prev,
                    threshold=rule.thresholds['oversold'],
                    severity=rule.severity,
                    signal_type="BUY",
                    message=f"RSI bullish crossover above {rule.thresholds['oversold']} on {timeframe}",
                    confidence=0.8,
                    action_required=True,
                    metadata={'pattern': 'oversold_recovery', 'strength': abs(rsi - rule.thresholds['oversold'])}
                ))
        
        # Overbought crossover
        if 'overbought_crossover' in rule.alert_conditions:
            if rsi_prev >= rule.thresholds['overbought'] and rsi < rule.thresholds['overbought']:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="RSI",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=rsi,
                    previous_value=rsi_prev,
                    threshold=rule.thresholds['overbought'],
                    severity=rule.severity,
                    signal_type="SELL",
                    message=f"RSI bearish crossover below {rule.thresholds['overbought']} on {timeframe}",
                    confidence=0.8,
                    action_required=True,
                    metadata={'pattern': 'overbought_reversal', 'strength': abs(rsi - rule.thresholds['overbought'])}
                ))
        
        # Extreme levels
        if rsi < rule.thresholds.get('extreme_oversold', 20):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="RSI",
                indicator_type=IndicatorType.MOMENTUM,
                current_value=rsi,
                previous_value=rsi_prev,
                threshold=rule.thresholds.get('extreme_oversold', 20),
                severity=AlertSeverity.CRITICAL,
                signal_type="BUY",
                message=f"RSI extremely oversold at {rsi:.1f} on {timeframe}",
                confidence=0.9,
                action_required=True,
                metadata={'pattern': 'extreme_oversold', 'reversal_probability': 0.85}
            ))
        
        elif rsi > rule.thresholds.get('extreme_overbought', 80):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="RSI",
                indicator_type=IndicatorType.MOMENTUM,
                current_value=rsi,
                previous_value=rsi_prev,
                threshold=rule.thresholds.get('extreme_overbought', 80),
                severity=AlertSeverity.CRITICAL,
                signal_type="SELL",
                message=f"RSI extremely overbought at {rsi:.1f} on {timeframe}",
                confidence=0.9,
                action_required=True,
                metadata={'pattern': 'extreme_overbought', 'reversal_probability': 0.85}
            ))
        
        return alerts
    
    def _evaluate_macd_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate MACD-specific rules"""
        alerts = []
        
        if not all(key in indicators for key in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            return alerts
        
        macd = indicators['MACD']
        macd_signal = indicators['MACD_Signal']
        macd_hist = indicators['MACD_Histogram']
        macd_hist_prev = indicators.get('MACD_Histogram_prev', macd_hist)
        
        # Bullish crossover
        if 'bullish_crossover' in rule.alert_conditions:
            if macd_hist_prev <= 0 and macd_hist > 0:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="MACD",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=macd_hist,
                    previous_value=macd_hist_prev,
                    threshold=0.0,
                    severity=rule.severity,
                    signal_type="BUY",
                    message=f"MACD bullish crossover on {timeframe}",
                    confidence=0.75,
                    action_required=True,
                    metadata={'macd': macd, 'signal': macd_signal, 'crossover_strength': abs(macd_hist)}
                ))
        
        # Bearish crossover
        if 'bearish_crossover' in rule.alert_conditions:
            if macd_hist_prev >= 0 and macd_hist < 0:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="MACD",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=macd_hist,
                    previous_value=macd_hist_prev,
                    threshold=0.0,
                    severity=rule.severity,
                    signal_type="SELL",
                    message=f"MACD bearish crossover on {timeframe}",
                    confidence=0.75,
                    action_required=True,
                    metadata={'macd': macd, 'signal': macd_signal, 'crossover_strength': abs(macd_hist)}
                ))
        
        return alerts
    
    def _evaluate_ma_crossover_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate Moving Average crossover rules"""
        alerts = []
        
        # EMA 20/50 crossover
        if all(key in indicators for key in ['EMA_20', 'EMA_50']):
            ema_20 = indicators['EMA_20']
            ema_50 = indicators['EMA_50']
            current_price = indicators.get('Current_Price', 0)
            
            # Golden cross pattern
            if ema_20 > ema_50 and current_price > ema_20:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="MA_CROSSOVER",
                    indicator_type=IndicatorType.TREND,
                    current_value=ema_20,
                    previous_value=ema_50,
                    threshold=0.0,
                    severity=AlertSeverity.MEDIUM,
                    signal_type="BUY",
                    message=f"EMA 20/50 golden cross pattern on {timeframe}",
                    confidence=0.7,
                    action_required=True,
                    metadata={'ema_20': ema_20, 'ema_50': ema_50, 'price': current_price}
                ))
            
            # Death cross pattern
            elif ema_20 < ema_50 and current_price < ema_20:
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="MA_CROSSOVER",
                    indicator_type=IndicatorType.TREND,
                    current_value=ema_20,
                    previous_value=ema_50,
                    threshold=0.0,
                    severity=AlertSeverity.MEDIUM,
                    signal_type="SELL",
                    message=f"EMA 20/50 death cross pattern on {timeframe}",
                    confidence=0.7,
                    action_required=True,
                    metadata={'ema_20': ema_20, 'ema_50': ema_50, 'price': current_price}
                ))
        
        return alerts
    
    def _evaluate_bollinger_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate Bollinger Bands rules"""
        alerts = []
        
        if not all(key in indicators for key in ['BB_Upper', 'BB_Lower', 'BB_Position', 'BB_Width']):
            return alerts
        
        bb_position = indicators['BB_Position']
        bb_width = indicators['BB_Width']
        current_price = indicators.get('Current_Price', 0)
        
        # Bollinger Band squeeze
        if bb_width < rule.thresholds.get('squeeze', 0.02):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="BOLLINGER_BANDS",
                indicator_type=IndicatorType.VOLATILITY,
                current_value=bb_width,
                previous_value=bb_width,
                threshold=rule.thresholds.get('squeeze', 0.02),
                severity=AlertSeverity.MEDIUM,
                signal_type="WARNING",
                message=f"Bollinger Band squeeze detected on {timeframe} - potential breakout",
                confidence=0.6,
                action_required=False,
                metadata={'bb_width': bb_width, 'bb_position': bb_position}
            ))
        
        # Band touch/bounce
        if bb_position <= 0.1:  # Near lower band
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="BOLLINGER_BANDS",
                indicator_type=IndicatorType.VOLATILITY,
                current_value=bb_position,
                previous_value=bb_position,
                threshold=0.1,
                severity=AlertSeverity.MEDIUM,
                signal_type="BUY",
                message=f"Price near Bollinger lower band on {timeframe}",
                confidence=0.65,
                action_required=True,
                metadata={'bb_position': bb_position, 'reversal_probability': 0.7}
            ))
        
        elif bb_position >= 0.9:  # Near upper band
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="BOLLINGER_BANDS",
                indicator_type=IndicatorType.VOLATILITY,
                current_value=bb_position,
                previous_value=bb_position,
                threshold=0.9,
                severity=AlertSeverity.MEDIUM,
                signal_type="SELL",
                message=f"Price near Bollinger upper band on {timeframe}",
                confidence=0.65,
                action_required=True,
                metadata={'bb_position': bb_position, 'reversal_probability': 0.7}
            ))
        
        return alerts
    
    def _evaluate_volume_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate Volume rules"""
        alerts = []
        
        if 'Volume_Ratio' not in indicators:
            return alerts
        
        volume_ratio = indicators['Volume_Ratio']
        
        # Volume spike
        if volume_ratio > rule.thresholds.get('volume_spike', 2.0):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="VOLUME",
                indicator_type=IndicatorType.VOLUME,
                current_value=volume_ratio,
                previous_value=1.0,
                threshold=rule.thresholds.get('volume_spike', 2.0),
                severity=AlertSeverity.HIGH,
                signal_type="INFO",
                message=f"Unusual volume spike {volume_ratio:.1f}x on {timeframe}",
                confidence=0.8,
                action_required=False,
                metadata={'volume_ratio': volume_ratio, 'spike_strength': volume_ratio}
            ))
        
        # Volume dry up
        elif volume_ratio < rule.thresholds.get('volume_dry_up', 0.5):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="VOLUME",
                indicator_type=IndicatorType.VOLUME,
                current_value=volume_ratio,
                previous_value=1.0,
                threshold=rule.thresholds.get('volume_dry_up', 0.5),
                severity=AlertSeverity.LOW,
                signal_type="WARNING",
                message=f"Volume dry up {volume_ratio:.2f}x on {timeframe}",
                confidence=0.6,
                action_required=False,
                metadata={'volume_ratio': volume_ratio}
            ))
        
        return alerts
    
    def _evaluate_support_resistance_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate Support/Resistance rules"""
        alerts = []
        
        if not all(key in indicators for key in ['Support_Distance', 'Resistance_Distance']):
            return alerts
        
        support_dist = indicators['Support_Distance']
        resistance_dist = indicators['Resistance_Distance']
        current_price = indicators.get('Current_Price', 0)
        
        # Near support level
        if support_dist < rule.thresholds.get('proximity', 0.5):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="SUPPORT_RESISTANCE",
                indicator_type=IndicatorType.SUPPORT_RESISTANCE,
                current_value=support_dist,
                previous_value=support_dist,
                threshold=rule.thresholds.get('proximity', 0.5),
                severity=AlertSeverity.HIGH,
                signal_type="BUY",
                message=f"Price near support level ({support_dist:.2f}%) on {timeframe}",
                confidence=0.75,
                action_required=True,
                metadata={'support_distance': support_dist, 'bounce_probability': 0.7}
            ))
        
        # Near resistance level
        if resistance_dist < rule.thresholds.get('proximity', 0.5):
            alerts.append(IndicatorAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="SUPPORT_RESISTANCE",
                indicator_type=IndicatorType.SUPPORT_RESISTANCE,
                current_value=resistance_dist,
                previous_value=resistance_dist,
                threshold=rule.thresholds.get('proximity', 0.5),
                severity=AlertSeverity.HIGH,
                signal_type="SELL",
                message=f"Price near resistance level ({resistance_dist:.2f}%) on {timeframe}",
                confidence=0.75,
                action_required=True,
                metadata={'resistance_distance': resistance_dist, 'rejection_probability': 0.7}
            ))
        
        return alerts
    
    def _evaluate_stochastic_rule(self, rule: MonitoringRule, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[IndicatorAlert]:
        """Evaluate Stochastic rules"""
        alerts = []
        
        if not all(key in indicators for key in ['Stoch_K', 'Stoch_D']):
            return alerts
        
        stoch_k = indicators['Stoch_K']
        stoch_d = indicators['Stoch_D']
        stoch_k_prev = indicators.get('Stoch_K_prev', stoch_k)
        
        # Oversold bullish
        if 'oversold_bullish' in rule.alert_conditions:
            if (stoch_k < rule.thresholds['oversold'] and stoch_k > stoch_d and 
                stoch_k_prev <= stoch_d):
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="STOCHASTIC",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=stoch_k,
                    previous_value=stoch_k_prev,
                    threshold=rule.thresholds['oversold'],
                    severity=rule.severity,
                    signal_type="BUY",
                    message=f"Stochastic oversold bullish crossover on {timeframe}",
                    confidence=0.7,
                    action_required=True,
                    metadata={'stoch_k': stoch_k, 'stoch_d': stoch_d}
                ))
        
        # Overbought bearish
        if 'overbought_bearish' in rule.alert_conditions:
            if (stoch_k > rule.thresholds['overbought'] and stoch_k < stoch_d and 
                stoch_k_prev >= stoch_d):
                alerts.append(IndicatorAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name="STOCHASTIC",
                    indicator_type=IndicatorType.MOMENTUM,
                    current_value=stoch_k,
                    previous_value=stoch_k_prev,
                    threshold=rule.thresholds['overbought'],
                    severity=rule.severity,
                    signal_type="SELL",
                    message=f"Stochastic overbought bearish crossover on {timeframe}",
                    confidence=0.7,
                    action_required=True,
                    metadata={'stoch_k': stoch_k, 'stoch_d': stoch_d}
                ))
        
        return alerts
    
    def process_symbol_timeframe(self, symbol: str, timeframe: str):
        """Process single symbol-timeframe combination"""
        try:
            # Fetch data
            df = self.fetch_timeframe_data(symbol, timeframe)
            if df.empty:
                return
            
            # Calculate indicators
            indicators = self.calculate_comprehensive_indicators(df)
            if not indicators:
                return
            
            # Store current reading
            reading = IndicatorReading(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                price_data={
                    'current': indicators.get('Current_Price', 0),
                    'high': indicators.get('High_Today', 0),
                    'low': indicators.get('Low_Today', 0),
                    'previous': indicators.get('Previous_Close', 0)
                },
                volume=int(indicators.get('Volume_Today', 0))
            )
            
            # Store in current data
            key = f"{symbol}_{timeframe}"
            self.current_data[key] = reading
            
            # Store in history
            if key not in self.indicator_history:
                self.indicator_history[key] = []
            
            self.indicator_history[key].append(reading)
            
            # Keep only last 100 readings
            if len(self.indicator_history[key]) > 100:
                self.indicator_history[key] = self.indicator_history[key][-100:]
            
            # Evaluate monitoring rules
            new_alerts = self.evaluate_monitoring_rules(symbol, timeframe, indicators)
            
            # Process alerts
            for alert in new_alerts:
                self.alerts.append(alert)
                self.alert_queue.put(alert)
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                
                logger.info(f"ALERT: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {e}")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Check if market hours
                current_time = datetime.now()
                if not self._is_market_hours(current_time):
                    logger.debug("Outside market hours, sleeping...")
                    time.sleep(60)  # Check every minute outside market hours
                    continue
                
                # Process all symbol-timeframe combinations
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        if not self.is_running:
                            break
                        
                        try:
                            self.process_symbol_timeframe(symbol, timeframe)
                        except Exception as e:
                            logger.error(f"Error processing {symbol} {timeframe}: {e}")
                        
                        # Small delay between requests to avoid rate limiting
                        time.sleep(0.5)
                    
                    if not self.is_running:
                        break
                
                # Calculate sleep time to maintain update interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    logger.debug(f"Monitoring cycle completed in {elapsed:.1f}s, sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Monitoring cycle took {elapsed:.1f}s, longer than {self.update_interval}s interval")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
        
        logger.info("Monitoring loop stopped")
    
    def _is_market_hours(self, current_time: datetime) -> bool:
        """Check if current time is within market hours"""
        if not self.is_market_hours:
            return True  # Allow monitoring outside market hours for testing
        
        # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        weekday = current_time.weekday()
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= current_time <= market_close
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time indicator monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Real-time indicator monitoring stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'is_running': self.is_running,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'update_interval': self.update_interval,
            'total_alerts': len(self.alerts),
            'monitoring_rules': len(self.monitoring_rules),
            'current_readings': len(self.current_data),
            'last_update': max([reading.timestamp for reading in self.current_data.values()]) if self.current_data else None,
            'market_hours': self.is_market_hours
        }
    
    def get_recent_alerts(self, limit: int = 20, severity: AlertSeverity = None) -> List[IndicatorAlert]:
        """Get recent alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_symbol_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary for specific symbol"""
        symbol_data = {}
        symbol_alerts = []
        
        for key, reading in self.current_data.items():
            if reading.symbol == symbol:
                symbol_data[reading.timeframe] = reading
        
        for alert in self.alerts:
            if alert.symbol == symbol:
                symbol_alerts.append(alert)
        
        return {
            'symbol': symbol,
            'timeframes': list(symbol_data.keys()),
            'current_price': symbol_data[next(iter(symbol_data))].price_data['current'] if symbol_data else 0,
            'recent_alerts': sorted(symbol_alerts, key=lambda x: x.timestamp, reverse=True)[:10],
            'indicator_readings': {tf: reading.indicators for tf, reading in symbol_data.items()}
        }
    
    def export_monitoring_data(self, filename: str = None) -> str:
        """Export monitoring data to file"""
        if not filename:
            filename = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'status': self.get_current_status(),
                'current_data': {key: asdict(reading) for key, reading in self.current_data.items()},
                'recent_alerts': [asdict(alert) for alert in self.get_recent_alerts(100)],
                'monitoring_rules': [asdict(rule) for rule in self.monitoring_rules],
                'symbol_summaries': {symbol: self.get_symbol_summary(symbol) for symbol in self.symbols}
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            export_data = convert_datetime(export_data)
            
            # Save to file
            export_path = self.data_dir / filename
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Monitoring data exported to {export_path}")
            return str(export_path)
        
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            return ""

# Utility functions

def create_default_monitor(symbols: List[str] = None) -> RealTimeIndicatorMonitor:
    """Create monitor with default configuration"""
    if symbols is None:
        symbols = ["^NSEI", "^NSEBANK"]
    
    monitor = RealTimeIndicatorMonitor(symbols=symbols, update_interval=30)
    
    # Add custom alert callback
    def alert_callback(alert: IndicatorAlert):
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            print(f"🚨 {alert.severity.value} ALERT: {alert.message}")
    
    monitor.add_alert_callback(alert_callback)
    
    return monitor

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = create_default_monitor(["^NSEI"])
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        print("🚀 Real-time indicator monitoring started!")
        print("Monitoring NIFTY across multiple timeframes...")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(10)
            status = monitor.get_current_status()
            print(f"📊 Status: {status['total_alerts']} alerts, {status['current_readings']} readings")
            
            # Show recent high-severity alerts
            recent_alerts = monitor.get_recent_alerts(5, AlertSeverity.HIGH)
            if recent_alerts:
                print("Recent HIGH alerts:")
                for alert in recent_alerts:
                    print(f"  - {alert.message}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        monitor.stop_monitoring()
        
        # Export final data
        export_file = monitor.export_monitoring_data()
        print(f"📁 Data exported to: {export_file}")