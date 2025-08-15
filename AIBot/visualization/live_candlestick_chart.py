#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Candlestick Chart System
=============================

Real-time candlestick charts with technical indicators, pattern recognition,
and ML-based signal overlays for trading analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template_string, jsonify
import threading
import time
import asyncio

logger = logging.getLogger(__name__)

class LiveCandlestickChart:
    """
    Real-time candlestick chart with technical analysis and ML signals
    """
    
    def __init__(self, symbol: str = "^NSEI", timeframe: str = "5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()
        self.last_update = datetime.now()
        self.update_interval = 60  # seconds
        self.chart_data = {}
        self.ml_signals = []
        self.patterns_detected = []
        
        # Chart configuration
        self.chart_config = {
            'height': 800,
            'width': 1400,
            'candle_colors': {
                'increasing': '#26a69a',
                'decreasing': '#ef5350'
            },
            'indicator_colors': {
                'sma_20': '#ff9800',
                'ema_20': '#2196f3',
                'bb_upper': '#9c27b0',
                'bb_lower': '#9c27b0',
                'volume': '#607d8b'
            }
        }
        
        # Pattern recognition config
        self.pattern_config = {
            'doji_threshold': 0.1,
            'hammer_ratio': 2.0,
            'engulfing_min_body': 0.6
        }
        
        logger.info(f"LiveCandlestickChart initialized for {symbol} ({timeframe})")
    
    def fetch_realtime_data(self) -> pd.DataFrame:
        """
        Fetch real-time market data
        """
        try:
            # Determine period based on timeframe
            if self.timeframe in ['1m', '5m']:
                period = '5d'
            elif self.timeframe in ['15m', '30m']:
                period = '10d'
            elif self.timeframe == '1h':
                period = '30d'
            else:
                period = '90d'
            
            df = yf.download(self.symbol, period=period, interval=self.timeframe)
            
            if not df.empty:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                df = df.dropna()
                
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)
                
                # Detect candlestick patterns
                df = self.detect_candlestick_patterns(df)
                
                self.data = df
                self.last_update = datetime.now()
                
                logger.debug(f"Fetched {len(df)} candles for {self.symbol}")
                return df
            else:
                logger.warning(f"No data fetched for {self.symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        """
        # Moving Averages
        for period in [5, 10, 20, 50]:
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
        
        # Volume indicators
        if len(df) >= 10:
            df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Stochastic
        if len(df) >= 14:
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # ATR (Average True Range)
        if len(df) >= 14:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns using ML-enhanced logic
        """
        # Calculate basic candle properties
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Total_Range'] = df['High'] - df['Low']
        df['Body_Ratio'] = df['Body_Size'] / df['Total_Range']
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        
        # Initialize pattern columns
        patterns = ['Doji', 'Hammer', 'Hanging_Man', 'Shooting_Star', 'Engulfing_Bull', 
                   'Engulfing_Bear', 'Morning_Star', 'Evening_Star', 'Harami_Bull', 'Harami_Bear']
        
        for pattern in patterns:
            df[pattern] = 0
        
        # Doji Pattern
        doji_mask = df['Body_Ratio'] <= self.pattern_config['doji_threshold']
        df.loc[doji_mask, 'Doji'] = 1
        
        # Hammer Pattern
        hammer_mask = (
            (df['Lower_Shadow'] >= df['Body_Size'] * self.pattern_config['hammer_ratio']) &
            (df['Upper_Shadow'] <= df['Body_Size'] * 0.1) &
            (df['Body_Ratio'] > 0.1)
        )
        df.loc[hammer_mask, 'Hammer'] = 1
        
        # Hanging Man (same as hammer but at top of uptrend)
        hanging_man_mask = hammer_mask & (df['Close'] < df['Open'])
        df.loc[hanging_man_mask, 'Hanging_Man'] = 1
        
        # Shooting Star
        shooting_star_mask = (
            (df['Upper_Shadow'] >= df['Body_Size'] * self.pattern_config['hammer_ratio']) &
            (df['Lower_Shadow'] <= df['Body_Size'] * 0.1) &
            (df['Body_Ratio'] > 0.1)
        )
        df.loc[shooting_star_mask, 'Shooting_Star'] = 1
        
        # Engulfing Patterns
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Bullish Engulfing
            if (current['Close'] > current['Open'] and  # Current is bullish
                previous['Close'] < previous['Open'] and  # Previous is bearish
                current['Open'] < previous['Close'] and  # Current opens below previous close
                current['Close'] > previous['Open'] and  # Current closes above previous open
                current['Body_Size'] > previous['Body_Size'] * self.pattern_config['engulfing_min_body']):
                df.iloc[i, df.columns.get_loc('Engulfing_Bull')] = 1
            
            # Bearish Engulfing
            if (current['Close'] < current['Open'] and  # Current is bearish
                previous['Close'] > previous['Open'] and  # Previous is bullish
                current['Open'] > previous['Close'] and  # Current opens above previous close
                current['Close'] < previous['Open'] and  # Current closes below previous open
                current['Body_Size'] > previous['Body_Size'] * self.pattern_config['engulfing_min_body']):
                df.iloc[i, df.columns.get_loc('Engulfing_Bear')] = 1
        
        # Three-candle patterns (Morning Star, Evening Star)
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # Morning Star
            if (first['Close'] < first['Open'] and  # First candle is bearish
                second['Body_Ratio'] < 0.3 and  # Second candle is small
                third['Close'] > third['Open'] and  # Third candle is bullish
                third['Close'] > (first['Open'] + first['Close']) / 2):  # Third closes above first midpoint
                df.iloc[i, df.columns.get_loc('Morning_Star')] = 1
            
            # Evening Star
            if (first['Close'] > first['Open'] and  # First candle is bullish
                second['Body_Ratio'] < 0.3 and  # Second candle is small
                third['Close'] < third['Open'] and  # Third candle is bearish
                third['Close'] < (first['Open'] + first['Close']) / 2):  # Third closes below first midpoint
                df.iloc[i, df.columns.get_loc('Evening_Star')] = 1
        
        return df
    
    def create_candlestick_chart(self) -> go.Figure:
        """
        Create interactive candlestick chart with indicators
        """
        if self.data.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=('Price Action', 'Volume', 'RSI', 'MACD')
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='NIFTY 50',
                increasing_line_color=self.chart_config['candle_colors']['increasing'],
                decreasing_line_color=self.chart_config['candle_colors']['decreasing']
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=self.chart_config['indicator_colors']['sma_20'], width=1)
                ),
                row=1, col=1
            )
        
        if 'EMA_20' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['EMA_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color=self.chart_config['indicator_colors']['ema_20'], width=1)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if all(col in self.data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.chart_config['indicator_colors']['bb_upper'], width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.chart_config['indicator_colors']['bb_lower'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(156, 39, 176, 0.1)'
                ),
                row=1, col=1
            )
        
        # Add pattern annotations
        self.add_pattern_annotations(fig)
        
        # Add ML signals
        self.add_ml_signals(fig)
        
        # Volume chart
        colors = ['red' if close < open else 'green' 
                 for open, close in zip(self.data['Open'], self.data['Close'])]
        
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD
        if all(col in self.data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )
            
            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'NIFTY 50 - {self.timeframe} Live Chart',
            height=self.chart_config['height'],
            width=self.chart_config['width'],
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def add_pattern_annotations(self, fig: go.Figure):
        """
        Add candlestick pattern annotations to chart
        """
        patterns = ['Doji', 'Hammer', 'Hanging_Man', 'Shooting_Star', 
                   'Engulfing_Bull', 'Engulfing_Bear', 'Morning_Star', 'Evening_Star']
        
        pattern_colors = {
            'Doji': 'yellow',
            'Hammer': 'green',
            'Hanging_Man': 'orange',
            'Shooting_Star': 'red',
            'Engulfing_Bull': 'lime',
            'Engulfing_Bear': 'crimson',
            'Morning_Star': 'cyan',
            'Evening_Star': 'magenta'
        }
        
        for pattern in patterns:
            if pattern in self.data.columns:
                pattern_data = self.data[self.data[pattern] == 1]
                
                for idx in pattern_data.index:
                    fig.add_annotation(
                        x=idx,
                        y=pattern_data.loc[idx, 'High'] * 1.02,
                        text=pattern.replace('_', ' '),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=pattern_colors.get(pattern, 'white'),
                        font=dict(color=pattern_colors.get(pattern, 'white'), size=10),
                        bgcolor='rgba(0,0,0,0.8)',
                        row=1, col=1
                    )
    
    def add_ml_signals(self, fig: go.Figure):
        """
        Add ML-generated trading signals to chart
        """
        # This would integrate with your ML models
        # For now, adding placeholder signal logic
        
        if len(self.ml_signals) > 0:
            for signal in self.ml_signals:
                color = 'green' if signal['type'] == 'BUY' else 'red'
                symbol = '▲' if signal['type'] == 'BUY' else '▼'
                
                fig.add_annotation(
                    x=signal['timestamp'],
                    y=signal['price'],
                    text=f"{symbol} {signal['type']}",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor=color,
                    font=dict(color=color, size=12),
                    bgcolor='rgba(0,0,0,0.9)',
                    row=1, col=1
                )
    
    def update_ml_signals(self, signals: List[Dict]):
        """
        Update ML signals for display
        """
        self.ml_signals = signals[-50:]  # Keep last 50 signals
    
    def get_chart_data_json(self) -> str:
        """
        Get chart data in JSON format for web interface
        """
        if self.data.empty:
            return json.dumps({})
        
        # Prepare data for frontend
        chart_data = {
            'timestamp': self.data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': self.data['Open'].tolist(),
            'high': self.data['High'].tolist(),
            'low': self.data['Low'].tolist(),
            'close': self.data['Close'].tolist(),
            'volume': self.data['Volume'].tolist(),
        }
        
        # Add indicators if available
        indicators = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
        for indicator in indicators:
            if indicator in self.data.columns:
                chart_data[indicator.lower()] = self.data[indicator].fillna(None).tolist()
        
        # Add patterns
        patterns = ['Doji', 'Hammer', 'Engulfing_Bull', 'Engulfing_Bear']
        for pattern in patterns:
            if pattern in self.data.columns:
                pattern_indices = self.data[self.data[pattern] == 1].index
                chart_data[f'pattern_{pattern.lower()}'] = pattern_indices.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        # Add ML signals
        chart_data['ml_signals'] = self.ml_signals
        
        # Add metadata
        chart_data['metadata'] = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'last_update': self.last_update.isoformat(),
            'total_candles': len(self.data)
        }
        
        return json.dumps(chart_data, default=str)
    
    def get_latest_analysis(self) -> Dict[str, Any]:
        """
        Get latest technical analysis summary
        """
        if self.data.empty:
            return {}
        
        latest = self.data.iloc[-1]
        analysis = {
            'current_price': latest['Close'],
            'change': latest['Close'] - self.data.iloc[-2]['Close'] if len(self.data) > 1 else 0,
            'change_percent': ((latest['Close'] - self.data.iloc[-2]['Close']) / self.data.iloc[-2]['Close'] * 100) if len(self.data) > 1 else 0,
            'volume': latest['Volume'],
            'high_24h': self.data['High'].tail(24).max() if len(self.data) >= 24 else latest['High'],
            'low_24h': self.data['Low'].tail(24).min() if len(self.data) >= 24 else latest['Low'],
        }
        
        # Technical indicators summary
        if 'RSI' in self.data.columns and not pd.isna(latest['RSI']):
            analysis['rsi'] = latest['RSI']
            analysis['rsi_signal'] = 'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'
        
        if 'MACD' in self.data.columns:
            analysis['macd'] = latest['MACD']
            analysis['macd_signal'] = 'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'
        
        # Moving average signals
        ma_signals = []
        if 'SMA_20' in self.data.columns:
            ma_signals.append('Above SMA20' if latest['Close'] > latest['SMA_20'] else 'Below SMA20')
        if 'EMA_20' in self.data.columns:
            ma_signals.append('Above EMA20' if latest['Close'] > latest['EMA_20'] else 'Below EMA20')
        
        analysis['ma_signals'] = ma_signals
        
        # Pattern detection summary
        patterns_found = []
        pattern_cols = [col for col in self.data.columns if col in ['Doji', 'Hammer', 'Engulfing_Bull', 'Engulfing_Bear']]
        for pattern in pattern_cols:
            if latest[pattern] == 1:
                patterns_found.append(pattern)
        
        analysis['patterns'] = patterns_found
        analysis['timestamp'] = latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        
        return analysis
    
    def start_realtime_updates(self):
        """
        Start real-time data updates in background thread
        """
        def update_loop():
            while True:
                try:
                    self.fetch_realtime_data()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error in real-time update: {e}")
                    time.sleep(30)  # Wait longer on error
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        logger.info("Real-time updates started")
    
    def export_chart_html(self, filename: str = None) -> str:
        """
        Export chart as HTML file
        """
        if not filename:
            filename = f"nifty_chart_{self.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        fig = self.create_candlestick_chart()
        fig.write_html(filename)
        
        logger.info(f"Chart exported to {filename}")
        return filename

# Web interface for live chart
CHART_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NIFTY Live Chart - {{timeframe}}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #1e1e1e; 
            color: white; 
        }
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 20px; 
        }
        .controls { 
            display: flex; 
            gap: 10px; 
            align-items: center; 
        }
        .btn { 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        .btn:hover { 
            background: #0056b3; 
        }
        .status { 
            display: flex; 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .status-item { 
            background: #2a2a2a; 
            padding: 10px; 
            border-radius: 4px; 
            min-width: 120px; 
            text-align: center; 
        }
        .chart-container { 
            width: 100%; 
            height: 800px; 
        }
        .analysis-panel {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .positive { color: #4caf50; }
        .negative { color: #f44336; }
        .neutral { color: #ff9800; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NIFTY 50 Live Chart - {{timeframe}}</h1>
        <div class="controls">
            <select id="timeframe" class="btn">
                <option value="1m">1 Minute</option>
                <option value="5m" selected>5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="1d">1 Day</option>
            </select>
            <button class="btn" onclick="refreshChart()">Refresh</button>
            <button class="btn" onclick="toggleAutoRefresh()">Auto Refresh: <span id="autoStatus">ON</span></button>
        </div>
    </div>
    
    <div class="status">
        <div class="status-item">
            <div>Current Price</div>
            <div id="currentPrice">Loading...</div>
        </div>
        <div class="status-item">
            <div>Change</div>
            <div id="priceChange">Loading...</div>
        </div>
        <div class="status-item">
            <div>Volume</div>
            <div id="volume">Loading...</div>
        </div>
        <div class="status-item">
            <div>RSI</div>
            <div id="rsi">Loading...</div>
        </div>
        <div class="status-item">
            <div>MACD</div>
            <div id="macd">Loading...</div>
        </div>
    </div>
    
    <div id="chart" class="chart-container"></div>
    
    <div class="analysis-panel">
        <h3>Technical Analysis</h3>
        <div id="analysis">Loading analysis...</div>
    </div>

    <script>
        let autoRefresh = true;
        let refreshInterval;
        
        function updateChart() {
            fetch('/api/chart-data')
                .then(response => response.json())
                .then(data => {
                    if (data.timestamp && data.timestamp.length > 0) {
                        // Create candlestick trace
                        const candlestick = {
                            x: data.timestamp,
                            open: data.open,
                            high: data.high,
                            low: data.low,
                            close: data.close,
                            type: 'candlestick',
                            name: 'NIFTY 50',
                            increasing: {line: {color: '#26a69a'}},
                            decreasing: {line: {color: '#ef5350'}}
                        };
                        
                        const traces = [candlestick];
                        
                        // Add moving averages
                        if (data.sma_20) {
                            traces.push({
                                x: data.timestamp,
                                y: data.sma_20,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'SMA 20',
                                line: {color: '#ff9800', width: 1}
                            });
                        }
                        
                        if (data.ema_20) {
                            traces.push({
                                x: data.timestamp,
                                y: data.ema_20,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'EMA 20',
                                line: {color: '#2196f3', width: 1}
                            });
                        }
                        
                        const layout = {
                            title: 'NIFTY 50 Live Chart',
                            xaxis: {
                                rangeslider: {visible: false},
                                gridcolor: 'rgba(128,128,128,0.2)'
                            },
                            yaxis: {
                                gridcolor: 'rgba(128,128,128,0.2)'
                            },
                            template: 'plotly_dark',
                            height: 800
                        };
                        
                        Plotly.newPlot('chart', traces, layout);
                    }
                })
                .catch(error => console.error('Error updating chart:', error));
        }
        
        function updateAnalysis() {
            fetch('/api/analysis')
                .then(response => response.json())
                .then(data => {
                    // Update status bar
                    document.getElementById('currentPrice').textContent = data.current_price ? data.current_price.toFixed(2) : 'N/A';
                    
                    const changeEl = document.getElementById('priceChange');
                    if (data.change) {
                        const changeText = `${data.change.toFixed(2)} (${data.change_percent.toFixed(2)}%)`;
                        changeEl.textContent = changeText;
                        changeEl.className = data.change >= 0 ? 'positive' : 'negative';
                    }
                    
                    document.getElementById('volume').textContent = data.volume ? data.volume.toLocaleString() : 'N/A';
                    
                    const rsiEl = document.getElementById('rsi');
                    if (data.rsi) {
                        rsiEl.textContent = data.rsi.toFixed(1);
                        rsiEl.className = data.rsi > 70 ? 'negative' : data.rsi < 30 ? 'positive' : 'neutral';
                    }
                    
                    document.getElementById('macd').textContent = data.macd_signal || 'N/A';
                    
                    // Update analysis panel
                    let analysisHTML = '<div>';
                    
                    if (data.ma_signals && data.ma_signals.length > 0) {
                        analysisHTML += '<strong>Moving Averages:</strong> ' + data.ma_signals.join(', ') + '<br>';
                    }
                    
                    if (data.rsi_signal) {
                        analysisHTML += '<strong>RSI Signal:</strong> ' + data.rsi_signal + '<br>';
                    }
                    
                    if (data.patterns && data.patterns.length > 0) {
                        analysisHTML += '<strong>Patterns Detected:</strong> ' + data.patterns.join(', ') + '<br>';
                    }
                    
                    if (data.timestamp) {
                        analysisHTML += '<strong>Last Update:</strong> ' + new Date(data.timestamp).toLocaleString();
                    }
                    
                    analysisHTML += '</div>';
                    document.getElementById('analysis').innerHTML = analysisHTML;
                })
                .catch(error => console.error('Error updating analysis:', error));
        }
        
        function refreshChart() {
            updateChart();
            updateAnalysis();
        }
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('autoStatus').textContent = autoRefresh ? 'ON' : 'OFF';
            
            if (autoRefresh) {
                refreshInterval = setInterval(refreshChart, 30000); // 30 seconds
            } else {
                clearInterval(refreshInterval);
            }
        }
        
        // Initialize
        updateChart();
        updateAnalysis();
        refreshInterval = setInterval(refreshChart, 30000);
        
        // Timeframe change
        document.getElementById('timeframe').addEventListener('change', function() {
            // This would require backend support to change timeframe
            console.log('Timeframe changed to:', this.value);
        });
    </script>
</body>
</html>
"""

def create_chart_web_app(chart: LiveCandlestickChart, port: int = 8081):
    """
    Create Flask web app for live chart
    """
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        return render_template_string(CHART_HTML_TEMPLATE, timeframe=chart.timeframe)
    
    @app.route('/api/chart-data')
    def get_chart_data():
        return chart.get_chart_data_json()
    
    @app.route('/api/analysis')
    def get_analysis():
        return jsonify(chart.get_latest_analysis())
    
    @app.route('/api/refresh', methods=['POST'])
    def refresh_data():
        chart.fetch_realtime_data()
        return jsonify({'success': True, 'message': 'Data refreshed'})
    
    return app

if __name__ == "__main__":
    # Example usage
    chart = LiveCandlestickChart("^NSEI", "5m")
    chart.fetch_realtime_data()
    
    # Create web app
    app = create_chart_web_app(chart, 8081)
    
    # Start real-time updates
    chart.start_realtime_updates()
    
    print("Live Chart Server starting on http://localhost:8081")
    app.run(host='127.0.0.1', port=8081, debug=False)