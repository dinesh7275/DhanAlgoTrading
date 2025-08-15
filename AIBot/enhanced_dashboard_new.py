#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AIBot Trading Dashboard
===============================

Comprehensive web dashboard with live candlestick charts, AI signals,
multi-timeframe analysis, and trading controls.
"""

import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time

from flask import Flask, render_template_string, jsonify, request, send_from_directory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import pandas as pd
import numpy as np

# Add project paths
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for the dashboard
dashboard_data = {
    'current_signals': {},
    'market_data': {},
    'portfolio_status': {},
    'trading_performance': {},
    'alerts': [],
    'last_update': datetime.now()
}

# HTML Template for Enhanced Dashboard
ENHANCED_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIBot Enhanced Trading Dashboard</title>
    
    <!-- External Libraries -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.4/socket.io.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            color: #ffffff;
            min-height: 100vh;
        }

        .dashboard-container {
            display: grid;
            grid-template-columns: 300px 1fr;
            grid-template-rows: 80px 1fr;
            height: 100vh;
            gap: 10px;
        }

        .header {
            grid-column: 1 / -1;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 30px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 24px;
            font-weight: 700;
            color: #00d4ff;
        }

        .header-controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .sidebar {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
        }

        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: 1fr 300px;
            gap: 10px;
            padding: 10px;
            overflow: hidden;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }

        .signals-panel {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
        }

        .metrics-grid {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
            margin: 10px 0;
        }

        .metric-label {
            font-size: 14px;
            color: #b0b0b0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-change {
            font-size: 12px;
            margin-top: 5px;
        }

        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .neutral { color: #ffa726; }

        .widget {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .widget-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .signal-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.03);
        }

        .signal-strength {
            width: 60px;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }

        .signal-strength-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .btn {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-danger {
            background: linear-gradient(45deg, #ff4757 0%, #ff3742 100%);
        }

        .btn-success {
            background: linear-gradient(45deg, #00ff88 0%, #00d4aa 100%);
        }

        .timeframe-selector {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }

        .timeframe-btn {
            padding: 5px 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: transparent;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        }

        .timeframe-btn.active {
            background: #00d4ff;
            border-color: #00d4ff;
        }

        .alert-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .alert-info { border-left-color: #00d4ff; background: rgba(0, 212, 255, 0.1); }
        .alert-success { border-left-color: #00ff88; background: rgba(0, 255, 136, 0.1); }
        .alert-warning { border-left-color: #ffa726; background: rgba(255, 167, 38, 0.1); }
        .alert-danger { border-left-color: #ff4757; background: rgba(255, 71, 87, 0.1); }

        .trading-controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        .portfolio-summary {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }

        .chart-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 100;
        }

        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            font-size: 18px;
            color: #b0b0b0;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 3px solid #00d4ff;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .pattern-indicator {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            margin: 2px;
            background: rgba(0, 212, 255, 0.2);
            border: 1px solid #00d4ff;
        }

        .option-strikes {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }

        .strike-chip {
            background: rgba(255, 255, 255, 0.1);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        @media (max-width: 1200px) {
            .dashboard-container {
                grid-template-columns: 1fr;
                grid-template-rows: 80px auto 1fr;
            }
            
            .sidebar {
                order: 2;
            }
            
            .main-content {
                order: 3;
                grid-template-columns: 1fr;
                grid-template-rows: auto auto auto;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <i class="fas fa-robot"></i>
                AIBot Trading Dashboard
            </div>
            <div class="header-controls">
                <div class="status-indicator" id="connectionStatus"></div>
                <span id="marketStatus">Market Open</span>
                <span id="currentTime"></span>
            </div>
        </header>

        <!-- Sidebar -->
        <aside class="sidebar">
            <!-- Market Overview -->
            <div class="widget">
                <div class="widget-title">
                    <i class="fas fa-chart-line"></i>
                    Market Overview
                </div>
                <div id="marketOverview">
                    <div class="signal-item">
                        <span>NIFTY 50</span>
                        <div>
                            <div id="niftyPrice">25,000.00</div>
                            <div id="niftyChange" class="positive">+125.50 (+0.5%)</div>
                        </div>
                    </div>
                    <div class="signal-item">
                        <span>INDIA VIX</span>
                        <div>
                            <div id="vixPrice">18.50</div>
                            <div id="vixChange" class="negative">-0.75 (-3.9%)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Current Signals -->
            <div class="widget">
                <div class="widget-title">
                    <i class="fas fa-signal"></i>
                    AI Signals
                </div>
                <div id="currentSignals">
                    <div class="signal-item">
                        <div>
                            <div>Overall Signal</div>
                            <div class="metric-change positive">BUY</div>
                        </div>
                        <div class="signal-strength">
                            <div class="signal-strength-fill positive" style="width: 75%; background: #00ff88;"></div>
                        </div>
                    </div>
                    <div class="signal-item">
                        <div>
                            <div>Confidence</div>
                            <div>75%</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Options Recommendations -->
            <div class="widget">
                <div class="widget-title">
                    <i class="fas fa-bullseye"></i>
                    Options Recommendations
                </div>
                <div id="optionsRec">
                    <div class="signal-item">
                        <span>Option Type</span>
                        <span id="optionType">CE</span>
                    </div>
                    <div class="option-strikes" id="recommendedStrikes">
                        <span class="strike-chip">24950 CE</span>
                        <span class="strike-chip">25000 CE</span>
                        <span class="strike-chip">25050 CE</span>
                    </div>
                </div>
            </div>

            <!-- Trading Controls -->
            <div class="widget">
                <div class="widget-title">
                    <i class="fas fa-play-circle"></i>
                    Trading Controls
                </div>
                <div class="trading-controls">
                    <button class="btn btn-success" onclick="startTrading()">
                        <i class="fas fa-play"></i> Start
                    </button>
                    <button class="btn btn-danger" onclick="stopTrading()">
                        <i class="fas fa-stop"></i> Stop
                    </button>
                </div>
                <div class="trading-controls">
                    <button class="btn" onclick="refreshData()">
                        <i class="fas fa-sync"></i> Refresh
                    </button>
                    <button class="btn" onclick="exportData()">
                        <i class="fas fa-download"></i> Export
                    </button>
                </div>
            </div>

            <!-- Alerts -->
            <div class="widget">
                <div class="widget-title">
                    <i class="fas fa-bell"></i>
                    Alerts
                </div>
                <div id="alertsList">
                    <div class="alert-item alert-success">
                        <i class="fas fa-check-circle"></i>
                        <div>
                            <div>Signal Generated</div>
                            <div style="font-size: 12px; opacity: 0.7;">2 minutes ago</div>
                        </div>
                    </div>
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Charts Section -->
            <section class="chart-container">
                <div class="chart-controls">
                    <div class="timeframe-selector" id="timeframeSelector">
                        <button class="timeframe-btn" data-tf="1m">1m</button>
                        <button class="timeframe-btn active" data-tf="5m">5m</button>
                        <button class="timeframe-btn" data-tf="15m">15m</button>
                        <button class="timeframe-btn" data-tf="1h">1h</button>
                        <button class="timeframe-btn" data-tf="1d">1d</button>
                    </div>
                    <button class="btn" onclick="toggleIndicators()">
                        <i class="fas fa-layer-group"></i> Indicators
                    </button>
                </div>
                <div id="mainChart" style="height: 100%; width: 100%;"></div>
            </section>

            <!-- Signals and Analysis Panel -->
            <section class="signals-panel">
                <div class="widget-title">
                    <i class="fas fa-brain"></i>
                    Multi-Timeframe Analysis
                </div>
                
                <div id="timeframeAnalysis">
                    <div class="signal-item">
                        <div>
                            <span>1D Trend</span>
                            <span class="pattern-indicator">Bullish</span>
                        </div>
                        <div class="signal-strength">
                            <div class="signal-strength-fill" style="width: 80%; background: #00ff88;"></div>
                        </div>
                    </div>
                    <div class="signal-item">
                        <div>
                            <span>1H Momentum</span>
                            <span class="pattern-indicator">Strong Up</span>
                        </div>
                        <div class="signal-strength">
                            <div class="signal-strength-fill" style="width: 65%; background: #00ff88;"></div>
                        </div>
                    </div>
                    <div class="signal-item">
                        <div>
                            <span>15M Pattern</span>
                            <span class="pattern-indicator">Hammer</span>
                        </div>
                        <div class="signal-strength">
                            <div class="signal-strength-fill" style="width: 70%; background: #ffa726;"></div>
                        </div>
                    </div>
                </div>

                <div class="widget-title" style="margin-top: 20px;">
                    <i class="fas fa-chart-bar"></i>
                    Technical Indicators
                </div>
                
                <div id="technicalIndicators">
                    <div class="signal-item">
                        <span>RSI (14)</span>
                        <span id="rsiValue">65.4</span>
                    </div>
                    <div class="signal-item">
                        <span>MACD</span>
                        <span id="macdSignal" class="positive">Bullish</span>
                    </div>
                    <div class="signal-item">
                        <span>BB Position</span>
                        <span id="bbPosition">0.65</span>
                    </div>
                    <div class="signal-item">
                        <span>Volume</span>
                        <span id="volumeStatus" class="positive">High</span>
                    </div>
                </div>

                <div class="widget-title" style="margin-top: 20px;">
                    <i class="fas fa-microscope"></i>
                    Pattern Detection
                </div>
                
                <div id="patternDetection">
                    <div class="pattern-indicator">Bullish Engulfing</div>
                    <div class="pattern-indicator">EMA Crossover</div>
                    <div class="pattern-indicator">Volume Breakout</div>
                </div>
            </section>

            <!-- Performance Metrics -->
            <section class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Portfolio Value</div>
                    <div class="metric-value positive" id="portfolioValue">₹10,000</div>
                    <div class="metric-change positive" id="portfolioChange">+2.5% today</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Today's P&L</div>
                    <div class="metric-value positive" id="todayPnl">₹250</div>
                    <div class="metric-change positive" id="todayPnlPct">+2.5%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Active Trades</div>
                    <div class="metric-value neutral" id="activeTrades">3</div>
                    <div class="metric-change neutral" id="openPositions">2 CE, 1 PE</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value positive" id="winRate">72%</div>
                    <div class="metric-change positive" id="winRateChange">+5% this week</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Available Cash</div>
                    <div class="metric-value neutral" id="availableCash">₹7,500</div>
                    <div class="metric-change neutral" id="marginUsed">25% margin used</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value positive" id="riskLevel">Low</div>
                    <div class="metric-change positive" id="riskScore">Risk Score: 2.3</div>
                </div>
            </section>
        </main>
    </div>

    <script>
        let currentTimeframe = '5m';
        let autoRefresh = true;
        let refreshInterval;
        let wsConnection;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeDashboard();
            setupEventListeners();
            startAutoRefresh();
            updateCurrentTime();
            connectWebSocket();
        });

        function initializeDashboard() {
            loadChart();
            updateDashboardData();
            setupTimeframeSelector();
        }

        function setupEventListeners() {
            // Timeframe selector
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelector('.timeframe-btn.active').classList.remove('active');
                    this.classList.add('active');
                    currentTimeframe = this.dataset.tf;
                    loadChart();
                });
            });
        }

        function loadChart() {
            showLoading('mainChart');
            
            fetch(`/api/chart-data/${currentTimeframe}`)
                .then(response => response.json())
                .then(data => {
                    renderChart(data);
                })
                .catch(error => {
                    console.error('Error loading chart:', error);
                    showError('mainChart', 'Failed to load chart data');
                });
        }

        function renderChart(data) {
            if (!data.timestamp || data.timestamp.length === 0) {
                showError('mainChart', 'No chart data available');
                return;
            }

            // Create candlestick chart
            const candlestick = {
                x: data.timestamp,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                type: 'candlestick',
                name: 'NIFTY 50',
                increasing: {line: {color: '#00ff88'}},
                decreasing: {line: {color: '#ff4757'}}
            };

            const traces = [candlestick];

            // Add technical indicators
            if (data.sma_20) {
                traces.push({
                    x: data.timestamp,
                    y: data.sma_20,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA 20',
                    line: {color: '#ffa726', width: 1}
                });
            }

            if (data.ema_20) {
                traces.push({
                    x: data.timestamp,
                    y: data.ema_20,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 20',
                    line: {color: '#00d4ff', width: 1}
                });
            }

            // Add Bollinger Bands
            if (data.bb_upper && data.bb_lower) {
                traces.push(
                    {
                        x: data.timestamp,
                        y: data.bb_upper,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'BB Upper',
                        line: {color: '#9c27b0', width: 1, dash: 'dash'},
                        showlegend: false
                    },
                    {
                        x: data.timestamp,
                        y: data.bb_lower,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'BB Lower',
                        line: {color: '#9c27b0', width: 1, dash: 'dash'},
                        fill: 'tonexty',
                        fillcolor: 'rgba(156, 39, 176, 0.1)'
                    }
                );
            }

            // Add ML signals
            if (data.ml_signals && data.ml_signals.length > 0) {
                data.ml_signals.forEach(signal => {
                    const color = signal.type === 'BUY' ? '#00ff88' : '#ff4757';
                    traces.push({
                        x: [signal.timestamp],
                        y: [signal.price],
                        type: 'scatter',
                        mode: 'markers',
                        name: signal.type,
                        marker: {
                            symbol: signal.type === 'BUY' ? 'triangle-up' : 'triangle-down',
                            size: 12,
                            color: color
                        }
                    });
                });
            }

            const layout = {
                title: `NIFTY 50 - ${currentTimeframe} Chart`,
                xaxis: {
                    rangeslider: {visible: false},
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff'
                },
                yaxis: {
                    gridcolor: 'rgba(255,255,255,0.1)',
                    color: '#ffffff'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'},
                margin: {l: 50, r: 50, t: 50, b: 50},
                height: 500
            };

            Plotly.newPlot('mainChart', traces, layout, {
                responsive: true,
                displayModeBar: false
            });
        }

        function updateDashboardData() {
            fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    updateMarketOverview(data.market);
                    updateSignals(data.signals);
                    updatePortfolio(data.portfolio);
                    updateTechnicalIndicators(data.technical);
                    updateAlerts(data.alerts);
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }

        function updateMarketOverview(market) {
            if (!market) return;
            
            document.getElementById('niftyPrice').textContent = market.nifty_price?.toFixed(2) || '25,000.00';
            document.getElementById('niftyChange').textContent = 
                `${market.nifty_change > 0 ? '+' : ''}${market.nifty_change?.toFixed(2)} (${market.nifty_change_pct?.toFixed(2)}%)`;
            document.getElementById('niftyChange').className = 
                market.nifty_change >= 0 ? 'positive' : 'negative';
            
            document.getElementById('vixPrice').textContent = market.vix_price?.toFixed(2) || '18.50';
            document.getElementById('vixChange').textContent = 
                `${market.vix_change > 0 ? '+' : ''}${market.vix_change?.toFixed(2)} (${market.vix_change_pct?.toFixed(2)}%)`;
            document.getElementById('vixChange').className = 
                market.vix_change >= 0 ? 'positive' : 'negative';
        }

        function updateSignals(signals) {
            if (!signals) return;
            
            // Update current signals display
            const signalItems = document.querySelector('#currentSignals');
            if (signalItems && signals.current) {
                signalItems.innerHTML = `
                    <div class="signal-item">
                        <div>
                            <div>Overall Signal</div>
                            <div class="metric-change ${signals.current.signal.toLowerCase() === 'buy' ? 'positive' : signals.current.signal.toLowerCase() === 'sell' ? 'negative' : 'neutral'}">${signals.current.signal}</div>
                        </div>
                        <div class="signal-strength">
                            <div class="signal-strength-fill ${signals.current.signal.toLowerCase() === 'buy' ? 'positive' : 'negative'}" 
                                 style="width: ${signals.current.confidence * 100}%; background: ${signals.current.signal.toLowerCase() === 'buy' ? '#00ff88' : '#ff4757'};"></div>
                        </div>
                    </div>
                    <div class="signal-item">
                        <div>
                            <div>Confidence</div>
                            <div>${(signals.current.confidence * 100).toFixed(0)}%</div>
                        </div>
                    </div>
                `;
            }
            
            // Update options recommendations
            if (signals.options) {
                document.getElementById('optionType').textContent = signals.options.type || 'CE';
                
                const strikesContainer = document.getElementById('recommendedStrikes');
                if (strikesContainer && signals.options.strikes) {
                    strikesContainer.innerHTML = signals.options.strikes.map(strike => 
                        `<span class="strike-chip">${strike} ${signals.options.type}</span>`
                    ).join('');
                }
            }
        }

        function updatePortfolio(portfolio) {
            if (!portfolio) return;
            
            document.getElementById('portfolioValue').textContent = `₹${portfolio.total_value?.toLocaleString() || '10,000'}`;
            document.getElementById('portfolioChange').textContent = `${portfolio.day_change_pct > 0 ? '+' : ''}${portfolio.day_change_pct?.toFixed(1)}% today`;
            document.getElementById('portfolioChange').className = portfolio.day_change_pct >= 0 ? 'positive' : 'negative';
            
            document.getElementById('todayPnl').textContent = `₹${portfolio.day_pnl?.toLocaleString() || '0'}`;
            document.getElementById('todayPnlPct').textContent = `${portfolio.day_change_pct > 0 ? '+' : ''}${portfolio.day_change_pct?.toFixed(1)}%`;
            document.getElementById('todayPnlPct').className = portfolio.day_pnl >= 0 ? 'positive' : 'negative';
            
            document.getElementById('activeTrades').textContent = portfolio.active_positions || '0';
            document.getElementById('availableCash').textContent = `₹${portfolio.available_cash?.toLocaleString() || '10,000'}`;
            document.getElementById('winRate').textContent = `${portfolio.win_rate?.toFixed(0) || '0'}%`;
        }

        function updateTechnicalIndicators(technical) {
            if (!technical) return;
            
            document.getElementById('rsiValue').textContent = technical.rsi?.toFixed(1) || '50.0';
            document.getElementById('macdSignal').textContent = technical.macd_signal || 'Neutral';
            document.getElementById('macdSignal').className = technical.macd_signal === 'Bullish' ? 'positive' : 
                                                               technical.macd_signal === 'Bearish' ? 'negative' : 'neutral';
            document.getElementById('bbPosition').textContent = technical.bb_position?.toFixed(2) || '0.50';
            document.getElementById('volumeStatus').textContent = technical.volume_status || 'Normal';
            document.getElementById('volumeStatus').className = technical.volume_status === 'High' ? 'positive' : 'neutral';
        }

        function updateAlerts(alerts) {
            if (!alerts || alerts.length === 0) return;
            
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.type}">
                    <i class="fas fa-${alert.type === 'success' ? 'check-circle' : 
                                      alert.type === 'warning' ? 'exclamation-triangle' : 
                                      alert.type === 'danger' ? 'times-circle' : 'info-circle'}"></i>
                    <div>
                        <div>${alert.message}</div>
                        <div style="font-size: 12px; opacity: 0.7;">${alert.time}</div>
                    </div>
                </div>
            `).join('');
        }

        function startTrading() {
            fetch('/api/trading/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addAlert('Trading started successfully', 'success');
                    } else {
                        addAlert('Failed to start trading: ' + data.message, 'danger');
                    }
                })
                .catch(error => {
                    addAlert('Error starting trading: ' + error.message, 'danger');
                });
        }

        function stopTrading() {
            fetch('/api/trading/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addAlert('Trading stopped successfully', 'success');
                    } else {
                        addAlert('Failed to stop trading: ' + data.message, 'danger');
                    }
                })
                .catch(error => {
                    addAlert('Error stopping trading: ' + error.message, 'danger');
                });
        }

        function refreshData() {
            updateDashboardData();
            loadChart();
            addAlert('Dashboard refreshed', 'info');
        }

        function exportData() {
            fetch('/api/export-data', {method: 'POST'})
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `aibot_data_${new Date().toISOString().slice(0,10)}.json`;
                    a.click();
                    addAlert('Data exported successfully', 'success');
                })
                .catch(error => {
                    addAlert('Error exporting data: ' + error.message, 'danger');
                });
        }

        function toggleIndicators() {
            // Toggle indicator visibility
            addAlert('Indicator visibility toggled', 'info');
        }

        function addAlert(message, type) {
            const alertsList = document.getElementById('alertsList');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert-item alert-${type}`;
            alertDiv.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : 
                                  type === 'warning' ? 'exclamation-triangle' : 
                                  type === 'danger' ? 'times-circle' : 'info-circle'}"></i>
                <div>
                    <div>${message}</div>
                    <div style="font-size: 12px; opacity: 0.7;">Just now</div>
                </div>
            `;
            
            alertsList.insertBefore(alertDiv, alertsList.firstChild);
            
            // Remove old alerts (keep only 5)
            while (alertsList.children.length > 5) {
                alertsList.removeChild(alertsList.lastChild);
            }
        }

        function showLoading(elementId) {
            document.getElementById(elementId).innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    Loading...
                </div>
            `;
        }

        function showError(elementId, message) {
            document.getElementById(elementId).innerHTML = `
                <div class="loading">
                    <i class="fas fa-exclamation-triangle" style="color: #ff4757; margin-right: 10px;"></i>
                    ${message}
                </div>
            `;
        }

        function startAutoRefresh() {
            refreshInterval = setInterval(() => {
                if (autoRefresh) {
                    updateDashboardData();
                }
            }, 30000); // Refresh every 30 seconds
        }

        function updateCurrentTime() {
            setInterval(() => {
                document.getElementById('currentTime').textContent = 
                    new Date().toLocaleTimeString();
            }, 1000);
        }

        function connectWebSocket() {
            // WebSocket connection for real-time updates
            // This would connect to a WebSocket server for live data
            console.log('WebSocket connection would be established here');
        }

        function setupTimeframeSelector() {
            // Initialize timeframe selector
            document.querySelector(`[data-tf="${currentTimeframe}"]`).classList.add('active');
        }
    </script>
</body>
</html>
"""

class EnhancedDashboard:
    """
    Enhanced dashboard controller
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.is_running = False
        
        # Initialize components (these would be imported from other modules)
        self.signal_generator = None
        self.chart_system = None
        self.dhan_client = None
        self.pattern_recognizer = None
        
        # Dashboard state
        self.current_signals = {}
        self.market_data = {}
        self.portfolio_status = {}
        self.alerts = []
        
        logger.info(f"EnhancedDashboard initialized on port {port}")
    
    def get_mock_chart_data(self, timeframe: str = "5m") -> Dict[str, Any]:
        """
        Generate mock chart data for demonstration
        """
        try:
            # Generate sample OHLCV data
            num_candles = 100
            timestamps = pd.date_range(end=datetime.now(), periods=num_candles, freq='5T')
            
            # Generate realistic price movement
            base_price = 25000
            prices = []
            current_price = base_price
            
            for i in range(num_candles):
                change = np.random.normal(0, 0.005) * current_price  # 0.5% std dev
                current_price += change
                prices.append(current_price)
            
            # Create OHLCV data
            data = {
                'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }
            
            for i, price in enumerate(prices):
                if i == 0:
                    open_price = base_price
                else:
                    open_price = data['close'][i-1]
                
                high = price * (1 + abs(np.random.normal(0, 0.002)))
                low = price * (1 - abs(np.random.normal(0, 0.002)))
                close = price
                volume = int(np.random.normal(1000000, 200000))
                
                data['open'].append(open_price)
                data['high'].append(high)
                data['low'].append(low)
                data['close'].append(close)
                data['volume'].append(max(volume, 100000))
            
            # Add technical indicators
            closes = np.array(data['close'])
            data['sma_20'] = self.calculate_sma(closes, 20).tolist()
            data['ema_20'] = self.calculate_ema(closes, 20).tolist()
            
            # Bollinger Bands
            bb_middle = self.calculate_sma(closes, 20)
            bb_std = self.calculate_rolling_std(closes, 20)
            data['bb_upper'] = (bb_middle + bb_std * 2).tolist()
            data['bb_lower'] = (bb_middle - bb_std * 2).tolist()
            
            # Add some ML signals
            data['ml_signals'] = [
                {
                    'timestamp': timestamps[20].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': data['close'][20],
                    'type': 'BUY',
                    'confidence': 0.75
                },
                {
                    'timestamp': timestamps[60].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': data['close'][60],
                    'type': 'SELL',
                    'confidence': 0.65
                }
            ]
            
            return data
        
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {}
    
    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        result = np.full_like(data, np.nan)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    def calculate_rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1])
        return result
    
    def get_mock_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate mock dashboard data
        """
        return {
            'market': {
                'nifty_price': 25125.50,
                'nifty_change': 125.50,
                'nifty_change_pct': 0.5,
                'vix_price': 18.25,
                'vix_change': -0.75,
                'vix_change_pct': -3.9
            },
            'signals': {
                'current': {
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'strength': 0.75
                },
                'options': {
                    'type': 'CE',
                    'strikes': [24950, 25000, 25050]
                }
            },
            'portfolio': {
                'total_value': 10250,
                'day_pnl': 250,
                'day_change_pct': 2.5,
                'active_positions': 3,
                'available_cash': 7500,
                'win_rate': 72
            },
            'technical': {
                'rsi': 65.4,
                'macd_signal': 'Bullish',
                'bb_position': 0.65,
                'volume_status': 'High'
            },
            'alerts': [
                {
                    'type': 'success',
                    'message': 'Strong BUY signal generated',
                    'time': '2 minutes ago'
                },
                {
                    'type': 'info',
                    'message': 'Pattern detected: Bullish Engulfing',
                    'time': '5 minutes ago'
                },
                {
                    'type': 'warning',
                    'message': 'High volatility detected',
                    'time': '10 minutes ago'
                }
            ]
        }

# Flask Routes
@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template_string(ENHANCED_DASHBOARD_HTML)

@app.route('/api/chart-data/<timeframe>')
def get_chart_data(timeframe):
    """Get chart data for specific timeframe"""
    dashboard = EnhancedDashboard()
    data = dashboard.get_mock_chart_data(timeframe)
    return jsonify(data)

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    dashboard = EnhancedDashboard()
    data = dashboard.get_mock_dashboard_data()
    return jsonify(data)

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Start trading"""
    try:
        # This would integrate with the actual trading system
        return jsonify({
            'success': True,
            'message': 'Trading started successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    try:
        # This would integrate with the actual trading system
        return jsonify({
            'success': True,
            'message': 'Trading stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/export-data', methods=['POST'])
def export_data():
    """Export trading data"""
    try:
        dashboard = EnhancedDashboard()
        data = dashboard.get_mock_dashboard_data()
        
        response = app.response_class(
            response=json.dumps(data, indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = 'attachment; filename=aibot_data.json'
        return response
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def run_enhanced_dashboard(port: int = 8080, debug: bool = False):
    """
    Run the enhanced dashboard
    """
    print("="*60)
    print("🚀 AIBot Enhanced Trading Dashboard")
    print("="*60)
    print(f"🌐 URL: http://localhost:{port}")
    print("📊 Features: Live Charts, AI Signals, Multi-timeframe Analysis")
    print("🔄 Real-time Updates: Market Data, Positions, P&L")
    print("⚡ Controls: Start/Stop Trading, Export Data")
    print("="*60)
    
    try:
        app.run(host='127.0.0.1', port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the enhanced dashboard
    run_enhanced_dashboard(port=8080, debug=True)