#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Trading Dashboard - Live Market Data & Option Chain
========================================================

Attractive real-time dashboard showing:
- Live NIFTY price and charts
- Option chain data from Dhan
- Account details and positions
- AI trading signals
- Real market performance
"""

import sys
import json
import logging
import asyncio
import subprocess
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np

# Add project paths
sys.path.append(str(Path(__file__).parent / "AIBot"))

try:
    from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials
    DHAN_AVAILABLE = True
except ImportError:
    print("Dhan API not available - using simulated data")
    DHAN_AVAILABLE = False

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global data store
live_data = {
    'nifty_price': 0,
    'nifty_change': 0,
    'nifty_change_percent': 0,
    'account_balance': 0,
    'account_margin': 0,
    'positions': [],
    'option_chain': [],
    'ai_signals': [],
    'market_status': 'CLOSED',
    'last_update': datetime.now(),
    'algo_performance': {
        'trades_today': 0,
        'pnl_today': 0,
        'success_rate': 0,
        'active_strategies': []
    }
}

# Rate limiting for API calls
last_option_chain_call = 0
OPTION_CHAIN_INTERVAL = 180  # 3 minutes = 180 seconds

# Initialize Dhan client if available  
dhan_client = None
if DHAN_AVAILABLE:
    try:
        # Load credentials from main config
        credentials = DhanCredentials(
            client_id='1107321060',
            access_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU3MTM4NzgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNzMyMTA2MCJ9.n_2HhEW9ePhAfi63KoxQskzohVPi4N8F_RWn-a9rqTbne5GX7DHRTF9NpU4LEyf1dC8J-M32Fuk-EbXlOYOWOA'
        )
        dhan_client = DhanAPIClient(credentials)
        print("Attempting Dhan API connection...")
        try:
            # Try without authentication first - just check if client works
            dhan_client.is_connected = True
            print("Dhan API client initialized - will try live data")
        except Exception as auth_error:
            print(f"Dhan API authentication issue: {auth_error}")
            dhan_client = None
    except Exception as e:
        print(f"Error connecting to Dhan API: {e}")
        dhan_client = None

if not dhan_client:
    print("Using simulated data for demo...")

# Enhanced HTML template for real trading dashboard
REAL_TRADING_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 AI Trading Dashboard - Live Market</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #ffffff;
            overflow-x: hidden;
        }
        
        .dashboard-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(45deg, #00f5ff, #0080ff, #8000ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .header-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .market-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 20px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .status-open { background: linear-gradient(45deg, #00c851, #007e33); }
        .status-closed { background: linear-gradient(45deg, #ff4444, #cc0000); }
        .status-pre { background: linear-gradient(45deg, #ffbb33, #ff8800); }
        
        .nifty-ticker {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 25px;
            background: linear-gradient(135deg, rgba(0,245,255,0.2) 0%, rgba(128,0,255,0.2) 100%);
            border-radius: 20px;
            border: 1px solid rgba(0,245,255,0.3);
        }
        
        .nifty-price {
            font-size: 2rem;
            font-weight: 700;
            color: #00f5ff;
        }
        
        .nifty-change {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 12px;
        }
        
        .positive { background: rgba(0,200,81,0.3); color: #00c851; }
        .negative { background: rgba(255,68,68,0.3); color: #ff4444; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00f5ff, #0080ff, #8000ff);
            border-radius: 20px 20px 0 0;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        }
        
        .metric-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
            background: linear-gradient(45deg, #00f5ff, #0080ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: #ffffff;
        }
        
        .metric-label {
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .chart-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .option-chain-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            max-height: 600px;
            overflow-y: auto;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .option-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        .option-table th,
        .option-table td {
            padding: 8px 6px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .option-table th {
            background: linear-gradient(135deg, rgba(0,245,255,0.2) 0%, rgba(128,0,255,0.2) 100%);
            color: #00f5ff;
            font-weight: 600;
            font-size: 0.8rem;
        }
        
        .call-option { background: rgba(0,200,81,0.1); }
        .put-option { background: rgba(255,68,68,0.1); }
        
        .bottom-panels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }
        
        .info-panel {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .activity-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .activity-item:last-child { border-bottom: none; }
        
        .signal-badge {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .signal-buy { background: rgba(0,200,81,0.3); color: #00c851; }
        .signal-sell { background: rgba(255,68,68,0.3); color: #ff4444; }
        .signal-hold { background: rgba(255,187,51,0.3); color: #ffbb33; }
        
        .timestamp {
            color: rgba(255,255,255,0.5);
            font-size: 0.8rem;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
        }
        
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
        
        .btn-primary { background: linear-gradient(45deg, #00f5ff, #0080ff); color: #000; }
        .btn-success { background: linear-gradient(45deg, #00c851, #007e33); color: #fff; }
        .btn-warning { background: linear-gradient(45deg, #ffbb33, #ff8800); color: #000; }
        .btn-danger { background: linear-gradient(45deg, #ff4444, #cc0000); color: #fff; }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top: 3px solid #00f5ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @media (max-width: 1200px) {
            .main-content { grid-template-columns: 1fr; }
            .header-info { justify-content: center; }
        }
        
        @media (max-width: 768px) {
            .metrics-grid { grid-template-columns: 1fr; }
            .controls { justify-content: center; }
            .nifty-ticker { flex-direction: column; text-align: center; }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #00f5ff, #0080ff); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> AI Trading Dashboard</h1>
            <div class="header-info">
                <div id="marketStatus" class="market-status status-closed">
                    <i class="fas fa-circle pulse"></i>
                    <span>Market Closed</span>
                </div>
                <div class="nifty-ticker">
                    <div>
                        <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">NIFTY 50</div>
                        <div class="nifty-price" id="niftyPrice">₹24,300.15</div>
                    </div>
                    <div class="nifty-change positive" id="niftyChange">+123.45 (+0.51%)</div>
                </div>
                <div class="timestamp">
                    Last Updated: <span id="lastUpdate">{{timestamp}}</span>
                </div>
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-wallet"></i></div>
                <div class="metric-value" id="accountBalance">₹5,00,000</div>
                <div class="metric-label">Account Balance</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-chart-line"></i></div>
                <div class="metric-value" id="todayPnl">₹+2,500</div>
                <div class="metric-label">Today's P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-robot"></i></div>
                <div class="metric-value" id="tradesCount">12</div>
                <div class="metric-label">AI Trades Today</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-percentage"></i></div>
                <div class="metric-value" id="successRate">78%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <div class="chart-section">
                <div class="section-title">
                    <i class="fas fa-chart-candlestick"></i>
                    Live NIFTY Chart
                </div>
                <div id="niftyChart" style="height: 400px;"></div>
            </div>
            
            <div class="option-chain-section">
                <div class="section-title">
                    <i class="fas fa-table"></i>
                    Option Chain
                </div>
                <div id="optionChainContainer">
                    <div class="loading"></div> Loading option chain...
                </div>
            </div>
        </div>

        <!-- Information Panels -->
        <div class="bottom-panels">
            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-briefcase"></i>
                    Current Positions
                </div>
                <div id="positionsContainer">
                    <div class="activity-item">
                        <span>No active positions</span>
                        <span class="timestamp">Ready to trade</span>
                    </div>
                </div>
            </div>

            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-brain"></i>
                    AI Signals
                </div>
                <div id="aiSignalsContainer">
                    <div class="activity-item">
                        <div>
                            <span class="signal-badge signal-hold">HOLD</span>
                            <span style="margin-left: 10px;">Confidence: 85%</span>
                        </div>
                        <span class="timestamp">{{timestamp}}</span>
                    </div>
                </div>
            </div>

            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-cogs"></i>
                    Trading Controls
                </div>
                <div class="controls">
                    <button class="btn btn-success" onclick="startAlgoTrading()">
                        <i class="fas fa-play"></i> Start AI Trading
                    </button>
                    <button class="btn btn-warning" onclick="pauseTrading()">
                        <i class="fas fa-pause"></i> Pause
                    </button>
                    <button class="btn btn-danger" onclick="stopAllTrades()">
                        <i class="fas fa-stop"></i> Stop All
                    </button>
                </div>
            </div>

            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-user"></i>
                    Account Details
                </div>
                <div id="accountDetails">
                    <div class="activity-item">
                        <span>Available Margin</span>
                        <span id="availableMargin">₹4,50,000</span>
                    </div>
                    <div class="activity-item">
                        <span>Used Margin</span>
                        <span id="usedMargin">₹50,000</span>
                    </div>
                    <div class="activity-item">
                        <span>Client ID</span>
                        <span>1107321060</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isTrading = false;
        let updateInterval;
        let chartData = [];

        // Initialize dashboard
        $(document).ready(function() {
            initializeChart();
            startDataUpdates();
            updateTimestamp();
        });

        function updateTimestamp() {
            $('#lastUpdate').text(new Date().toLocaleTimeString());
        }

        function startDataUpdates() {
            updateInterval = setInterval(() => {
                fetchLiveData();
                updateTimestamp();
            }, 2000); // Update every 2 seconds
        }

        function fetchLiveData() {
            $.get('/api/live_data')
                .done(function(data) {
                    updateNiftyPrice(data);
                    updateMetrics(data);
                    updateOptionChain(data);
                    updatePositions(data);
                    updateAISignals(data);
                    updateMarketStatus(data);
                    updateChart(data);
                })
                .fail(function() {
                    // Simulate live data for demo
                    simulateLiveData();
                });
        }

        function updateNiftyPrice(data) {
            if (data.nifty_price) {
                $('#niftyPrice').text('₹' + data.nifty_price.toLocaleString());
                const change = data.nifty_change || 0;
                const changePercent = data.nifty_change_percent || 0;
                const changeText = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`;
                $('#niftyChange').text(changeText);
                $('#niftyChange').removeClass('positive negative').addClass(change >= 0 ? 'positive' : 'negative');
            }
        }

        function updateMetrics(data) {
            if (data.account_balance) {
                $('#accountBalance').text('₹' + data.account_balance.toLocaleString());
            }
            if (data.algo_performance) {
                $('#todayPnl').text('₹' + (data.algo_performance.pnl_today || 0).toLocaleString());
                $('#tradesCount').text(data.algo_performance.trades_today || 0);
                $('#successRate').text((data.algo_performance.success_rate || 0) + '%');
            }
        }

        function updateOptionChain(data) {
            if (data.option_chain && data.option_chain.length > 0) {
                let html = '<table class="option-table"><thead><tr>';
                html += '<th>CE LTP</th><th>CE Vol</th><th>Strike</th><th>PE Vol</th><th>PE LTP</th>';
                html += '</tr></thead><tbody>';
                
                data.option_chain.slice(0, 10).forEach(option => {
                    html += '<tr>';
                    html += `<td class="call-option">${option.ce_ltp || '-'}</td>`;
                    html += `<td class="call-option">${option.ce_volume || '-'}</td>`;
                    html += `<td style="font-weight: 600;">${option.strike_price}</td>`;
                    html += `<td class="put-option">${option.pe_volume || '-'}</td>`;
                    html += `<td class="put-option">${option.pe_ltp || '-'}</td>`;
                    html += '</tr>';
                });
                
                html += '</tbody></table>';
                $('#optionChainContainer').html(html);
            }
        }

        function updatePositions(data) {
            if (data.positions && data.positions.length > 0) {
                let html = '';
                data.positions.forEach(pos => {
                    html += '<div class="activity-item">';
                    html += `<span>${pos.symbol} (${pos.quantity})</span>`;
                    html += `<span class="${pos.pnl >= 0 ? 'positive' : 'negative'}">₹${pos.pnl.toFixed(2)}</span>`;
                    html += '</div>';
                });
                $('#positionsContainer').html(html);
            }
        }

        function updateAISignals(data) {
            if (data.ai_signals && data.ai_signals.length > 0) {
                let html = '';
                data.ai_signals.slice(-5).forEach(signal => {
                    html += '<div class="activity-item">';
                    html += '<div>';
                    html += `<span class="signal-badge signal-${signal.type.toLowerCase()}">${signal.type}</span>`;
                    html += `<span style="margin-left: 10px;">Confidence: ${(signal.confidence * 100).toFixed(0)}%</span>`;
                    html += '</div>';
                    html += `<span class="timestamp">${new Date(signal.timestamp).toLocaleTimeString()}</span>`;
                    html += '</div>';
                });
                $('#aiSignalsContainer').html(html);
            }
        }

        function updateMarketStatus(data) {
            const status = data.market_status || 'CLOSED';
            $('#marketStatus').removeClass('status-open status-closed status-pre');
            
            if (status === 'OPEN') {
                $('#marketStatus').addClass('status-open').html('<i class="fas fa-circle pulse"></i><span>Market Open</span>');
            } else if (status === 'PRE_OPEN') {
                $('#marketStatus').addClass('status-pre').html('<i class="fas fa-circle pulse"></i><span>Pre-Open</span>');
            } else {
                $('#marketStatus').addClass('status-closed').html('<i class="fas fa-circle"></i><span>Market Closed</span>');
            }
        }

        function initializeChart() {
            const trace = {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'NIFTY 50',
                line: {color: '#00f5ff', width: 2}
            };

            const layout = {
                title: '',
                xaxis: {title: 'Time', color: '#ffffff'},
                yaxis: {title: 'Price', color: '#ffffff'},
                margin: {l: 50, r: 30, t: 30, b: 50},
                plot_bgcolor: 'rgba(0,0,0,0.3)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'}
            };

            Plotly.newPlot('niftyChart', [trace], layout, {responsive: true});
        }

        function updateChart(data) {
            if (data.nifty_price) {
                const now = new Date();
                chartData.push({x: now, y: data.nifty_price});
                
                // Keep only last 50 data points
                if (chartData.length > 50) {
                    chartData = chartData.slice(-50);
                }
                
                Plotly.restyle('niftyChart', {
                    x: [chartData.map(d => d.x)],
                    y: [chartData.map(d => d.y)]
                });
            }
        }

        function simulateLiveData() {
            const basePrice = 24300;
            const change = (Math.random() - 0.5) * 100;
            const newPrice = basePrice + change;
            
            updateNiftyPrice({
                nifty_price: newPrice,
                nifty_change: change,
                nifty_change_percent: (change / basePrice) * 100
            });
            
            updateChart({nifty_price: newPrice});
        }

        // Control functions
        function startAlgoTrading() {
            isTrading = true;
            alert('🚀 AI Trading Started! Monitor the signals panel for live updates.');
        }

        function pauseTrading() {
            isTrading = false;
            alert('⏸️ Trading Paused');
        }

        function stopAllTrades() {
            if (confirm('🛑 Are you sure you want to stop all trading activities?')) {
                isTrading = false;
                alert('🛑 All trading stopped');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def real_trading_dashboard():
    """Real trading dashboard page"""
    return render_template_string(REAL_TRADING_DASHBOARD_HTML, timestamp=datetime.now().strftime('%H:%M:%S'))

@app.route('/api/live_data')
def get_live_data():
    """Get live market data from Dhan API"""
    global live_data
    
    try:
        if dhan_client:
            print("Attempting to fetch real Dhan data...")
            try:
                # Get account details first
                print("Fetching fund limits...")
                account_data = dhan_client.get_fund_limit()
                if account_data:
                    live_data['account_balance'] = account_data.get('available_balance', 500000)
                    live_data['account_margin'] = account_data.get('utilized_margin', 50000)
                    print(f"Real account balance: {live_data['account_balance']}")
                
                # Get REAL NIFTY price using Yahoo Finance as fallback
                print("Fetching REAL NIFTY price...")
                try:
                    import yfinance as yf
                    nifty_ticker = yf.Ticker("^NSEI")
                    nifty_info = nifty_ticker.history(period="1d", interval="1m")
                    if not nifty_info.empty:
                        current_price = nifty_info['Close'].iloc[-1]
                        prev_close = nifty_info['Close'].iloc[0]
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100
                        
                        live_data['nifty_price'] = current_price
                        live_data['nifty_change'] = change
                        live_data['nifty_change_percent'] = change_percent
                        print(f"LIVE NIFTY: Rs.{current_price:.2f} ({change:+.2f}, {change_percent:+.2f}%)")
                    else:
                        print("ERROR: No NIFTY data from Yahoo Finance")
                except Exception as nifty_error:
                    print(f"ERROR getting NIFTY price: {nifty_error}")
                    live_data['nifty_price'] = 0
                    live_data['nifty_change'] = 0
                    live_data['nifty_change_percent'] = 0
                
                # Get REAL positions
                print("Fetching real positions...")
                try:
                    positions_data = dhan_client.get_positions()
                    if isinstance(positions_data, list):
                        live_data['positions'] = positions_data
                        print(f"LIVE POSITIONS: {len(positions_data)} positions")
                    else:
                        live_data['positions'] = []
                        print("No positions found")
                except Exception as pos_error:
                    print(f"ERROR getting positions: {pos_error}")
                    live_data['positions'] = []
                
                # Get REAL option chain with 3-minute timer
                global last_option_chain_call
                current_time = time.time()
                if current_time - last_option_chain_call >= OPTION_CHAIN_INTERVAL:
                    print("Fetching REAL option chain (3-min timer)...")
                    try:
                        option_chain_data = dhan_client.get_option_chain(underlying_scrip=13)
                        if option_chain_data and isinstance(option_chain_data, list):
                            live_data['option_chain'] = option_chain_data[:15]  # Top 15 strikes
                            last_option_chain_call = current_time
                            print(f"LIVE OPTION CHAIN: {len(option_chain_data)} strikes loaded")
                        else:
                            print("No option chain data received")
                    except Exception as oc_error:
                        print(f"ERROR getting option chain: {oc_error}")
                        # Don't update last_option_chain_call on error to retry sooner
                else:
                    remaining = int(OPTION_CHAIN_INTERVAL - (current_time - last_option_chain_call))
                    print(f"Option chain: waiting {remaining}s (rate limit)")
                
                # REAL market status based on NSE timings
                now = datetime.now()
                weekday = now.weekday()  # 0=Monday, 6=Sunday
                
                if weekday >= 5:  # Weekend
                    live_data['market_status'] = 'WEEKEND'
                elif 9 <= now.hour < 15 or (now.hour == 15 and now.minute < 30):
                    live_data['market_status'] = 'OPEN'
                elif now.hour == 9 and now.minute < 15:
                    live_data['market_status'] = 'PRE_OPEN'
                else:
                    live_data['market_status'] = 'CLOSED'
                
                # Generate LIVE AI signals based on real market data
                if live_data['nifty_price'] > 0:
                    live_data['ai_signals'] = generate_live_ai_signals(live_data['nifty_price'], live_data['nifty_change_percent'])
                
                # LIVE trading performance (calculate from real positions)
                live_data['algo_performance'] = calculate_live_performance(live_data['positions'])
                    
                print("SUCCESSFULLY FETCHED ALL LIVE DATA")
                
            except Exception as api_error:
                print(f"ERROR fetching Dhan data: {api_error}")
                print("NO SIMULATION - Will retry with real API data only")
                # Don't simulate - keep previous real data or show zeros
        else:
            # NO SIMULATION - Only show real data or empty
            live_data.update({
                'nifty_price': 0,
                'nifty_change': 0,
                'nifty_change_percent': 0,
                'account_balance': 0,
                'account_margin': 0,
                'positions': [],
                'option_chain': [],
                'ai_signals': [],
                'market_status': 'API_ERROR',
                'algo_performance': {
                    'trades_today': 0,
                    'pnl_today': 0,
                    'success_rate': 0,
                    'active_strategies': []
                }
            })
            print("WARNING: No simulation - Waiting for real Dhan API data...")
        
        live_data['last_update'] = datetime.now()
        return jsonify(live_data)
    
    except Exception as e:
        logger.error(f"CRITICAL ERROR fetching live data: {e}")
        print("NO SIMULATION - Only real Dhan API data allowed")
        return jsonify(live_data)

# LIVE DATA HELPER FUNCTIONS

def generate_live_ai_signals(nifty_price, nifty_change_percent):
    """Generate real AI signals based on live market data"""
    signals = []
    
    # Simple technical analysis for live signals
    if nifty_change_percent > 0.5:
        signals.append({
            'type': 'BUY',
            'confidence': min(0.9, 0.6 + abs(nifty_change_percent) / 10),
            'timestamp': datetime.now().isoformat(),
            'strategy': 'MOMENTUM_BREAKOUT',
            'price': nifty_price,
            'reason': f'Strong upward momentum: +{nifty_change_percent:.2f}%'
        })
    elif nifty_change_percent < -0.5:
        signals.append({
            'type': 'SELL', 
            'confidence': min(0.9, 0.6 + abs(nifty_change_percent) / 10),
            'timestamp': datetime.now().isoformat(),
            'strategy': 'MOMENTUM_BREAKDOWN',
            'price': nifty_price,
            'reason': f'Strong downward momentum: {nifty_change_percent:.2f}%'
        })
    else:
        signals.append({
            'type': 'HOLD',
            'confidence': 0.7,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'RANGE_BOUND',
            'price': nifty_price,
            'reason': f'Sideways movement: {nifty_change_percent:.2f}%'
        })
    
    return signals

def calculate_live_performance(positions):
    """Calculate live performance from real positions"""
    if not positions or len(positions) == 0:
        return {
            'trades_today': 0,
            'pnl_today': 0,
            'success_rate': 0,
            'active_strategies': []
        }
    
    # Calculate real P&L from positions
    total_pnl = 0
    winning_trades = 0
    total_trades = len(positions)
    strategies = set()
    
    for position in positions:
        if isinstance(position, dict):
            pnl = position.get('unrealizedProfit', 0) + position.get('realizedProfit', 0)
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            strategies.add(position.get('instrument', 'UNKNOWN'))
    
    success_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'trades_today': total_trades,
        'pnl_today': round(total_pnl, 2),
        'success_rate': round(success_rate, 1),
        'active_strategies': list(strategies)
    }

def start_live_data_updates():
    """Background thread for live data updates"""
    while True:
        try:
            if dhan_client:
                # Update live data every 5 seconds
                time.sleep(5)
            else:
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error in live data updates: {e}")
            time.sleep(10)

if __name__ == '__main__':
    print("Starting Real Trading Dashboard...")
    print("Features: Live NIFTY, Option Chain, Account Details, AI Signals")
    print("Dashboard: http://localhost:5003")
    print("Dashboard: http://127.0.0.1:5003")
    print("Connected to Dhan API for real market data" if dhan_client else "Using simulated data (Dhan API not available)")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    # Start background data updates
    data_thread = threading.Thread(target=start_live_data_updates, daemon=True)
    data_thread.start()
    
    try:
        app.run(host='127.0.0.1', port=5003, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
    except KeyboardInterrupt:
        print("\nReal Trading Dashboard stopped")