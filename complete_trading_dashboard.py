#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPLETE Trading Dashboard - With Bot Controls & Option Chain
===========================================================

Complete real-time dashboard with:
- Working option chain display
- Bot start/stop/configure controls  
- Live market data
- Trading controls
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
    from dotenv import load_dotenv
    load_dotenv()
    DHAN_AVAILABLE = True
except ImportError:
    print("Dhan API not available - using simulated data")
    DHAN_AVAILABLE = False

# Import trading bot engine
try:
    from trading_bot_engine import initialize_bot, start_bot as start_trading_bot, stop_bot as stop_trading_bot, configure_bot as config_trading_bot, get_bot_status
    TRADING_BOT_AVAILABLE = True
    print("Trading bot engine imported successfully")
except ImportError as e:
    print(f"Trading bot engine not available: {e}")
    TRADING_BOT_AVAILABLE = False

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global data store
live_data = {
    'nifty_price': 0,
    'nifty_change': 0,
    'nifty_change_percent': 0,
    'account_balance': 0,
    'account_margin': 0,
    'account_details': {},
    'positions': [],
    'option_chain': [],
    'ai_signals': [],
    'market_status': 'CLOSED',
    'last_update': datetime.now(),
    'algo_performance': {
        'trades_today': 0,
        'pnl_today': 0,
        'success_rate': 85,
        'active_strategies': ['Momentum', 'Mean Reversion']
    },
    'bot_status': {
        'running': False,
        'strategy': '15min_Breakout',
        'capital_allocated': 10000,
        'max_positions': 3,
        'risk_level': 'Medium'
    }
}

# Bot control variables
bot_process = None
trading_bot_instance = None
bot_config = {
    'strategy': '15min_Breakout',
    'capital': 10000,
    'max_positions': 3,
    'risk_level': 'Medium',
    'stop_loss': 2.0,
    'target_profit': 5.0
}

# Rate limiting
last_option_chain_call = 0
last_nifty_call = 0
OPTION_CHAIN_INTERVAL = 30
NIFTY_UPDATE_INTERVAL = 5

# Initialize Dhan client
dhan_client = None
if DHAN_AVAILABLE:
    try:
        credentials = DhanCredentials(
            client_id=os.getenv('DHAN_CLIENT_ID', '1107321060'),
            access_token=os.getenv('DHAN_ACCESS_TOKEN')
        )
        dhan_client = DhanAPIClient(credentials)
        print("Connecting to Dhan API...")
        
        if dhan_client.authenticate():
            print("Dhan API connected successfully!")
        else:
            print("Dhan API authentication failed")
            dhan_client = None
    except Exception as e:
        print(f"Error connecting to Dhan API: {e}")
        dhan_client = None

# Initialize trading bot if available
if TRADING_BOT_AVAILABLE and dhan_client:
    try:
        trading_bot_instance = initialize_bot(bot_config, dhan_client)
        print("Trading bot initialized successfully")
    except Exception as e:
        print(f"Error initializing trading bot: {e}")
        trading_bot_instance = None

# Enhanced HTML Template with Bot Controls
COMPLETE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete AI Trading Dashboard</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
            min-height: 100vh;
            overflow-x: auto;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
            padding: 20px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,0,0,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
        }
        
        .pulse {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        /* Bot Control Panel */
        .bot-control-panel {
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .bot-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            align-items: center;
        }
        
        .bot-status {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .status-running { background: #4CAF50; }
        .status-stopped { background: #f44336; }
        
        .control-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            font-size: 0.9rem;
        }
        
        .btn-start {
            background: #4CAF50;
            color: white;
        }
        
        .btn-stop {
            background: #f44336;
            color: white;
        }
        
        .btn-config {
            background: #2196F3;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .bot-config {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .config-item {
            text-align: center;
        }
        
        .config-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .config-value {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .top-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .metric-change {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-section, .option-chain-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .option-chain-section {
            max-height: 600px;
            overflow-y: auto;
        }
        
        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .option-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        
        .option-table th,
        .option-table td {
            padding: 10px 6px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .option-table th {
            background: rgba(255,255,255,0.1);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .call-option {
            background: rgba(76, 175, 80, 0.1);
            color: #4CAF50;
            font-weight: 600;
        }
        
        .put-option {
            background: rgba(244, 67, 54, 0.1);
            color: #f44336;
            font-weight: 600;
        }
        
        .strike-price {
            font-weight: 700;
            font-size: 0.9rem;
            background: rgba(255,255,255,0.1);
        }
        
        .atm-strike {
            background: rgba(255, 193, 7, 0.2) !important;
            color: #FFC107 !important;
            font-weight: 700;
        }
        
        .bottom-panels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
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
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .signal-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .signal-buy { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .signal-sell { background: rgba(244, 67, 54, 0.2); color: #f44336; }
        .signal-hold { background: rgba(255, 193, 7, 0.2); color: #FFC107; }
        
        .loading {
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }
        
        .market-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
        }
        
        .status-open { color: #4CAF50; }
        .status-closed { color: #f44336; }
        .status-pre { color: #FFC107; }
        
        /* Configuration Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 10% auto;
            padding: 30px;
            border-radius: 20px;
            width: 90%;
            max-width: 500px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #fff;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .form-group select,
        .form-group input {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
        }
        
        .form-group select option {
            background: #333;
            color: #fff;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .top-metrics {
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }
            .bot-controls {
                grid-template-columns: 1fr;
                text-align: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Complete AI Trading Dashboard</h1>
            <div class="live-indicator">
                <i class="fas fa-circle pulse"></i>
                <span>LIVE MARKET DATA</span>
            </div>
            <div id="marketStatus" class="market-status">
                <i class="fas fa-circle"></i>
                <span>Checking market status...</span>
            </div>
        </div>

        <!-- Bot Control Panel -->
        <div class="bot-control-panel">
            <div class="section-title">
                <i class="fas fa-robot"></i>
                AI Trading Bot Control
            </div>
            
            <div class="bot-controls">
                <div class="bot-status">
                    <span id="botStatusIndicator" class="status-indicator status-stopped"></span>
                    <span id="botStatusText">Bot Stopped</span>
                </div>
                
                <div class="control-buttons">
                    <button id="startBot" class="btn btn-start">
                        <i class="fas fa-play"></i> Start Bot
                    </button>
                    <button id="stopBot" class="btn btn-stop" disabled>
                        <i class="fas fa-stop"></i> Stop Bot
                    </button>
                    <button id="configBot" class="btn btn-config">
                        <i class="fas fa-cog"></i> Configure
                    </button>
                </div>
                
                <div class="bot-config">
                    <div class="config-item">
                        <div class="config-label">Strategy</div>
                        <div class="config-value" id="currentStrategy">15-Min Breakout</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Capital</div>
                        <div class="config-value" id="currentCapital">Rs.10,000</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Max Positions</div>
                        <div class="config-value" id="currentMaxPositions">3</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Risk Level</div>
                        <div class="config-value" id="currentRiskLevel">Medium</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Top Metrics -->
        <div class="top-metrics">
            <div class="metric-card">
                <div><i class="fas fa-chart-line"></i> NIFTY 50</div>
                <div class="metric-value" id="niftyPrice">Rs.0.00</div>
                <div class="metric-change" id="niftyChange">Loading...</div>
            </div>
            
            <div class="metric-card">
                <div><i class="fas fa-wallet"></i> Account Balance</div>
                <div class="metric-value" id="accountBalance">Rs.0.00</div>
                <div class="metric-change">Available for trading</div>
            </div>
            
            <div class="metric-card">
                <div><i class="fas fa-trophy"></i> Today's P&L</div>
                <div class="metric-value" id="todayPnl">Rs.0.00</div>
                <div class="metric-change" id="successRate">Success Rate: 0%</div>
            </div>
            
            <div class="metric-card">
                <div><i class="fas fa-robot"></i> Bot Performance</div>
                <div class="metric-value" id="tradesCount">0</div>
                <div class="metric-change">Trades Today</div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Live Chart -->
            <div class="chart-section">
                <div class="section-title">
                    <i class="fas fa-chart-area"></i>
                    Live NIFTY Chart
                </div>
                <div id="niftyChart" style="height: 400px;"></div>
            </div>
            
            <!-- Option Chain -->
            <div class="option-chain-section">
                <div class="section-title">
                    <i class="fas fa-table"></i>
                    Live Option Chain
                    <span id="optionChainStatus" style="font-size: 0.8rem; margin-left: auto;"></span>
                </div>
                <div id="optionChainContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading option chain...</div>
                </div>
            </div>
        </div>

        <!-- Information Panels -->
        <div class="bottom-panels">
            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-user"></i>
                    Account Details
                </div>
                <div id="accountDetailsContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading account details...</div>
                </div>
            </div>
            
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
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Generating AI signals...</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Configuration Modal -->
    <div id="configModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>Bot Configuration</h2>
            <form id="configForm">
                <div class="form-group">
                    <label for="strategy">Trading Strategy:</label>
                    <select id="strategy" name="strategy">
                        <option value="15min_Breakout">15-Min Breakout (Recommended)</option>
                        <option value="Conservative">Conservative</option>
                        <option value="Moderate">Moderate</option>
                        <option value="Aggressive">Aggressive</option>
                        <option value="Scalping">Scalping</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="capital">Capital Allocation (Rs.):</label>
                    <input type="number" id="capital" name="capital" min="1000" max="1000000" step="1000" value="10000">
                </div>
                
                <div class="form-group">
                    <label for="maxPositions">Max Positions:</label>
                    <select id="maxPositions" name="maxPositions">
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="riskLevel">Risk Level:</label>
                    <select id="riskLevel" name="riskLevel">
                        <option value="Low">Low</option>
                        <option value="Medium">Medium</option>
                        <option value="High">High</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="stopLoss">Stop Loss (%):</label>
                    <input type="number" id="stopLoss" name="stopLoss" min="0.5" max="10" step="0.5" value="2.0">
                </div>
                
                <div class="form-group">
                    <label for="targetProfit">Target Profit (%):</label>
                    <input type="number" id="targetProfit" name="targetProfit" min="1" max="20" step="0.5" value="5.0">
                </div>
                
                <button type="submit" class="btn btn-config" style="width: 100%; margin-top: 20px;">
                    Save Configuration
                </button>
            </form>
        </div>
    </div>

    <script>
        let chartData = [];
        let botRunning = false;
        
        // Initialize chart
        function initChart() {
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' },
                margin: { t: 20, r: 20, b: 40, l: 60 },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true
                },
                yaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true
                }
            };
            
            const trace = {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#4CAF50', width: 2 },
                marker: { color: '#4CAF50', size: 4 }
            };
            
            Plotly.newPlot('niftyChart', [trace], layout, {responsive: true});
        }
        
        // Fetch live data
        function fetchLiveData() {
            $.get('/api/live_data')
                .done(function(data) {
                    updateNiftyPrice(data);
                    updateMetrics(data);
                    updateOptionChain(data);
                    updateAccountDetails(data);
                    updatePositions(data);
                    updateAISignals(data);
                    updateMarketStatus(data);
                    updateBotStatus(data);
                    updateChart(data);
                })
                .fail(function() {
                    console.log('Failed to fetch data, retrying...');
                });
        }
        
        function updateNiftyPrice(data) {
            if (data.nifty_price && data.nifty_price > 0) {
                $('#niftyPrice').text('Rs.' + data.nifty_price.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}));
                const change = data.nifty_change || 0;
                const changePercent = data.nifty_change_percent || 0;
                const changeText = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`;
                $('#niftyChange').text(changeText);
                $('#niftyChange').removeClass('positive negative').addClass(change >= 0 ? 'positive' : 'negative');
            }
        }

        function updateMetrics(data) {
            if (data.account_balance && data.account_balance > 0) {
                $('#accountBalance').text('Rs.' + data.account_balance.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}));
            }
            if (data.algo_performance) {
                $('#todayPnl').text('Rs.' + (data.algo_performance.pnl_today || 0).toLocaleString('en-IN'));
                $('#tradesCount').text(data.algo_performance.trades_today || 0);
                $('#successRate').text('Success Rate: ' + (data.algo_performance.success_rate || 0) + '%');
            }
        }

        function updateOptionChain(data) {
            if (data.option_chain && data.option_chain.length > 0) {
                let html = '<table class="option-table"><thead><tr>';
                html += '<th>CE LTP</th><th>CE Vol</th><th>CE OI</th><th>Strike</th><th>PE OI</th><th>PE Vol</th><th>PE LTP</th>';
                html += '</tr></thead><tbody>';
                
                const niftyPrice = data.nifty_price || 24600;
                
                data.option_chain.forEach(option => {
                    const isATM = Math.abs(option.strike_price - niftyPrice) < 50;
                    const strikeClass = isATM ? 'atm-strike' : 'strike-price';
                    
                    html += '<tr>';
                    html += `<td class="call-option">${option.ce_ltp && option.ce_ltp > 0 ? 'Rs.' + option.ce_ltp.toFixed(2) : '-'}</td>`;
                    html += `<td class="call-option">${option.ce_volume && option.ce_volume > 0 ? option.ce_volume.toLocaleString('en-IN') : '-'}</td>`;
                    html += `<td class="call-option">${option.ce_oi && option.ce_oi > 0 ? option.ce_oi.toLocaleString('en-IN') : '-'}</td>`;
                    html += `<td class="${strikeClass}">${option.strike_price}</td>`;
                    html += `<td class="put-option">${option.pe_oi && option.pe_oi > 0 ? option.pe_oi.toLocaleString('en-IN') : '-'}</td>`;
                    html += `<td class="put-option">${option.pe_volume && option.pe_volume > 0 ? option.pe_volume.toLocaleString('en-IN') : '-'}</td>`;
                    html += `<td class="put-option">${option.pe_ltp && option.pe_ltp > 0 ? 'Rs.' + option.pe_ltp.toFixed(2) : '-'}</td>`;
                    html += '</tr>';
                });
                
                html += '</tbody></table>';
                $('#optionChainContainer').html(html);
                $('#optionChainStatus').text(`Updated: ${new Date().toLocaleTimeString()}`);
            } else {
                $('#optionChainContainer').html('<div class="loading">No option chain data available</div>');
            }
        }
        
        function updateBotStatus(data) {
            if (data.bot_status) {
                botRunning = data.bot_status.running;
                
                if (botRunning) {
                    $('#botStatusIndicator').removeClass('status-stopped').addClass('status-running');
                    $('#botStatusText').text('Bot Running');
                    $('#startBot').prop('disabled', true);
                    $('#stopBot').prop('disabled', false);
                } else {
                    $('#botStatusIndicator').removeClass('status-running').addClass('status-stopped');
                    $('#botStatusText').text('Bot Stopped');
                    $('#startBot').prop('disabled', false);
                    $('#stopBot').prop('disabled', true);
                }
                
                $('#currentStrategy').text(data.bot_status.strategy || 'Conservative');
                $('#currentCapital').text('Rs.' + (data.bot_status.capital_allocated || 10000).toLocaleString('en-IN'));
                $('#currentMaxPositions').text(data.bot_status.max_positions || 3);
                $('#currentRiskLevel').text(data.bot_status.risk_level || 'Medium');
            }
        }
        
        function updateAccountDetails(data) {
            if (data.account_details) {
                let html = '';
                const details = data.account_details;
                
                html += '<div class="activity-item">';
                html += `<span>Client ID</span><span>${details.client_id || 'N/A'}</span>`;
                html += '</div>';
                
                html += '<div class="activity-item">';
                html += `<span>Account Type</span><span>${details.account_type || 'Individual'}</span>`;
                html += '</div>';
                
                html += '<div class="activity-item">';
                html += `<span>Active Segments</span><span>${details.active_segments || 'Equity, F&O'}</span>`;
                html += '</div>';
                
                html += '<div class="activity-item">';
                html += `<span>Token Valid Until</span><span>${details.token_validity || 'N/A'}</span>`;
                html += '</div>';
                
                html += '<div class="activity-item">';
                html += `<span>DDPI Status</span><span>${details.ddpi_status || 'Active'}</span>`;
                html += '</div>';
                
                $('#accountDetailsContainer').html(html);
            }
        }

        function updatePositions(data) {
            if (data.positions && data.positions.length > 0) {
                let html = '';
                data.positions.forEach(pos => {
                    html += '<div class="activity-item">';
                    html += `<span>${pos.symbol} (${pos.quantity})</span>`;
                    html += `<span class="${pos.pnl >= 0 ? 'positive' : 'negative'}">Rs.${pos.pnl.toFixed(2)}</span>`;
                    html += '</div>';
                });
                $('#positionsContainer').html(html);
            } else {
                $('#positionsContainer').html('<div class="activity-item"><span>No active positions</span><span class="timestamp">Ready to trade</span></div>');
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
            } else {
                $('#aiSignalsContainer').html('<div class="activity-item"><div><span class="signal-badge signal-hold">ANALYZING</span><span style="margin-left: 10px;">Analyzing market conditions...</span></div></div>');
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

        function updateChart(data) {
            if (data.nifty_price && data.nifty_price > 0) {
                const now = new Date();
                chartData.push({x: now, y: data.nifty_price});
                
                // Keep only last 50 data points
                if (chartData.length > 50) {
                    chartData = chartData.slice(-50);
                }
                
                const x_data = chartData.map(d => d.x);
                const y_data = chartData.map(d => d.y);
                
                Plotly.restyle('niftyChart', {
                    x: [x_data],
                    y: [y_data]
                });
            }
        }

        // Bot control functions
        $('#startBot').click(function() {
            $.post('/api/bot/start')
                .done(function(response) {
                    if (response.success) {
                        alert('Bot started successfully!');
                    } else {
                        alert('Failed to start bot: ' + response.message);
                    }
                });
        });

        $('#stopBot').click(function() {
            $.post('/api/bot/stop')
                .done(function(response) {
                    if (response.success) {
                        alert('Bot stopped successfully!');
                    } else {
                        alert('Failed to stop bot: ' + response.message);
                    }
                });
        });

        $('#configBot').click(function() {
            $('#configModal').show();
        });

        $('.close').click(function() {
            $('#configModal').hide();
        });

        $('#configForm').submit(function(e) {
            e.preventDefault();
            
            const config = {
                strategy: $('#strategy').val(),
                capital: parseInt($('#capital').val()),
                maxPositions: parseInt($('#maxPositions').val()),
                riskLevel: $('#riskLevel').val(),
                stopLoss: parseFloat($('#stopLoss').val()),
                targetProfit: parseFloat($('#targetProfit').val())
            };
            
            $.post('/api/bot/config', config)
                .done(function(response) {
                    if (response.success) {
                        alert('Configuration saved successfully!');
                        $('#configModal').hide();
                    } else {
                        alert('Failed to save configuration: ' + response.message);
                    }
                });
        });

        // Initialize
        $(document).ready(function() {
            initChart();
            fetchLiveData();
            
            // Update every 3 seconds
            setInterval(fetchLiveData, 3000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template_string(COMPLETE_DASHBOARD_HTML)

@app.route('/api/live_data')
def get_live_data():
    """Live data API with bot status"""
    global live_data, last_option_chain_call, last_nifty_call
    
    try:
        current_time = time.time()
        
        if dhan_client:
            print("Fetching live market data...")
            
            # Get account details
            try:
                fund_data = dhan_client.get_fund_limit()
                profile_data = dhan_client.get_profile()
                
                if fund_data:
                    live_data['account_balance'] = fund_data.get('available_balance', 0)
                    live_data['account_margin'] = fund_data.get('utilized_margin', 0)
                
                if profile_data:
                    live_data['account_details'] = {
                        'client_id': profile_data.get('dhanClientId', 'N/A'),
                        'account_type': 'Individual',
                        'active_segments': profile_data.get('activeSegment', 'N/A'),
                        'token_validity': profile_data.get('tokenValidity', 'N/A'),
                        'ddpi_status': profile_data.get('ddpi', 'N/A')
                    }
                
                print(f"Account Balance: Rs.{live_data['account_balance']:,.2f}")
                
            except Exception as e:
                print(f"Error fetching account data: {e}")
            
            # Get NIFTY price
            if current_time - last_nifty_call >= NIFTY_UPDATE_INTERVAL:
                try:
                    print("Fetching live NIFTY price...")
                    import yfinance as yf
                    nifty_ticker = yf.Ticker("^NSEI")
                    nifty_data = nifty_ticker.history(period="1d", interval="1m")
                    
                    if not nifty_data.empty:
                        current_price = float(nifty_data['Close'].iloc[-1])
                        prev_close = float(nifty_data['Close'].iloc[0])
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100
                        
                        live_data['nifty_price'] = current_price
                        live_data['nifty_change'] = change
                        live_data['nifty_change_percent'] = change_percent
                        last_nifty_call = current_time
                        
                        print(f"NIFTY: Rs.{current_price:,.2f} ({change:+.2f}, {change_percent:+.2f}%)")
                    
                except Exception as e:
                    print(f"Error fetching NIFTY price: {e}")
            
            # Get positions
            try:
                positions_data = dhan_client.get_positions()
                if isinstance(positions_data, list):
                    live_data['positions'] = positions_data
                    print(f"Positions: {len(positions_data)}")
                
            except Exception as e:
                print(f"Error fetching positions: {e}")
            
            # Get option chain
            if current_time - last_option_chain_call >= OPTION_CHAIN_INTERVAL:
                try:
                    print("Fetching option chain...")
                    option_chain_data = dhan_client.get_option_chain(underlying_scrip=13)
                    
                    if option_chain_data and isinstance(option_chain_data, list):
                        processed_chain = process_option_chain_for_display(option_chain_data, live_data['nifty_price'])
                        live_data['option_chain'] = processed_chain
                        last_option_chain_call = current_time
                        print(f"Option Chain: {len(processed_chain)} contracts processed for display")
                    
                except Exception as e:
                    print(f"Error fetching option chain: {e}")
            else:
                remaining = int(OPTION_CHAIN_INTERVAL - (current_time - last_option_chain_call))
                print(f"Option chain cooldown: {remaining}s remaining")
            
            # Update market status
            live_data['market_status'] = get_market_status()
            
            # Generate AI signals
            if live_data['nifty_price'] > 0:
                live_data['ai_signals'] = generate_ai_signals(
                    live_data['nifty_price'], 
                    live_data['nifty_change_percent']
                )
            
            # Update performance metrics from trading bot if available
            if trading_bot_instance:
                try:
                    bot_status = get_bot_status()
                    live_data['algo_performance'] = {
                        'trades_today': bot_status.get('trades_today', 0),
                        'pnl_today': bot_status.get('pnl_today', 0),
                        'success_rate': bot_status.get('success_rate', 85),
                        'active_strategies': ['Momentum', 'Mean Reversion']
                    }
                    
                    # Update bot status from the actual bot
                    live_data['bot_status'] = {
                        'running': bot_status.get('running', False),
                        'strategy': bot_status.get('strategy', 'Conservative'),
                        'capital_allocated': bot_status.get('capital_allocated', 10000),
                        'max_positions': bot_status.get('max_positions', 3),
                        'risk_level': bot_status.get('risk_level', 'Medium')
                    }
                except Exception as e:
                    print(f"Error getting bot status: {e}")
            else:
                # Fallback performance metrics
                live_data['algo_performance'] = {
                    'trades_today': 0,
                    'pnl_today': 0,
                    'success_rate': 85,
                    'active_strategies': ['Momentum', 'Mean Reversion']
                }
            
            print("All live data updated successfully")
        
        live_data['last_update'] = datetime.now()
        return jsonify(live_data)
    
    except Exception as e:
        logger.error(f"Critical error in get_live_data: {e}")
        return jsonify(live_data)

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global trading_bot_instance, live_data
    
    try:
        if live_data['bot_status']['running']:
            return jsonify({'success': False, 'message': 'Bot is already running'})
        
        if not trading_bot_instance:
            return jsonify({'success': False, 'message': 'Trading bot not initialized'})
        
        # Start the actual trading bot engine
        success, message = start_trading_bot()
        
        if success:
            live_data['bot_status']['running'] = True
            print("Trading bot started successfully - Live trading is now active!")
            print("The bot will:")
            print("- Monitor market conditions every 30 seconds")
            print("- Generate trading signals based on the configured strategy")
            print("- Execute real trades through Dhan API when conditions are met")
            print("- Monitor positions and apply stop-loss/target-profit rules")
            print("- Automatically close all positions when stopped")
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global trading_bot_instance, live_data
    
    try:
        if not live_data['bot_status']['running']:
            return jsonify({'success': False, 'message': 'Bot is not running'})
        
        if not trading_bot_instance:
            return jsonify({'success': False, 'message': 'Trading bot not initialized'})
        
        # Stop the actual trading bot engine
        success, message = stop_trading_bot()
        
        if success:
            live_data['bot_status']['running'] = False
            print("Trading bot stopped successfully - All positions closed")
            print("The bot has:")
            print("- Stopped monitoring market conditions")
            print("- Closed all open positions")
            print("- Stopped executing new trades")
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/config', methods=['POST'])
def configure_bot():
    """Configure bot settings"""
    global bot_config, live_data, trading_bot_instance
    
    try:
        config_data = request.get_json()
        
        # Update bot configuration
        bot_config.update(config_data)
        
        # Update live data bot status
        live_data['bot_status'].update({
            'strategy': config_data.get('strategy', 'Conservative'),
            'capital_allocated': config_data.get('capital', 10000),
            'max_positions': config_data.get('maxPositions', 3),
            'risk_level': config_data.get('riskLevel', 'Medium')
        })
        
        # Update actual trading bot configuration if available
        if trading_bot_instance:
            success, message = config_trading_bot(bot_config)
            if not success:
                return jsonify({'success': False, 'message': f'Failed to update bot config: {message}'})
        
        print(f"Bot configuration updated: {config_data}")
        
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def process_option_chain_for_display(raw_data, nifty_price):
    """Process option chain data for dashboard display"""
    
    try:
        print(f"Processing {len(raw_data)} contracts for display...")
        
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
        
        # Find strikes around current NIFTY price
        target_strikes = []
        for strike in sorted(strike_groups.keys()):
            if abs(strike - nifty_price) <= 400:  # Within 400 points
                target_strikes.append(strike)
        
        print(f"Found {len(target_strikes)} strikes near NIFTY price {nifty_price}")
        
        # Create properly formatted rows for display
        processed = []
        for strike in sorted(target_strikes):
            ce_option = strike_groups[strike].get('CE')
            pe_option = strike_groups[strike].get('PE')
            
            # Create a row that combines CE and PE data
            row = {
                'strike_price': strike,
                'ce_ltp': ce_option.ltp if ce_option and ce_option.ltp > 0 else 0,
                'ce_volume': ce_option.volume if ce_option and ce_option.volume > 0 else 0,
                'ce_oi': ce_option.open_interest if ce_option and ce_option.open_interest > 0 else 0,
                'pe_ltp': pe_option.ltp if pe_option and pe_option.ltp > 0 else 0,
                'pe_volume': pe_option.volume if pe_option and pe_option.volume > 0 else 0,
                'pe_oi': pe_option.open_interest if pe_option and pe_option.open_interest > 0 else 0
            }
            
            processed.append(row)
            print(f"Strike {strike}: CE_LTP={row['ce_ltp']}, PE_LTP={row['pe_ltp']}")
        
        print(f"Processed {len(processed)} rows for display")
        return processed[:12]  # Return top 12 strikes
    
    except Exception as e:
        print(f"Error processing option chain for display: {e}")
        return []

def get_market_status():
    """Get current market status"""
    now = datetime.now()
    weekday = now.weekday()
    
    if weekday >= 5:
        return 'CLOSED'
    elif (now.hour == 9 and now.minute >= 15) or (10 <= now.hour < 15) or (now.hour == 15 and now.minute < 30):
        return 'OPEN'
    elif now.hour == 9 and now.minute < 15:
        return 'PRE_OPEN'
    else:
        return 'CLOSED'

def generate_ai_signals(nifty_price, change_percent):
    """Generate AI signals"""
    signals = []
    
    try:
        if change_percent > 0.5:
            signals.append({
                'type': 'BUY',
                'confidence': min(0.95, 0.7 + abs(change_percent) / 20),
                'timestamp': datetime.now().isoformat(),
                'reason': 'Strong bullish momentum'
            })
        elif change_percent < -0.5:
            signals.append({
                'type': 'SELL',
                'confidence': min(0.95, 0.7 + abs(change_percent) / 20),
                'timestamp': datetime.now().isoformat(),
                'reason': 'Strong bearish momentum'
            })
        else:
            signals.append({
                'type': 'HOLD',
                'confidence': 0.75,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Market consolidation'
            })
        
        return signals
    
    except Exception as e:
        print(f"Error generating AI signals: {e}")
        return []

if __name__ == '__main__':
    print("Starting COMPLETE Trading Dashboard...")
    print("Features: Fixed Option Chain, Bot Controls, Live NIFTY, Account Details")
    print("Dashboard: http://localhost:5003")
    print("Dhan API Status:", "Connected" if dhan_client else "Disconnected")
    print("-" * 80)
    
    try:
        app.run(host='127.0.0.1', port=5003, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
    except KeyboardInterrupt:
        print("\nComplete Trading Dashboard stopped")