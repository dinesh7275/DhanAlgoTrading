// Dashboard JavaScript - CORRECTED VERSION

// Global variables
let dashboardData = {};
let updateInterval = null;
let isUpdating = false;

// Configuration
const CONFIG = {
    UPDATE_INTERVAL: 10000, // 10 seconds
    RETRY_DELAY: 5000,      // 5 seconds on error
    MAX_RETRIES: 3,
    API_TIMEOUT: 15000      // 15 seconds
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    initializeDashboard();
});

// Initialize dashboard
function initializeDashboard() {
    updateDashboard();
    startPeriodicUpdates();
    setupEventListeners();
}

// Setup event listeners
function setupEventListeners() {
    // Handle Enter key in credential inputs
    const accessTokenInput = document.getElementById('accessToken');
    const clientIdInput = document.getElementById('clientId');

    if (accessTokenInput) {
        accessTokenInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') setCredentials();
        });
    }

    if (clientIdInput) {
        clientIdInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') setCredentials();
        });
    }

    // Handle Enter key in trade inputs
    const tradeSymbolInput = document.getElementById('tradeSymbol');
    const tradeQuantityInput = document.getElementById('tradeQuantity');
    const tradePriceInput = document.getElementById('tradePrice');

    if (tradeSymbolInput) {
        tradeSymbolInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && tradeQuantityInput) tradeQuantityInput.focus();
        });
    }

    if (tradeQuantityInput) {
        tradeQuantityInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && tradePriceInput) tradePriceInput.focus();
        });
    }

    if (tradePriceInput) {
        tradePriceInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') executeTrade('BUY');
        });
    }
}

// Start periodic updates
function startPeriodicUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }

    updateInterval = setInterval(() => {
        if (!isUpdating) {
            updateDashboard();
        }
    }, CONFIG.UPDATE_INTERVAL);
}

// Update dashboard data with retry logic
async function updateDashboard(retryCount = 0) {
    if (isUpdating) return;

    isUpdating = true;

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);

        const response = await fetch('/api/dashboard', {
            signal: controller.signal,
            headers: {
                'Cache-Control': 'no-cache',
                'Accept': 'application/json'
            }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        dashboardData = await response.json();

        if (dashboardData.error) {
            console.error('Dashboard error:', dashboardData.error);
            showError('Dashboard Error', dashboardData.error);
        } else {
            updateUI();
            clearError();
        }

    } catch (error) {
        console.error('Error fetching dashboard data:', error);

        if (retryCount < CONFIG.MAX_RETRIES) {
            console.log(`Retrying... (${retryCount + 1}/${CONFIG.MAX_RETRIES})`);
            setTimeout(() => {
                updateDashboard(retryCount + 1);
            }, CONFIG.RETRY_DELAY);
        } else {
            showError('Connection Error', 'Failed to connect to server. Please check your connection.');
        }

    } finally {
        isUpdating = false;
    }
}

// Update UI elements
function updateUI() {
    try {
        updateStatusIndicators();
        updateMarketData();
        updateTechnicalAnalysis();
        updatePnLAndTrades();
        updateOptionChain();
        updateTradesTable();
        updateLastUpdateTime();
    } catch (error) {
        console.error('Error updating UI:', error);
        showError('UI Error', 'Error updating dashboard display');
    }
}

// Update status indicators
function updateStatusIndicators() {
    const connectionStatus = document.getElementById('connectionStatus');
    const apiStatusText = document.getElementById('apiStatusText');
    const botStatus = document.getElementById('botStatus');
    const botStatusText = document.getElementById('botStatusText');

    if (!connectionStatus || !apiStatusText || !botStatus || !botStatusText) {
        console.warn('Status elements not found in DOM');
        return;
    }

    // API Connection Status
    if (dashboardData.credentials_set) {
        connectionStatus.classList.add('connected');
        apiStatusText.textContent = 'Connected';
        apiStatusText.className = 'positive';
    } else {
        connectionStatus.classList.remove('connected');
        apiStatusText.textContent = 'Disconnected';
        apiStatusText.className = 'negative';
    }

    // Bot Status
    if (dashboardData.bot_status === 'Running') {
        botStatus.classList.add('connected');
        botStatusText.textContent = 'Running';
        botStatusText.className = 'positive';
    } else {
        botStatus.classList.remove('connected');
        botStatusText.textContent = 'Stopped';
        botStatusText.className = 'neutral';
    }
}

// Update market data
function updateMarketData() {
    if (!dashboardData.market_data) return;

    const marketData = dashboardData.market_data;

    // Update Nifty LTP
    const ltpElement = document.getElementById('niftyLtp');
    if (ltpElement) {
        const ltp = marketData.ltp;
        if (ltp && ltp > 0) {
            ltpElement.textContent = `₹${ltp.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            ltpElement.className = 'metric-value positive';
        } else {
            ltpElement.textContent = '--';
            ltpElement.className = 'metric-value neutral';
        }
    }

    // Update change
    const change = marketData.change || 0;
    const changeElement = document.getElementById('niftyChange');
    if (changeElement) {
        changeElement.textContent = `₹${change.toFixed(2)}`;
        changeElement.className = `metric-value ${change >= 0 ? 'positive' : 'negative'}`;
    }

    // Update time
    const timeElement = document.getElementById('niftyTime');
    if (timeElement) {
        timeElement.textContent = marketData.timestamp || '--';
    }
}

// Update technical analysis
function updateTechnicalAnalysis() {
    if (!dashboardData.technical_analysis) {
        resetTechnicalAnalysis();
        return;
    }

    const tech = dashboardData.technical_analysis;

    // Update signal box
    const signalBox = document.getElementById('technicalSignal');
    if (signalBox) {
        const signal = tech.signal || 'HOLD';
        signalBox.textContent = signal;
        signalBox.className = `signal-box signal-${signal.toLowerCase()}`;
    }

    // Update indicators
    const rsiElement = document.getElementById('rsiValue');
    if (rsiElement) {
        rsiElement.textContent = tech.rsi ? tech.rsi.toFixed(2) : '--';
    }

    const ema9Element = document.getElementById('ema9Value');
    if (ema9Element) {
        ema9Element.textContent = tech.ema9 ? tech.ema9.toFixed(2) : '--';
    }

    const ema21Element = document.getElementById('ema21Value');
    if (ema21Element) {
        ema21Element.textContent = tech.ema21 ? tech.ema21.toFixed(2) : '--';
    }

    // Update confidence
    const confidenceElement = document.getElementById('confidenceValue');
    if (confidenceElement) {
        if (tech.confidence !== undefined && tech.confidence !== null) {
            const confidence = (tech.confidence * 100).toFixed(1);
            confidenceElement.textContent = `${confidence}%`;
            confidenceElement.className = `metric-value ${confidence > 60 ? 'positive' : confidence > 40 ? 'neutral' : 'negative'}`;
        } else {
            confidenceElement.textContent = '--';
            confidenceElement.className = 'metric-value neutral';
        }
    }
}

// Reset technical analysis display
function resetTechnicalAnalysis() {
    const signalBox = document.getElementById('technicalSignal');
    if (signalBox) {
        signalBox.textContent = 'HOLD';
        signalBox.className = 'signal-box signal-hold';
    }

    const elements = ['rsiValue', 'ema9Value', 'ema21Value', 'confidenceValue'];
    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = '--';
            element.className = 'metric-value neutral';
        }
    });
}

// Update P&L and trade statistics
function updatePnLAndTrades() {
    // Update daily P&L
    const pnl = dashboardData.daily_pnl || 0;
    const pnlElement = document.getElementById('dailyPnl');
    if (pnlElement) {
        pnlElement.textContent = `₹${pnl.toFixed(2)}`;
        pnlElement.className = `metric-value ${pnl >= 0 ? 'positive' : 'negative'}`;
    }

    // Update trade count
    const tradeCount = dashboardData.trade_history ? dashboardData.trade_history.length : 0;
    const tradeCountElement = document.getElementById('totalTrades');
    if (tradeCountElement) {
        tradeCountElement.textContent = tradeCount;
    }
}

// Update option chain table
function updateOptionChain() {
    const table = document.getElementById('optionChainTable');
    if (!table) return;

    const tableBody = table.getElementsByTagName('tbody')[0];
    if (!tableBody) return;

    if (!dashboardData.option_chain || dashboardData.option_chain.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" class="loading">No option data available</td></tr>';
        return;
    }

    tableBody.innerHTML = '';

    dashboardData.option_chain.forEach(option => {
        const row = tableBody.insertRow();

        // Format numbers for display
        const callLtp = option.call_ltp ? option.call_ltp.toFixed(2) : '0.00';
        const putLtp = option.put_ltp ? option.put_ltp.toFixed(2) : '0.00';
        const callVolume = option.call_volume ? option.call_volume.toLocaleString() : '0';
        const putVolume = option.put_volume ? option.put_volume.toLocaleString() : '0';
        const callOI = option.call_oi ? option.call_oi.toLocaleString() : '0';
        const putOI = option.put_oi ? option.put_oi.toLocaleString() : '0';

        row.innerHTML = `
            <td>${callLtp}</td>
            <td>${callVolume}</td>
            <td>${callOI}</td>
            <td><strong>${option.strike}</strong></td>
            <td>${putLtp}</td>
            <td>${putVolume}</td>
            <td>${putOI}</td>
        `;

        // Add hover effect for strike price
        const strikeCell = row.cells[3];
        if (strikeCell) {
            strikeCell.style.cursor = 'pointer';
            strikeCell.onclick = () => fillTradeSymbol(option.strike);
        }
    });
}

// Update trades table
function updateTradesTable() {
    const table = document.getElementById('tradesTable');
    if (!table) return;

    const tableBody = table.getElementsByTagName('tbody')[0];
    if (!tableBody) return;

    if (!dashboardData.trade_history || dashboardData.trade_history.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="loading">No trades yet...</td></tr>';
        return;
    }

    tableBody.innerHTML = '';

    // Show last 10 trades in reverse order (newest first)
    const recentTrades = dashboardData.trade_history.slice(-10).reverse();

    recentTrades.forEach(trade => {
        const row = tableBody.insertRow();
        const time = formatTime(trade.timestamp);
        const price = trade.price ? `₹${trade.price.toFixed(2)}` : '--';

        row.innerHTML = `
            <td>${time}</td>
            <td title="${trade.symbol}">${truncateSymbol(trade.symbol)}</td>
            <td class="${trade.action.toLowerCase() === 'buy' ? 'positive' : 'negative'}">${trade.action}</td>
            <td>${trade.quantity}</td>
            <td>${price}</td>
            <td><span class="status-${trade.status.toLowerCase()}">${trade.status}</span></td>
        `;

        // Add click handler to copy symbol
        const symbolCell = row.cells[1];
        if (symbolCell) {
            symbolCell.style.cursor = 'pointer';
            symbolCell.onclick = () => copyToClipboard(trade.symbol);
        }
    });
}

// Update last update time
function updateLastUpdateTime() {
    const lastUpdate = document.getElementById('lastUpdate');
    if (lastUpdate) {
        lastUpdate.textContent = dashboardData.last_update || new Date().toLocaleTimeString();
    }
}

// API Functions

// Set API credentials
async function setCredentials() {
    const accessTokenInput = document.getElementById('accessToken');
    const clientIdInput = document.getElementById('clientId');
    const messageDiv = document.getElementById('connectionMessage');

    if (!accessTokenInput || !clientIdInput || !messageDiv) {
        console.error('Credential form elements not found');
        return;
    }

    const accessToken = accessTokenInput.value.trim();
    const clientId = clientIdInput.value.trim();

    if (!accessToken || !clientId) {
        showMessage(messageDiv, 'error', 'Please enter both Access Token and Client ID');
        return;
    }

    try {
        showMessage(messageDiv, 'loading', 'Connecting to API...');

        const response = await fetch('/api/set_credentials', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                access_token: accessToken,
                client_id: clientId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.success) {
            showMessage(messageDiv, 'success', '✅ API Connected Successfully!');
            // Clear the form for security
            setTimeout(() => {
                accessTokenInput.value = '';
                clientIdInput.value = '';
                messageDiv.innerHTML = '';
            }, 3000);

            // Force update dashboard
            setTimeout(() => updateDashboard(), 1000);
        } else {
            showMessage(messageDiv, 'error', `❌ ${result.message}`);
        }

    } catch (error) {
        console.error('Credentials error:', error);
        showMessage(messageDiv, 'error', '❌ Connection failed. Please try again.');
    }
}

// Start trading bot
async function startBot() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (!dashboardData.credentials_set) {
        alert('Please connect API credentials first');
        return;
    }

    if (!startBtn) return;

    try {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';

        const response = await fetch('/api/start_bot', {
            method: 'POST',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.success) {
            console.log('Bot started successfully');
            showNotification('🚀 Trading bot started successfully', 'success');
        } else {
            alert(`Failed to start bot: ${result.message}`);
        }

    } catch (error) {
        console.error('Start bot error:', error);
        alert('Error starting bot. Please try again.');
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = 'Start Bot';
    }
}

// Stop trading bot
async function stopBot() {
    const stopBtn = document.getElementById('stopBtn');

    if (!stopBtn) return;

    try {
        stopBtn.disabled = true;
        stopBtn.textContent = 'Stopping...';

        const response = await fetch('/api/stop_bot', {
            method: 'POST',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.success) {
            console.log('Bot stopped successfully');
            showNotification('⏹️ Trading bot stopped', 'neutral');
        } else {
            alert(`Failed to stop bot: ${result.message}`);
        }

    } catch (error) {
        console.error('Stop bot error:', error);
        alert('Error stopping bot. Please try again.');
    } finally {
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Bot';
    }
}

// Execute manual trade
async function executeTrade(action) {
    const symbolInput = document.getElementById('tradeSymbol');
    const quantityInput = document.getElementById('tradeQuantity');
    const priceInput = document.getElementById('tradePrice');
    const messageDiv = document.getElementById('tradeMessage');

    if (!symbolInput || !quantityInput || !messageDiv) {
        console.error('Trade form elements not found');
        return;
    }

    const symbol = symbolInput.value.trim();
    const quantity = quantityInput.value;
    const price = priceInput ? priceInput.value : '';

    // Validation
    if (!symbol) {
        showMessage(messageDiv, 'error', 'Please enter a symbol');
        return;
    }

    if (!quantity || quantity <= 0) {
        showMessage(messageDiv, 'error', 'Please enter a valid quantity');
        return;
    }

    if (!dashboardData.credentials_set) {
        showMessage(messageDiv, 'error', 'Please connect API credentials first');
        return;
    }

    try {
        showMessage(messageDiv, 'loading', `Placing ${action} order...`);

        const response = await fetch('/api/execute_trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol.toUpperCase(),
                action: action,
                quantity: parseInt(quantity),
                price: parseFloat(price) || 0
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.success) {
            showMessage(messageDiv, 'success', `✅ ${action} order placed: ${result.orderId}`);

            // Clear form partially
            symbolInput.value = '';
            if (priceInput) priceInput.value = '';

            // Update dashboard
            setTimeout(() => updateDashboard(), 2000);
        } else {
            showMessage(messageDiv, 'error', `❌ ${result.message}`);
        }

    } catch (error) {
        console.error('Trade execution error:', error);
        showMessage(messageDiv, 'error', '❌ Trade execution failed. Please try again.');
    }
}

// Utility Functions

// Format timestamp for display
function formatTime(timestamp) {
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-IN', {
            hour12: true,
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return timestamp;
    }
}

// Truncate symbol for table display
function truncateSymbol(symbol) {
    if (!symbol) return '--';
    return symbol.length > 15 ? symbol.substring(0, 15) + '...' : symbol;
}

// Fill trade symbol from option chain
function fillTradeSymbol(strike) {
    const symbolInput = document.getElementById('tradeSymbol');
    if (!symbolInput) return;

    const today = new Date();
    const nextThursday = getNextThursday(today);
    const expiry = nextThursday.toISOString().slice(2, 10).replace(/-/g, '');

    // Default to CE (Call)
    const symbol = `NIFTY${expiry}${Math.round(strike)}CE`;
    symbolInput.value = symbol;

    showNotification(`Symbol filled: ${symbol}`, 'success');
}

// Get next Thursday date
function getNextThursday(date) {
    const result = new Date(date);
    const daysUntilThursday = (4 - result.getDay()) % 7;
    if (daysUntilThursday === 0 && result.getHours() >= 15) {
        result.setDate(result.getDate() + 7);
    } else {
        result.setDate(result.getDate() + daysUntilThursday);
    }
    return result;
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
        }
        showNotification(`Copied: ${text}`, 'success');
    } catch (error) {
        console.error('Clipboard error:', error);
        showNotification('Failed to copy text', 'error');
    }
}

// Show message in a container
function showMessage(container, type, message) {
    if (!container) return;

    container.innerHTML = `<div class="${type}">${message}</div>`;

    if (type === 'success' || type === 'loading') {
        setTimeout(() => {
            if (container.innerHTML.includes(message)) {
                container.innerHTML = '';
            }
        }, 5000);
    }
}

// Show temporary notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '8px',
        color: 'white',
        fontWeight: 'bold',
        zIndex: '10000',
        maxWidth: '300px',
        wordWrap: 'break-word',
        opacity: '0',
        transition: 'opacity 0.3s ease',
        background: type === 'success' ? '#10b981' :
                   type === 'error' ? '#ef4444' :
                   type === 'neutral' ? '#f59e0b' : '#3b82f6'
    });

    // Add to page
    document.body.appendChild(notification);

    // Fade in
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 10);

    // Remove after delay
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Show global error
function showError(title, message) {
    console.error(`${title}: ${message}`);
    showNotification(`${title}: ${message}`, 'error');
}

// Clear global error
function clearError() {
    // Remove any existing error notifications
    const notifications = document.querySelectorAll('.notification-error');
    notifications.forEach(notification => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + R to refresh dashboard
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        updateDashboard();
        showNotification('Dashboard refreshed', 'success');
    }

    // Escape to clear forms
    if (e.key === 'Escape') {
        const symbolInput = document.getElementById('tradeSymbol');
        const priceInput = document.getElementById('tradePrice');
        const tradeMessage = document.getElementById('tradeMessage');
        const connectionMessage = document.getElementById('connectionMessage');

        if (symbolInput) symbolInput.value = '';
        if (priceInput) priceInput.value = '';
        if (tradeMessage) tradeMessage.innerHTML = '';
        if (connectionMessage) connectionMessage.innerHTML = '';
    }
});

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Pause updates when tab is not visible
        if (updateInterval) {
            clearInterval(updateInterval);
            updateInterval = null;
        }
    } else {
        // Resume updates when tab becomes visible
        updateDashboard();
        startPeriodicUpdates();
    }
});

// Handle connection lost/restored
window.addEventListener('online', function() {
    showNotification('Connection restored', 'success');
    updateDashboard();
});

window.addEventListener('offline', function() {
    showNotification('Connection lost', 'error');
});

// Handle errors globally
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showError('JavaScript Error', e.message);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showError('Promise Error', e.reason.toString());
});

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        updateDashboard,
        setCredentials,
        startBot,
        stopBot,
        executeTrade
    };
}