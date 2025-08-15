/* Dashboard JavaScript for Dhan Algorithmic Trading */

// Global state management
const Dashboard = {
    state: {
        isTrading: false,
        autoRefresh: true,
        refreshInterval: 5000,
        currentView: 'overview',
        websocket: null,
        notifications: []
    },
    
    // Initialize dashboard
    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.startAutoRefresh();
        this.setupWebSocket();
        console.log('Dashboard initialized');
    },
    
    // Setup event listeners
    setupEventListeners() {
        // Trading controls
        const startTradingBtn = document.getElementById('start-trading');
        const stopTradingBtn = document.getElementById('stop-trading');
        
        if (startTradingBtn) {
            startTradingBtn.addEventListener('click', () => this.startTrading());
        }
        
        if (stopTradingBtn) {
            stopTradingBtn.addEventListener('click', () => this.stopTrading());
        }
        
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                this.state.autoRefresh = e.target.checked;
                if (this.state.autoRefresh) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }
        
        // Manual refresh button
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshDashboard());
        }
        
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchView(e.target.dataset.view);
            });
        });
        
        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => this.closeModal());
        });
        
        // Settings form
        const settingsForm = document.getElementById('settings-form');
        if (settingsForm) {
            settingsForm.addEventListener('submit', (e) => this.saveSettings(e));
        }
    },
    
    // Load initial data
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadPortfolioData(),
                this.loadTradingStatistics(),
                this.loadMarketOverview(),
                this.loadRecentTrades(),
                this.loadSystemStatus()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Error loading dashboard data', 'error');
        }
    },
    
    // Load portfolio data
    async loadPortfolioData() {
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();
            
            if (data.success) {
                this.updatePortfolioDisplay(data.portfolio);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading portfolio:', error);
        }
    },
    
    // Load trading statistics
    async loadTradingStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatisticsDisplay(data.statistics);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    },
    
    // Load market overview
    async loadMarketOverview() {
        try {
            const response = await fetch('/api/market-overview');
            const data = await response.json();
            
            if (data.success) {
                this.updateMarketDisplay(data.market);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading market data:', error);
        }
    },
    
    // Load recent trades
    async loadRecentTrades() {
        try {
            const response = await fetch('/api/trades/recent');
            const data = await response.json();
            
            if (data.success) {
                this.updateTradesDisplay(data.trades);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading trades:', error);
        }
    },
    
    // Load system status
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            if (data.success) {
                this.updateSystemStatus(data.status);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading system status:', error);
        }
    },
    
    // Update portfolio display
    updatePortfolioDisplay(portfolio) {
        // Update portfolio value
        const portfolioValue = document.getElementById('portfolio-value');
        if (portfolioValue) {
            portfolioValue.textContent = this.formatCurrency(portfolio.total_value);
        }
        
        // Update P&L
        const pnlElement = document.getElementById('total-pnl');
        if (pnlElement) {
            const pnl = portfolio.total_pnl || 0;
            pnlElement.textContent = this.formatCurrency(pnl);
            pnlElement.className = pnl >= 0 ? 'stat-value positive' : 'stat-value negative';
        }
        
        // Update positions count
        const positionsCount = document.getElementById('positions-count');
        if (positionsCount) {
            positionsCount.textContent = portfolio.positions?.length || 0;
        }
        
        // Update positions table
        this.updatePositionsTable(portfolio.positions || []);
    },
    
    // Update statistics display
    updateStatisticsDisplay(stats) {
        // Update total trades
        const totalTrades = document.getElementById('total-trades');
        if (totalTrades) {
            totalTrades.textContent = stats.total_trades || 0;
        }
        
        // Update win rate
        const winRate = document.getElementById('win-rate');
        if (winRate) {
            winRate.textContent = `${(stats.win_rate || 0).toFixed(1)}%`;
        }
        
        // Update today's trades
        const todayTrades = document.getElementById('today-trades');
        if (todayTrades) {
            todayTrades.textContent = stats.today_trades || 0;
        }
    },
    
    // Update market display
    updateMarketDisplay(market) {
        // Update Nifty 50
        const niftyPrice = document.getElementById('nifty-price');
        const niftyChange = document.getElementById('nifty-change');
        
        if (niftyPrice && market.nifty_50) {
            niftyPrice.textContent = market.nifty_50.current.toLocaleString();
        }
        
        if (niftyChange && market.nifty_50) {
            const change = market.nifty_50.change;
            const changePercent = market.nifty_50.change_percent;
            niftyChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`;
            niftyChange.className = change >= 0 ? 'market-change positive' : 'market-change negative';
        }
        
        // Update VIX
        const vixValue = document.getElementById('vix-value');
        if (vixValue && market.india_vix) {
            vixValue.textContent = market.india_vix.current.toFixed(2);
        }
        
        // Update market status
        const marketStatus = document.getElementById('market-status');
        if (marketStatus) {
            marketStatus.textContent = market.market_status;
            marketStatus.className = `status ${market.market_status === 'OPEN' ? 'status-active' : 'status-inactive'}`;
        }
    },
    
    // Update trades display
    updateTradesDisplay(trades) {
        const tradesTable = document.getElementById('trades-table-body');
        if (!tradesTable) return;
        
        tradesTable.innerHTML = '';
        
        trades.forEach(trade => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
                <td>${trade.symbol}</td>
                <td class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">${trade.action}</td>
                <td>${trade.quantity}</td>
                <td>${this.formatCurrency(trade.price)}</td>
                <td class="${(trade.pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                    ${this.formatCurrency(trade.pnl || 0)}
                </td>
                <td><span class="status ${trade.status.toLowerCase()}">${trade.status}</span></td>
            `;
            tradesTable.appendChild(row);
        });
    },
    
    // Update positions table
    updatePositionsTable(positions) {
        const positionsTable = document.getElementById('positions-table-body');
        if (!positionsTable) return;
        
        positionsTable.innerHTML = '';
        
        positions.forEach(position => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${position.symbol}</td>
                <td>${position.quantity}</td>
                <td>${this.formatCurrency(position.avg_price)}</td>
                <td>${this.formatCurrency(position.current_price)}</td>
                <td class="${position.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                    ${this.formatCurrency(position.unrealized_pnl)}
                </td>
                <td class="${position.pnl_percentage >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                    ${position.pnl_percentage.toFixed(2)}%
                </td>
            `;
            positionsTable.appendChild(row);
        });
    },
    
    // Update system status
    updateSystemStatus(status) {
        const systemStatus = document.getElementById('system-status');
        if (systemStatus) {
            systemStatus.textContent = status.status;
            systemStatus.className = `status ${status.status === 'ACTIVE' ? 'status-active' : 'status-inactive'}`;
        }
        
        // Update component statuses
        Object.entries(status.components || {}).forEach(([component, state]) => {
            const element = document.getElementById(`${component}-status`);
            if (element) {
                element.textContent = state;
                element.className = `status ${state === 'ACTIVE' || state === 'CONNECTED' ? 'status-active' : 'status-inactive'}`;
            }
        });
        
        // Update trading state
        this.state.isTrading = status.status === 'ACTIVE';
        this.updateTradingControls();
    },
    
    // Start trading
    async startTrading() {
        try {
            const response = await fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.state.isTrading = true;
                this.updateTradingControls();
                this.showNotification('Trading started successfully', 'success');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            this.showNotification('Failed to start trading', 'error');
        }
    },
    
    // Stop trading
    async stopTrading() {
        try {
            const response = await fetch('/api/trading/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.state.isTrading = false;
                this.updateTradingControls();
                this.showNotification('Trading stopped successfully', 'success');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error stopping trading:', error);
            this.showNotification('Failed to stop trading', 'error');
        }
    },
    
    // Update trading controls
    updateTradingControls() {
        const startBtn = document.getElementById('start-trading');
        const stopBtn = document.getElementById('stop-trading');
        
        if (startBtn) {
            startBtn.disabled = this.state.isTrading;
        }
        
        if (stopBtn) {
            stopBtn.disabled = !this.state.isTrading;
        }
    },
    
    // Switch view
    switchView(view) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeNavItem = document.querySelector(`[data-view="${view}"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
        }
        
        // Update content
        document.querySelectorAll('.view-content').forEach(content => {
            content.style.display = 'none';
        });
        
        const activeContent = document.getElementById(`${view}-view`);
        if (activeContent) {
            activeContent.style.display = 'block';
        }
        
        this.state.currentView = view;
    },
    
    // Auto refresh functionality
    startAutoRefresh() {
        if (this.refreshIntervalId) {
            clearInterval(this.refreshIntervalId);
        }
        
        this.refreshIntervalId = setInterval(() => {
            if (this.state.autoRefresh) {
                this.refreshDashboard();
            }
        }, this.state.refreshInterval);
    },
    
    stopAutoRefresh() {
        if (this.refreshIntervalId) {
            clearInterval(this.refreshIntervalId);
            this.refreshIntervalId = null;
        }
    },
    
    // Refresh dashboard data
    async refreshDashboard() {
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<span class="spinner"></span> Refreshing...';
        }
        
        try {
            await this.loadInitialData();
            this.showNotification('Dashboard data refreshed', 'success');
        } catch (error) {
            this.showNotification('Failed to refresh data', 'error');
        } finally {
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = 'Refresh';
            }
        }
    },
    
    // WebSocket setup
    setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.state.websocket = new WebSocket(wsUrl);
            
            this.state.websocket.onopen = () => {
                console.log('WebSocket connected');
            };
            
            this.state.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.state.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
            this.state.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Error setting up WebSocket:', error);
        }
    },
    
    // Handle WebSocket messages
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'portfolio_update':
                this.updatePortfolioDisplay(data.data);
                break;
            case 'trade_executed':
                this.loadRecentTrades();
                this.showNotification(`Trade executed: ${data.data.action} ${data.data.symbol}`, 'info');
                break;
            case 'market_update':
                this.updateMarketDisplay(data.data);
                break;
            case 'system_alert':
                this.showNotification(data.message, data.level);
                break;
            default:
                console.log('Unknown WebSocket message:', data);
        }
    },
    
    // Show notification
    showNotification(message, type = 'info') {
        const notification = {
            id: Date.now(),
            message,
            type,
            timestamp: new Date()
        };
        
        this.state.notifications.push(notification);
        this.displayNotification(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            this.removeNotification(notification.id);
        }, 5000);
    },
    
    // Display notification
    displayNotification(notification) {
        const container = document.getElementById('notifications-container') || this.createNotificationContainer();
        
        const notificationElement = document.createElement('div');
        notificationElement.className = `alert alert-${notification.type} notification`;
        notificationElement.id = `notification-${notification.id}`;
        notificationElement.innerHTML = `
            <span>${notification.message}</span>
            <button type="button" class="notification-close" onclick="Dashboard.removeNotification(${notification.id})">×</button>
        `;
        
        container.appendChild(notificationElement);
        
        // Add animation
        setTimeout(() => {
            notificationElement.classList.add('show');
        }, 10);
    },
    
    // Create notification container
    createNotificationContainer() {
        const container = document.createElement('div');
        container.id = 'notifications-container';
        container.className = 'notifications-container';
        document.body.appendChild(container);
        return container;
    },
    
    // Remove notification
    removeNotification(id) {
        const notification = document.getElementById(`notification-${id}`);
        if (notification) {
            notification.classList.add('hide');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
        
        this.state.notifications = this.state.notifications.filter(n => n.id !== id);
    },
    
    // Save settings
    async saveSettings(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const settings = {};
        
        for (let [key, value] of formData.entries()) {
            settings[key] = value;
        }
        
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Settings saved successfully', 'success');
                this.closeModal();
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showNotification('Failed to save settings', 'error');
        }
    },
    
    // Close modal
    closeModal() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    },
    
    // Utility functions
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 2
        }).format(amount);
    },
    
    formatNumber(number) {
        return new Intl.NumberFormat('en-IN').format(number);
    },
    
    formatPercentage(value) {
        return `${value.toFixed(2)}%`;
    }
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    Dashboard.init();
});

// Export for global access
window.Dashboard = Dashboard;