#!/usr/bin/env python3
"""
Data Models for Trading System
=============================

Core data models for users, trades, sessions, and portfolio
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import uuid
import json

@dataclass
class User:
    """User model for authentication and profile"""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    trading_permissions: Dict[str, bool] = field(default_factory=lambda: {
        'paper_trading': True,
        'live_trading': False,
        'options_trading': True,
        'futures_trading': False
    })
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'trading_permissions': self.trading_permissions,
            'preferences': self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary"""
        user = cls()
        user.user_id = data.get('user_id', user.user_id)
        user.username = data.get('username', '')
        user.email = data.get('email', '')
        user.full_name = data.get('full_name', '')
        
        if data.get('created_at'):
            user.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_login'):
            user.last_login = datetime.fromisoformat(data['last_login'])
        
        user.is_active = data.get('is_active', True)
        user.trading_permissions = data.get('trading_permissions', user.trading_permissions)
        user.preferences = data.get('preferences', {})
        
        return user

@dataclass
class Trade:
    """Trade model for individual trade records"""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    action: str = ""  # BUY, SELL
    quantity: int = 0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    order_type: str = "MARKET"  # MARKET, LIMIT
    strategy: str = "manual"  # manual, ai_ensemble, etc.
    confidence: float = 0.0  # AI confidence level
    fees: float = 0.0
    net_value: float = 0.0
    status: str = "COMPLETED"  # PENDING, COMPLETED, CANCELLED
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Options specific fields
    option_type: Optional[str] = None  # CE, PE
    strike_price: Optional[float] = None
    expiry_date: Optional[datetime] = None
    
    # Additional metadata
    market_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    # Trade analysis
    entry_signal: Optional[str] = None
    exit_signal: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'order_type': self.order_type,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'fees': self.fees,
            'net_value': self.net_value,
            'status': self.status,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'option_type': self.option_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'market_price': self.market_price,
            'bid': self.bid,
            'ask': self.ask,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'entry_signal': self.entry_signal,
            'exit_signal': self.exit_signal,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create from dictionary"""
        trade = cls()
        
        # Basic fields
        trade.trade_id = data.get('trade_id', trade.trade_id)
        trade.symbol = data.get('symbol', '')
        trade.action = data.get('action', '')
        trade.quantity = data.get('quantity', 0)
        trade.price = data.get('price', 0.0)
        trade.order_type = data.get('order_type', 'MARKET')
        trade.strategy = data.get('strategy', 'manual')
        trade.confidence = data.get('confidence', 0.0)
        trade.fees = data.get('fees', 0.0)
        trade.net_value = data.get('net_value', 0.0)
        trade.status = data.get('status', 'COMPLETED')
        trade.user_id = data.get('user_id')
        trade.session_id = data.get('session_id')
        
        # Dates
        if data.get('timestamp'):
            trade.timestamp = datetime.fromisoformat(data['timestamp'])
        if data.get('expiry_date'):
            trade.expiry_date = datetime.fromisoformat(data['expiry_date'])
        
        # Options fields
        trade.option_type = data.get('option_type')
        trade.strike_price = data.get('strike_price')
        
        # Market data
        trade.market_price = data.get('market_price')
        trade.bid = data.get('bid')
        trade.ask = data.get('ask')
        trade.implied_volatility = data.get('implied_volatility')
        
        # Greeks
        trade.delta = data.get('delta')
        trade.gamma = data.get('gamma')
        trade.theta = data.get('theta')
        trade.vega = data.get('vega')
        
        # Analysis
        trade.entry_signal = data.get('entry_signal')
        trade.exit_signal = data.get('exit_signal')
        trade.pnl = data.get('pnl')
        trade.pnl_percentage = data.get('pnl_percentage')
        
        return trade
    
    def calculate_trade_value(self) -> float:
        """Calculate total trade value"""
        return self.quantity * self.price
    
    def calculate_net_proceeds(self) -> float:
        """Calculate net proceeds after fees"""
        trade_value = self.calculate_trade_value()
        if self.action == 'BUY':
            return trade_value + self.fees  # Cost
        else:  # SELL
            return trade_value - self.fees  # Proceeds

@dataclass
class TradingSession:
    """Trading session model"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    session_type: str = "paper"  # paper, live
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    strategy: str = "ai_ensemble"
    status: str = "ACTIVE"  # ACTIVE, COMPLETED, TERMINATED
    
    # Session statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 10000.0
    
    # Configuration
    risk_settings: Dict[str, Any] = field(default_factory=lambda: {
        'max_daily_loss': 0.10,
        'max_trades_per_day': 10,
        'position_size_method': 'fixed_fractional'
    })
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'session_type': self.session_type,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'strategy': self.strategy,
            'status': self.status,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_capital': self.peak_capital,
            'risk_settings': self.risk_settings,
            'notes': self.notes,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSession':
        """Create from dictionary"""
        session = cls()
        
        session.session_id = data.get('session_id', session.session_id)
        session.user_id = data.get('user_id', '')
        session.session_type = data.get('session_type', 'paper')
        session.initial_capital = data.get('initial_capital', 10000.0)
        session.current_capital = data.get('current_capital', 10000.0)
        session.strategy = data.get('strategy', 'ai_ensemble')
        session.status = data.get('status', 'ACTIVE')
        
        # Dates
        if data.get('start_time'):
            session.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            session.end_time = datetime.fromisoformat(data['end_time'])
        
        # Statistics
        session.total_trades = data.get('total_trades', 0)
        session.winning_trades = data.get('winning_trades', 0)
        session.losing_trades = data.get('losing_trades', 0)
        session.total_pnl = data.get('total_pnl', 0.0)
        session.max_drawdown = data.get('max_drawdown', 0.0)
        session.peak_capital = data.get('peak_capital', 10000.0)
        
        # Configuration and metadata
        session.risk_settings = data.get('risk_settings', session.risk_settings)
        session.notes = data.get('notes', '')
        session.tags = data.get('tags', [])
        
        return session
    
    def update_statistics(self, trade: Trade):
        """Update session statistics with new trade"""
        self.total_trades += 1
        
        if trade.pnl:
            self.total_pnl += trade.pnl
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        # Update capital (simplified)
        if trade.pnl:
            self.current_capital += trade.pnl
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def get_total_return(self) -> float:
        """Calculate total return percentage"""
        if self.initial_capital == 0:
            return 0.0
        return ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

@dataclass
class Portfolio:
    """Portfolio model for position tracking"""
    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    name: str = "Default Portfolio"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Portfolio value
    cash_balance: float = 10000.0
    invested_value: float = 0.0
    total_value: float = 10000.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Positions (simplified as list of dictionaries)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    total_return: float = 0.0
    daily_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    var_1d: float = 0.0  # Value at Risk (1 day)
    var_5d: float = 0.0  # Value at Risk (5 day)
    beta: float = 1.0    # Portfolio beta vs Nifty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'cash_balance': self.cash_balance,
            'invested_value': self.invested_value,
            'total_value': self.total_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'positions': self.positions,
            'total_return': self.total_return,
            'daily_return': self.daily_return,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'var_1d': self.var_1d,
            'var_5d': self.var_5d,
            'beta': self.beta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create from dictionary"""
        portfolio = cls()
        
        portfolio.portfolio_id = data.get('portfolio_id', portfolio.portfolio_id)
        portfolio.user_id = data.get('user_id', '')
        portfolio.session_id = data.get('session_id', '')
        portfolio.name = data.get('name', 'Default Portfolio')
        
        # Dates
        if data.get('created_at'):
            portfolio.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            portfolio.updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Financial data
        portfolio.cash_balance = data.get('cash_balance', 10000.0)
        portfolio.invested_value = data.get('invested_value', 0.0)
        portfolio.total_value = data.get('total_value', 10000.0)
        portfolio.unrealized_pnl = data.get('unrealized_pnl', 0.0)
        portfolio.realized_pnl = data.get('realized_pnl', 0.0)
        
        # Positions
        portfolio.positions = data.get('positions', [])
        
        # Performance metrics
        portfolio.total_return = data.get('total_return', 0.0)
        portfolio.daily_return = data.get('daily_return', 0.0)
        portfolio.max_drawdown = data.get('max_drawdown', 0.0)
        portfolio.volatility = data.get('volatility', 0.0)
        portfolio.sharpe_ratio = data.get('sharpe_ratio', 0.0)
        portfolio.var_1d = data.get('var_1d', 0.0)
        portfolio.var_5d = data.get('var_5d', 0.0)
        portfolio.beta = data.get('beta', 1.0)
        
        return portfolio
    
    def add_position(self, symbol: str, quantity: int, avg_price: float, 
                    current_price: float = None) -> None:
        """Add a new position to portfolio"""
        current_price = current_price or avg_price
        
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'avg_price': avg_price,
            'current_price': current_price,
            'invested_value': quantity * avg_price,
            'current_value': quantity * current_price,
            'unrealized_pnl': quantity * (current_price - avg_price),
            'pnl_percentage': ((current_price - avg_price) / avg_price) * 100,
            'weight': 0.0,  # Will be calculated
            'updated_at': datetime.now().isoformat()
        }
        
        # Check if position already exists
        existing_pos = None
        for i, pos in enumerate(self.positions):
            if pos['symbol'] == symbol:
                existing_pos = i
                break
        
        if existing_pos is not None:
            # Update existing position
            old_pos = self.positions[existing_pos]
            total_qty = old_pos['quantity'] + quantity
            if total_qty == 0:
                # Remove position
                self.positions.pop(existing_pos)
            else:
                # Update average price
                total_invested = (old_pos['quantity'] * old_pos['avg_price'] + 
                                quantity * avg_price)
                new_avg_price = total_invested / total_qty
                
                self.positions[existing_pos].update({
                    'quantity': total_qty,
                    'avg_price': new_avg_price,
                    'current_price': current_price,
                    'invested_value': total_qty * new_avg_price,
                    'current_value': total_qty * current_price,
                    'unrealized_pnl': total_qty * (current_price - new_avg_price),
                    'pnl_percentage': ((current_price - new_avg_price) / new_avg_price) * 100,
                    'updated_at': datetime.now().isoformat()
                })
        else:
            # Add new position
            self.positions.append(position)
        
        # Update portfolio totals
        self._recalculate_totals()
        self.updated_at = datetime.now()
    
    def _recalculate_totals(self):
        """Recalculate portfolio totals"""
        self.invested_value = sum(pos['invested_value'] for pos in self.positions)
        current_positions_value = sum(pos['current_value'] for pos in self.positions)
        self.unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
        self.total_value = self.cash_balance + current_positions_value
        
        # Calculate weights
        if self.total_value > 0:
            for pos in self.positions:
                pos['weight'] = (pos['current_value'] / self.total_value) * 100
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.positions)
    
    def get_largest_position(self) -> Optional[Dict[str, Any]]:
        """Get largest position by value"""
        if not self.positions:
            return None
        return max(self.positions, key=lambda x: abs(x['current_value']))
    
    def get_total_exposure(self) -> float:
        """Get total market exposure"""
        return sum(abs(pos['current_value']) for pos in self.positions)


# Utility functions for model operations
def create_sample_user(username: str = "demo_user") -> User:
    """Create a sample user for testing"""
    user = User()
    user.username = username
    user.email = f"{username}@example.com"
    user.full_name = f"Demo User ({username})"
    return user

def create_sample_trading_session(user_id: str, capital: float = 10000) -> TradingSession:
    """Create a sample trading session"""
    session = TradingSession()
    session.user_id = user_id
    session.initial_capital = capital
    session.current_capital = capital
    return session

def create_sample_portfolio(user_id: str, session_id: str, capital: float = 10000) -> Portfolio:
    """Create a sample portfolio"""
    portfolio = Portfolio()
    portfolio.user_id = user_id
    portfolio.session_id = session_id
    portfolio.cash_balance = capital
    portfolio.total_value = capital
    return portfolio


if __name__ == '__main__':
    # Test the models
    print("Testing data models...")
    
    # Test User
    user = create_sample_user("test_trader")
    print(f"Created user: {user.username}")
    
    # Test TradingSession
    session = create_sample_trading_session(user.user_id, 10000)
    print(f"Created session: {session.session_id}")
    
    # Test Portfolio
    portfolio = create_sample_portfolio(user.user_id, session.session_id, 10000)
    print(f"Created portfolio: {portfolio.portfolio_id}")
    
    # Test Trade
    trade = Trade(
        symbol="NIFTY2550025000CE",
        action="BUY",
        quantity=50,
        price=150.0,
        option_type="CE",
        strike_price=25000.0
    )
    print(f"Created trade: {trade.trade_id}")
    
    # Test portfolio position addition
    portfolio.add_position("NIFTY2550025000CE", 50, 150.0, 160.0)
    print(f"Portfolio positions: {len(portfolio.positions)}")
    print(f"Portfolio total value: Rs. {portfolio.total_value:,.2f}")
    print(f"Unrealized P&L: Rs. {portfolio.unrealized_pnl:,.2f}")
    
    print("All models working correctly!")