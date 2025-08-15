"""
AlgoBot - Web-Based Algorithmic Trading Platform
===============================================

Professional trading platform with:
- Flask web application with real-time dashboard
- Live trading with Dhan API integration
- Risk management and portfolio tracking
- Indian options trading with tax calculations
- Real-time monitoring and alerts
"""

__version__ = "1.0.0"
__author__ = "DhanAlgo Trading Team"

# Import main modules
from .app import *
from .controllers import *
from .services import *
from .models import *

__all__ = [
    'app',
    'controllers',
    'services',
    'models',
    'live_trading',
    'utils'
]