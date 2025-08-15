"""
AIBot - AI-Powered Trading System
=================================

Advanced AI/ML trading system for Indian options market with:
- Real-time market data learning
- Multiple prediction models (IV, Price Movement, Anomaly Detection)
- Risk analysis and portfolio optimization
- Feature engineering and technical indicators
"""

__version__ = "1.0.0"
__author__ = "DhanAlgo Trading Team"

# Import main modules
from .models import *
from .utils import *

__all__ = [
    'models',
    'utils',
    'data',
    'training',
    'prediction',
    'evaluation'
]