"""
AI Models Package
================

Collection of AI models for algorithmic trading
"""

# Import all packages
from . import ivPridiction
from . import optionAnomoly
from . import niftyPriceMovement
from . import riskAnalysisModel
from . import indicators

# Import main classes for easy access
from .ivPridiction import (
    Nifty50DataPipeline,
    VolatilityFeatureCalculator,
    VolatilityRegimeAnalyzer,
    IndianMarketFeatures,
    SectorCorrelationAnalyzer,
    NiftyFeaturePreprocessor,
    NiftyVolatilityLSTM,
    AttentionLSTM,
    NiftyModelEvaluator,
    ModelComparator
)

from .optionAnomoly import (
    BlackScholesCalculator,
    NiftyOptionsDataPipeline,
    OptionsAnomalyAutoencoder,
    OptionsIsolationForest,
    HybridAnomalyDetector,
    ArbitrageOpportunityFinder
)

from .niftyPriceMovement import (
    NiftyDataPreprocessor,
    NiftyCNNClassifier,
    NiftyCNNLSTM,
    NiftyResidualCNN,
    NiftyLSTMClassifier,
    NiftyBidirectionalLSTM,
    NiftyAttentionLSTM,
    NiftyGRUClassifier,
    NiftyMovementEvaluator,
    ModelComparison
)

from .riskAnalysisModel import (
    PortfolioRiskCalculator,
    PositionSizer,
    RealTimeRiskMonitor,
    PortfolioStressTester
)

__all__ = [
    # Package names
    'ivPridiction',
    'optionAnomoly', 
    'niftyPriceMovement',
    'riskAnalysisModel',
    'indicators',
    
    # IV Prediction classes
    'Nifty50DataPipeline',
    'VolatilityFeatureCalculator',
    'VolatilityRegimeAnalyzer',
    'IndianMarketFeatures',
    'SectorCorrelationAnalyzer',
    'NiftyFeaturePreprocessor',
    'NiftyVolatilityLSTM',
    'AttentionLSTM',
    'NiftyModelEvaluator',
    'ModelComparator',
    
    # Option Anomaly classes
    'BlackScholesCalculator',
    'NiftyOptionsDataPipeline',
    'OptionsAnomalyAutoencoder',
    'OptionsIsolationForest',
    'HybridAnomalyDetector',
    'ArbitrageOpportunityFinder',
    
    # Nifty Price Movement classes
    'NiftyDataPreprocessor',
    'NiftyCNNClassifier',
    'NiftyCNNLSTM',
    'NiftyResidualCNN',
    'NiftyLSTMClassifier',
    'NiftyBidirectionalLSTM',
    'NiftyAttentionLSTM',
    'NiftyGRUClassifier',
    'NiftyMovementEvaluator',
    'ModelComparison',
    
    # Risk Analysis classes
    'PortfolioRiskCalculator',
    'PositionSizer',
    'RealTimeRiskMonitor',
    'PortfolioStressTester'
]