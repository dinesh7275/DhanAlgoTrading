#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Learning System
========================

Advanced learning system that continuously adapts AI models based on actual trading results,
market feedback, and performance metrics to improve prediction accuracy over time.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import threading
import time
import queue
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TradingResult:
    """Trading result data structure"""
    trade_id: str
    signal_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    duration_minutes: int
    strategy: str
    confidence: float
    signal_features: Dict[str, float]
    market_conditions: Dict[str, str]
    success: bool
    exit_reason: str

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    last_updated: datetime
    confidence_calibration: Dict[str, float]

@dataclass
class LearningUpdate:
    """Learning update information"""
    timestamp: datetime
    model_name: str
    update_type: str  # RETRAIN, TUNE, FEATURE_UPDATE, WEIGHT_ADJUST
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    features_added: List[str]
    features_removed: List[str]
    improvement: float
    
class AdaptiveLearningSystem:
    """
    Adaptive learning system that improves trading AI models based on actual results
    """
    
    def __init__(self, update_frequency: int = 24):  # Update every 24 hours
        self.update_frequency = update_frequency  # Hours
        self.is_running = False
        self.last_update = datetime.now()
        
        # Data storage
        self.trading_results = []
        self.model_performances = {}
        self.learning_history = []
        self.feature_importance_history = {}
        
        # Models to adapt
        self.models = {
            'signal_classifier': None,
            'risk_predictor': None,
            'volatility_predictor': None,
            'direction_predictor': None,
            'confidence_calibrator': None
        }
        
        # Feature engineering
        self.feature_extractors = {}
        self.feature_scalers = {}
        self.label_encoders = {}
        
        # Learning parameters
        self.learning_config = {
            'min_samples_for_update': 50,
            'performance_threshold': 0.05,  # 5% improvement required
            'confidence_decay': 0.95,
            'feature_selection_threshold': 0.01,
            'cross_validation_folds': 5,
            'ensemble_models': 3,
            'learning_rate_decay': 0.98,
            'regularization_strength': 0.01
        }
        
        # Market regime detection
        self.market_regimes = {
            'bull_trend': {'signals': [], 'performance': {}},
            'bear_trend': {'signals': [], 'performance': {}},
            'sideways': {'signals': [], 'performance': {}},
            'high_volatility': {'signals': [], 'performance': {}},
            'low_volatility': {'signals': [], 'performance': {}}
        }
        
        # Data directories
        self.data_dir = Path("data/adaptive_learning")
        self.models_dir = Path("models/adaptive")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize learning system
        self._initialize_models()
        
        logger.info("AdaptiveLearningSystem initialized")
    
    def _initialize_models(self):
        """Initialize or load existing models"""
        try:
            # Load existing models if available
            for model_name in self.models.keys():
                model_path = self.models_dir / f"{model_name}.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded existing model: {model_name}")
                else:
                    # Create new model based on type
                    self.models[model_name] = self._create_new_model(model_name)
                    logger.info(f"Created new model: {model_name}")
            
            # Load feature extractors and scalers
            self._load_feature_processors()
            
            # Load historical performance if available
            self._load_performance_history()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _create_new_model(self, model_name: str):
        """Create a new model based on the model type"""
        if model_name == 'signal_classifier':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_name == 'risk_predictor':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_name == 'volatility_predictor':
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_name == 'direction_predictor':
            return MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=42
            )
        elif model_name == 'confidence_calibrator':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            return RandomForestClassifier(random_state=42)
    
    def add_trading_result(self, result: TradingResult):
        """Add a new trading result for learning"""
        try:
            self.trading_results.append(result)
            
            # Categorize by market regime
            self._categorize_by_market_regime(result)
            
            # Trigger learning update if enough new data
            if len(self.trading_results) % self.learning_config['min_samples_for_update'] == 0:
                self._schedule_learning_update()
            
            logger.debug(f"Added trading result: {result.trade_id}")
            
        except Exception as e:
            logger.error(f"Error adding trading result: {e}")
    
    def _categorize_by_market_regime(self, result: TradingResult):
        """Categorize trading result by market regime"""
        try:
            market_conditions = result.market_conditions
            
            # Determine regime based on market conditions
            if market_conditions.get('trend') == 'BULLISH':
                self.market_regimes['bull_trend']['signals'].append(result)
            elif market_conditions.get('trend') == 'BEARISH':
                self.market_regimes['bear_trend']['signals'].append(result)
            else:
                self.market_regimes['sideways']['signals'].append(result)
            
            if market_conditions.get('volatility') == 'HIGH':
                self.market_regimes['high_volatility']['signals'].append(result)
            elif market_conditions.get('volatility') == 'LOW':
                self.market_regimes['low_volatility']['signals'].append(result)
            
        except Exception as e:
            logger.error(f"Error categorizing market regime: {e}")
    
    def _schedule_learning_update(self):
        """Schedule a learning update"""
        try:
            # Check if enough time has passed since last update
            time_since_update = datetime.now() - self.last_update
            if time_since_update.total_seconds() / 3600 >= self.update_frequency:
                
                # Run learning update in background thread
                update_thread = threading.Thread(
                    target=self._perform_learning_update,
                    daemon=True
                )
                update_thread.start()
                
        except Exception as e:
            logger.error(f"Error scheduling learning update: {e}")
    
    def _perform_learning_update(self):
        """Perform the actual learning update"""
        logger.info("Starting adaptive learning update...")
        
        try:
            # Prepare training data
            training_data = self._prepare_training_data()
            
            if not training_data or len(training_data['X']) < self.learning_config['min_samples_for_update']:
                logger.warning("Insufficient data for learning update")
                return
            
            # Update each model
            for model_name, model in self.models.items():
                try:
                    self._update_model(model_name, model, training_data)
                except Exception as e:
                    logger.error(f"Error updating model {model_name}: {e}")
            
            # Update feature importance
            self._update_feature_importance()
            
            # Optimize model weights based on performance
            self._optimize_model_weights()
            
            # Update regime-specific parameters
            self._update_regime_parameters()
            
            # Save updated models
            self._save_models()
            
            # Record learning update
            self.last_update = datetime.now()
            
            logger.info("Adaptive learning update completed successfully")
            
        except Exception as e:
            logger.error(f"Error in learning update: {e}")
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """Prepare training data from trading results"""
        try:
            if not self.trading_results:
                return {}
            
            # Extract features and labels
            features = []
            labels = []
            weights = []
            
            for result in self.trading_results:
                try:
                    # Extract features from signal
                    feature_vector = self._extract_features_from_result(result)
                    
                    # Create label (success/failure)
                    label = 1 if result.success else 0
                    
                    # Calculate sample weight based on recency and performance
                    weight = self._calculate_sample_weight(result)
                    
                    features.append(feature_vector)
                    labels.append(label)
                    weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Error processing result {result.trade_id}: {e}")
                    continue
            
            if not features:
                return {}
            
            # Convert to arrays
            X = np.array(features)
            y = np.array(labels)
            sample_weights = np.array(weights)
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return {
                'X': X_scaled,
                'y': y,
                'weights': sample_weights,
                'feature_names': self._get_feature_names(),
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {}
    
    def _extract_features_from_result(self, result: TradingResult) -> List[float]:
        """Extract feature vector from trading result"""
        features = []
        
        try:
            # Signal features
            signal_features = result.signal_features
            features.extend([
                signal_features.get('rsi', 50),
                signal_features.get('macd', 0),
                signal_features.get('bb_position', 0.5),
                signal_features.get('volume_ratio', 1.0),
                signal_features.get('atr', 0.02),
                signal_features.get('momentum', 0),
                signal_features.get('volatility', 0.2)
            ])
            
            # Market condition features
            market_conditions = result.market_conditions
            trend_encoding = {'BULLISH': 1, 'BEARISH': -1, 'SIDEWAYS': 0}
            volatility_encoding = {'HIGH': 2, 'NORMAL': 1, 'LOW': 0}
            volume_encoding = {'HIGH': 2, 'NORMAL': 1, 'LOW': 0}
            
            features.extend([
                trend_encoding.get(market_conditions.get('trend'), 0),
                volatility_encoding.get(market_conditions.get('volatility'), 1),
                volume_encoding.get(market_conditions.get('volume'), 1)
            ])
            
            # Trade features
            features.extend([
                result.confidence,
                result.duration_minutes / 1440,  # Normalize to days
                abs(result.pnl_percent),
                np.log1p(result.quantity)  # Log transform quantity
            ])
            
            # Time-based features
            hour_of_day = result.entry_time.hour
            day_of_week = result.entry_time.weekday()
            features.extend([
                np.sin(2 * np.pi * hour_of_day / 24),
                np.cos(2 * np.pi * hour_of_day / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0.0] * 17  # Return default feature vector
    
    def _calculate_sample_weight(self, result: TradingResult) -> float:
        """Calculate sample weight based on recency and other factors"""
        try:
            # Base weight
            weight = 1.0
            
            # Recency weight (more recent results get higher weight)
            days_ago = (datetime.now() - result.entry_time).days
            recency_weight = self.learning_config['confidence_decay'] ** days_ago
            weight *= recency_weight
            
            # Performance weight (more significant results get higher weight)
            performance_weight = 1.0 + abs(result.pnl_percent)
            weight *= performance_weight
            
            # Confidence weight
            confidence_weight = result.confidence
            weight *= confidence_weight
            
            return max(weight, 0.1)  # Minimum weight
            
        except Exception as e:
            logger.error(f"Error calculating sample weight: {e}")
            return 1.0
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        return [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'atr', 'momentum', 'volatility',
            'trend', 'market_volatility', 'market_volume',
            'confidence', 'duration_days', 'pnl_magnitude', 'log_quantity',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
    
    def _update_model(self, model_name: str, model: Any, training_data: Dict[str, Any]):
        """Update a specific model with new training data"""
        try:
            X = training_data['X']
            y = training_data['y']
            weights = training_data['weights']
            
            # Get current performance
            current_performance = self._evaluate_model_performance(model, X, y)
            
            # Create and train new model
            new_model = self._create_new_model(model_name)
            
            # Train with sample weights if supported
            if hasattr(new_model, 'fit') and 'sample_weight' in new_model.fit.__code__.co_varnames:
                new_model.fit(X, y, sample_weight=weights)
            else:
                new_model.fit(X, y)
            
            # Evaluate new model performance
            new_performance = self._evaluate_model_performance(new_model, X, y)
            
            # Update model if performance improved
            improvement = new_performance['accuracy'] - current_performance['accuracy']
            if improvement > self.learning_config['performance_threshold']:
                self.models[model_name] = new_model
                
                # Record learning update
                update = LearningUpdate(
                    timestamp=datetime.now(),
                    model_name=model_name,
                    update_type='RETRAIN',
                    performance_before=current_performance,
                    performance_after=new_performance,
                    features_added=[],
                    features_removed=[],
                    improvement=improvement
                )
                self.learning_history.append(update)
                
                logger.info(f"Updated {model_name} - Accuracy improved by {improvement:.3f}")
            else:
                logger.debug(f"No significant improvement for {model_name}")
                
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
    
    def _evaluate_model_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if len(np.unique(y)) < 2:
                return {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, 
                cv=min(self.learning_config['cross_validation_folds'], len(y) // 2),
                scoring='accuracy'
            )
            
            # Predictions for detailed metrics
            y_pred = model.predict(X)
            
            return {
                'accuracy': np.mean(cv_scores),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
    
    def _update_feature_importance(self):
        """Update feature importance analysis"""
        try:
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self._get_feature_names()
                    
                    # Store feature importance
                    if model_name not in self.feature_importance_history:
                        self.feature_importance_history[model_name] = []
                    
                    importance_dict = dict(zip(feature_names, importances))
                    self.feature_importance_history[model_name].append({
                        'timestamp': datetime.now(),
                        'importances': importance_dict
                    })
                    
                    # Keep only recent history
                    if len(self.feature_importance_history[model_name]) > 100:
                        self.feature_importance_history[model_name] = self.feature_importance_history[model_name][-100:]
                    
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
    
    def _optimize_model_weights(self):
        """Optimize ensemble model weights based on recent performance"""
        try:
            # Calculate performance metrics for each model
            model_weights = {}
            
            for model_name in self.models.keys():
                # Get recent results for this model
                recent_results = [
                    r for r in self.trading_results[-100:] 
                    if r.strategy == model_name
                ]
                
                if len(recent_results) >= 10:
                    win_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
                    avg_return = np.mean([r.pnl_percent for r in recent_results])
                    
                    # Calculate weight based on performance
                    performance_score = win_rate * 0.6 + (avg_return + 1) * 0.4
                    model_weights[model_name] = max(performance_score, 0.1)
                else:
                    model_weights[model_name] = 1.0
            
            # Normalize weights
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                self.model_weights = {k: v / total_weight for k, v in model_weights.items()}
            
        except Exception as e:
            logger.error(f"Error optimizing model weights: {e}")
    
    def _update_regime_parameters(self):
        """Update parameters for different market regimes"""
        try:
            for regime, data in self.market_regimes.items():
                if len(data['signals']) >= 20:
                    # Calculate regime-specific performance
                    successes = sum(1 for signal in data['signals'] if signal.success)
                    total = len(data['signals'])
                    win_rate = successes / total
                    
                    avg_return = np.mean([s.pnl_percent for s in data['signals']])
                    avg_duration = np.mean([s.duration_minutes for s in data['signals']])
                    
                    data['performance'] = {
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'avg_duration': avg_duration,
                        'total_trades': total,
                        'last_updated': datetime.now()
                    }
                    
        except Exception as e:
            logger.error(f"Error updating regime parameters: {e}")
    
    def predict_signal_success(self, signal_features: Dict[str, float], 
                             market_conditions: Dict[str, str], 
                             confidence: float) -> Dict[str, float]:
        """Predict probability of signal success"""
        try:
            # Extract features
            feature_vector = self._extract_signal_features(signal_features, market_conditions, confidence)
            
            # Scale features
            if hasattr(self, 'feature_scaler') and self.feature_scaler:
                feature_vector = self.feature_scaler.transform([feature_vector])
            else:
                feature_vector = np.array([feature_vector])
            
            # Get predictions from all models
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(feature_vector)[0]
                        predictions[model_name] = prob[1] if len(prob) > 1 else prob[0]
                    else:
                        pred = model.predict(feature_vector)[0]
                        predictions[model_name] = float(pred)
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
                    predictions[model_name] = 0.5
            
            # Ensemble prediction with weights
            if hasattr(self, 'model_weights'):
                weighted_prediction = sum(
                    predictions[name] * self.model_weights.get(name, 1.0)
                    for name in predictions.keys()
                ) / len(predictions)
            else:
                weighted_prediction = np.mean(list(predictions.values()))
            
            return {
                'ensemble_probability': weighted_prediction,
                'individual_predictions': predictions,
                'confidence_adjusted': weighted_prediction * confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting signal success: {e}")
            return {'ensemble_probability': 0.5, 'individual_predictions': {}, 'confidence_adjusted': 0.5}
    
    def _extract_signal_features(self, signal_features: Dict[str, float], 
                                market_conditions: Dict[str, str], 
                                confidence: float) -> List[float]:
        """Extract features for prediction"""
        features = []
        
        # Signal features
        features.extend([
            signal_features.get('rsi', 50),
            signal_features.get('macd', 0),
            signal_features.get('bb_position', 0.5),
            signal_features.get('volume_ratio', 1.0),
            signal_features.get('atr', 0.02),
            signal_features.get('momentum', 0),
            signal_features.get('volatility', 0.2)
        ])
        
        # Market condition features
        trend_encoding = {'BULLISH': 1, 'BEARISH': -1, 'SIDEWAYS': 0}
        volatility_encoding = {'HIGH': 2, 'NORMAL': 1, 'LOW': 0}
        volume_encoding = {'HIGH': 2, 'NORMAL': 1, 'LOW': 0}
        
        features.extend([
            trend_encoding.get(market_conditions.get('trend'), 0),
            volatility_encoding.get(market_conditions.get('volatility'), 1),
            volume_encoding.get(market_conditions.get('volume'), 1)
        ])
        
        # Other features
        features.extend([confidence, 0, 0, 0])  # Placeholders for trade-specific features
        
        # Time features
        now = datetime.now()
        features.extend([
            np.sin(2 * np.pi * now.hour / 24),
            np.cos(2 * np.pi * now.hour / 24),
            np.sin(2 * np.pi * now.weekday() / 7),
            np.cos(2 * np.pi * now.weekday() / 7)
        ])
        
        return features
    
    def get_learning_performance(self) -> Dict[str, Any]:
        """Get comprehensive learning system performance"""
        try:
            # Calculate overall metrics
            if not self.trading_results:
                return {'error': 'No trading results available'}
            
            recent_results = self.trading_results[-100:]  # Last 100 trades
            
            total_trades = len(recent_results)
            successful_trades = sum(1 for r in recent_results if r.success)
            win_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            avg_return = np.mean([r.pnl_percent for r in recent_results])
            total_return = sum(r.pnl_percent for r in recent_results)
            
            # Calculate Sharpe ratio (simplified)
            returns = [r.pnl_percent for r in recent_results]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Model-specific performance
            model_performance = {}
            for model_name in self.models.keys():
                model_results = [r for r in recent_results if r.strategy == model_name]
                if model_results:
                    model_win_rate = sum(1 for r in model_results if r.success) / len(model_results)
                    model_avg_return = np.mean([r.pnl_percent for r in model_results])
                    
                    model_performance[model_name] = {
                        'trades': len(model_results),
                        'win_rate': model_win_rate,
                        'avg_return': model_avg_return
                    }
            
            # Recent learning updates
            recent_updates = [
                asdict(update) for update in self.learning_history[-10:]
            ]
            
            # Regime performance
            regime_performance = {}
            for regime, data in self.market_regimes.items():
                if 'performance' in data:
                    regime_performance[regime] = data['performance']
            
            return {
                'overall_performance': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio
                },
                'model_performance': model_performance,
                'regime_performance': regime_performance,
                'recent_learning_updates': recent_updates,
                'last_update': self.last_update.isoformat(),
                'total_learning_updates': len(self.learning_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting learning performance: {e}")
            return {'error': str(e)}
    
    def _save_models(self):
        """Save all models and metadata"""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
            
            # Save feature processors
            if hasattr(self, 'feature_scaler') and self.feature_scaler:
                joblib.dump(self.feature_scaler, self.models_dir / "feature_scaler.pkl")
            
            # Save learning metadata
            metadata = {
                'last_update': self.last_update.isoformat(),
                'learning_history': [asdict(update) for update in self.learning_history],
                'feature_importance_history': self.feature_importance_history,
                'market_regimes': self.market_regimes,
                'model_weights': getattr(self, 'model_weights', {})
            }
            
            with open(self.data_dir / "learning_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.debug("Models and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_feature_processors(self):
        """Load feature processors"""
        try:
            scaler_path = self.models_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                
        except Exception as e:
            logger.warning(f"Could not load feature processors: {e}")
    
    def _load_performance_history(self):
        """Load historical performance data"""
        try:
            metadata_path = self.data_dir / "learning_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.learning_history = [
                    LearningUpdate(**update) for update in metadata.get('learning_history', [])
                ]
                self.feature_importance_history = metadata.get('feature_importance_history', {})
                self.market_regimes = metadata.get('market_regimes', self.market_regimes)
                self.model_weights = metadata.get('model_weights', {})
                
                if 'last_update' in metadata:
                    self.last_update = datetime.fromisoformat(metadata['last_update'])
                
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
    
    def export_learning_data(self, filename: str = None) -> str:
        """Export learning system data"""
        if not filename:
            filename = f"adaptive_learning_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'trading_results': [asdict(result) for result in self.trading_results],
                'learning_performance': self.get_learning_performance(),
                'feature_importance_history': self.feature_importance_history,
                'market_regimes': self.market_regimes,
                'learning_config': self.learning_config
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            export_data = convert_datetime(export_data)
            
            export_path = self.data_dir / filename
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Learning data exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return ""
    
    def force_learning_update(self):
        """Force an immediate learning update"""
        logger.info("Forcing learning update...")
        self._perform_learning_update()
    
    def get_model_predictions_explanation(self, signal_features: Dict[str, float], 
                                        market_conditions: Dict[str, str], 
                                        confidence: float) -> Dict[str, Any]:
        """Get detailed explanation of model predictions"""
        try:
            predictions = self.predict_signal_success(signal_features, market_conditions, confidence)
            
            # Get feature importance for explanation
            explanations = {}
            feature_vector = self._extract_signal_features(signal_features, market_conditions, confidence)
            feature_names = self._get_feature_names()
            
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Calculate contribution of each feature
                    contributions = []
                    for i, (name, value, importance) in enumerate(zip(feature_names, feature_vector, importances)):
                        contribution = value * importance
                        contributions.append({
                            'feature': name,
                            'value': value,
                            'importance': importance,
                            'contribution': contribution
                        })
                    
                    explanations[model_name] = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)
            
            return {
                'predictions': predictions,
                'feature_explanations': explanations,
                'top_factors': explanations.get('signal_classifier', [])[:5] if explanations else []
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction explanation: {e}")
            return {'predictions': {}, 'feature_explanations': {}, 'top_factors': []}

# Example usage and testing functions

def create_sample_learning_system() -> AdaptiveLearningSystem:
    """Create a sample learning system with test data"""
    learning_system = AdaptiveLearningSystem()
    
    # Add some sample trading results
    for i in range(100):
        result = TradingResult(
            trade_id=f"trade_{i}",
            signal_id=f"signal_{i}",
            entry_time=datetime.now() - timedelta(days=i),
            exit_time=datetime.now() - timedelta(days=i, hours=-2),
            entry_price=25000 + np.random.normal(0, 100),
            exit_price=25000 + np.random.normal(0, 120),
            quantity=50,
            pnl=np.random.normal(500, 1000),
            pnl_percent=np.random.normal(0.02, 0.05),
            duration_minutes=120 + np.random.randint(-60, 180),
            strategy='signal_classifier',
            confidence=np.random.uniform(0.6, 0.9),
            signal_features={
                'rsi': np.random.uniform(30, 70),
                'macd': np.random.normal(0, 0.5),
                'bb_position': np.random.uniform(0.2, 0.8),
                'volume_ratio': np.random.uniform(0.8, 2.0),
                'atr': np.random.uniform(0.01, 0.04)
            },
            market_conditions={
                'trend': np.random.choice(['BULLISH', 'BEARISH', 'SIDEWAYS']),
                'volatility': np.random.choice(['HIGH', 'NORMAL', 'LOW']),
                'volume': np.random.choice(['HIGH', 'NORMAL', 'LOW'])
            },
            success=np.random.choice([True, False], p=[0.6, 0.4]),
            exit_reason='TARGET' if np.random.random() > 0.3 else 'STOP_LOSS'
        )
        
        learning_system.add_trading_result(result)
    
    return learning_system

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create learning system
    learning_system = create_sample_learning_system()
    
    # Force a learning update
    learning_system.force_learning_update()
    
    # Get performance
    performance = learning_system.get_learning_performance()
    print("📊 Learning System Performance:")
    print(f"Win Rate: {performance['overall_performance']['win_rate']:.2%}")
    print(f"Average Return: {performance['overall_performance']['avg_return']:.2%}")
    print(f"Total Trades: {performance['overall_performance']['total_trades']}")
    
    # Test prediction
    test_signal = {
        'rsi': 65,
        'macd': 0.2,
        'bb_position': 0.7,
        'volume_ratio': 1.5,
        'atr': 0.025
    }
    
    test_conditions = {
        'trend': 'BULLISH',
        'volatility': 'NORMAL',
        'volume': 'HIGH'
    }
    
    prediction = learning_system.predict_signal_success(test_signal, test_conditions, 0.8)
    print(f"\n🔮 Signal Success Prediction: {prediction['ensemble_probability']:.2%}")
    
    # Export data
    export_file = learning_system.export_learning_data()
    print(f"\n💾 Learning data exported to: {export_file}")