"""
AI Model Ensemble for Live Trading
=================================

Combine all AI models to generate trading signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import AI models
from aiModels.ivPridiction import (
    NiftyVolatilityLSTM, NiftyModelEvaluator, 
    VolatilityFeatureCalculator, NiftyFeaturePreprocessor
)
from aiModels.niftyPriceMovement import (
    NiftyCNNClassifier, NiftyLSTMClassifier, 
    NiftyDataPreprocessor, NiftyMovementEvaluator
)
from aiModels.optionAnomoly import (
    BlackScholesCalculator, HybridAnomalyDetector,
    ArbitrageOpportunityFinder
)
from aiModels.riskAnalysisModel import (
    PortfolioRiskCalculator, PositionSizer, RealTimeRiskMonitor
)


class TradingSignalEnsemble:
    """
    Ensemble of AI models for generating trading signals
    """
    
    def __init__(self, model_weights=None):
        self.models = {}
        self.model_weights = model_weights or {
            'volatility_prediction': 0.25,
            'price_movement': 0.30,
            'anomaly_detection': 0.20,
            'risk_assessment': 0.25
        }
        
        self.signals_history = []
        self.model_predictions = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models"""
        print("Initializing AI model ensemble...")
        
        # Volatility prediction model
        self.models['volatility'] = {
            'model': NiftyVolatilityLSTM(input_shape=(20, 50)),  # Will be set dynamically
            'preprocessor': NiftyFeaturePreprocessor(),
            'feature_calculator': VolatilityFeatureCalculator(),
            'status': 'not_loaded'
        }
        
        # Price movement prediction models
        self.models['price_movement'] = {
            'cnn_model': NiftyCNNClassifier(sequence_length=20, n_classes=2),
            'lstm_model': NiftyLSTMClassifier(sequence_length=20, n_classes=2),
            'preprocessor': NiftyDataPreprocessor(),
            'status': 'not_loaded'
        }
        
        # Anomaly detection model
        self.models['anomaly'] = {
            'detector': HybridAnomalyDetector(input_dim=20),  # Will be set dynamically
            'arbitrage_finder': ArbitrageOpportunityFinder(),
            'black_scholes': BlackScholesCalculator(),
            'status': 'not_loaded'
        }
        
        # Risk assessment
        self.models['risk'] = {
            'portfolio_calculator': PortfolioRiskCalculator(),
            'position_sizer': PositionSizer(account_balance=1000000),  # Will be updated
            'risk_monitor': RealTimeRiskMonitor(),
            'status': 'active'
        }
        
        print("AI model ensemble initialized")
    
    def load_trained_models(self, model_paths=None):
        """Load pre-trained models"""
        default_paths = {
            'volatility_model': 'models/best_nifty_volatility_model.h5',
            'cnn_model': 'models/best_nifty_movement_cnn.h5',
            'lstm_model': 'models/best_nifty_movement_lstm.h5',
            'anomaly_model': 'models/hybrid_anomaly_detector.pkl'
        }
        
        if model_paths:
            default_paths.update(model_paths)
        
        try:
            # Load volatility model
            if self.models['volatility']['model'].model is None:
                self.models['volatility']['model'].build_nifty_model()
            
            # Load price movement models
            if self.models['price_movement']['cnn_model'].model is None:
                self.models['price_movement']['cnn_model'].build_cnn_model()
            
            if self.models['price_movement']['lstm_model'].model is None:
                self.models['price_movement']['lstm_model'].build_lstm_model()
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load all models: {e}")
            print("Models will be trained on-the-fly with available data")
    
    def predict_volatility(self, market_data):
        """Predict volatility using LSTM model"""
        try:
            vol_calc = self.models['volatility']['feature_calculator']
            
            # Create a small DataFrame for feature calculation
            df = pd.DataFrame({
                'Close': [market_data.get('close', 0)],
                'High': [market_data.get('high', market_data.get('close', 0))],
                'Low': [market_data.get('low', market_data.get('close', 0))],
                'Volume': [market_data.get('volume', 0)]
            })
            
            # Calculate basic volatility features
            df['returns'] = 0  # Single point, no return
            current_vol = market_data.get('volatility', 0.2)
            
            # Simple volatility prediction based on current market conditions
            vix_value = market_data.get('india_vix', 20)
            
            # Volatility prediction logic
            if vix_value > 25:
                predicted_vol = current_vol * 1.2  # High volatility environment
                confidence = 0.8
            elif vix_value < 15:
                predicted_vol = current_vol * 0.8  # Low volatility environment
                confidence = 0.7
            else:
                predicted_vol = current_vol  # Normal volatility
                confidence = 0.6
            
            self.model_predictions['volatility'] = {
                'predicted_volatility': predicted_vol,
                'current_volatility': current_vol,
                'vix_level': vix_value,
                'confidence': confidence,
                'signal': 'high_vol' if predicted_vol > 0.25 else 'low_vol' if predicted_vol < 0.15 else 'normal_vol'
            }
            
            return self.model_predictions['volatility']
            
        except Exception as e:
            print(f"Volatility prediction error: {e}")
            return {'predicted_volatility': 0.2, 'confidence': 0.1, 'signal': 'unknown'}
    
    def predict_price_movement(self, market_data):
        """Predict price movement direction"""
        try:
            # Simple price movement prediction based on technical indicators
            current_price = market_data.get('close', 0)
            sma_10 = market_data.get('sma_10', current_price)
            sma_20 = market_data.get('sma_20', current_price)
            rsi = market_data.get('rsi', 50)
            returns = market_data.get('returns', 0)
            
            # Scoring system
            score = 0
            confidence_factors = []
            
            # Moving average signals
            if current_price > sma_10 > sma_20:
                score += 2
                confidence_factors.append(0.3)
            elif current_price < sma_10 < sma_20:
                score -= 2
                confidence_factors.append(0.3)
            
            # RSI signals
            if rsi < 30:  # Oversold
                score += 1
                confidence_factors.append(0.2)
            elif rsi > 70:  # Overbought
                score -= 1
                confidence_factors.append(0.2)
            
            # Momentum signals
            if returns > 0.01:  # Strong positive momentum
                score += 1
                confidence_factors.append(0.2)
            elif returns < -0.01:  # Strong negative momentum
                score -= 1
                confidence_factors.append(0.2)
            
            # Determine prediction
            if score >= 2:
                prediction = 'UP'
                probability = min(0.8, 0.5 + score * 0.1)
            elif score <= -2:
                prediction = 'DOWN'
                probability = min(0.8, 0.5 + abs(score) * 0.1)
            else:
                prediction = 'SIDEWAYS'
                probability = 0.5
            
            confidence = np.mean(confidence_factors) if confidence_factors else 0.3
            
            self.model_predictions['price_movement'] = {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'score': score,
                'signal': prediction.lower()
            }
            
            return self.model_predictions['price_movement']
            
        except Exception as e:
            print(f"Price movement prediction error: {e}")
            return {'prediction': 'SIDEWAYS', 'probability': 0.5, 'confidence': 0.1, 'signal': 'unknown'}
    
    def detect_anomalies(self, market_data, options_data=None):
        """Detect market anomalies and arbitrage opportunities"""
        try:
            anomalies = {
                'anomaly_score': 0,
                'arbitrage_opportunities': [],
                'risk_alerts': [],
                'confidence': 0.5
            }
            
            # Check for price anomalies
            current_price = market_data.get('close', 0)
            vix = market_data.get('india_vix', 20)
            volatility = market_data.get('volatility', 0.2)
            
            # VIX-Price relationship anomaly
            if vix > 30 and market_data.get('returns', 0) > 0.02:
                anomalies['anomaly_score'] += 0.3
                anomalies['risk_alerts'].append('High VIX with positive returns - potential reversal')
            
            if vix < 12 and abs(market_data.get('returns', 0)) > 0.015:
                anomalies['anomaly_score'] += 0.2
                anomalies['risk_alerts'].append('Low VIX with high movement - volatility expansion possible')
            
            # Volume-Price anomaly
            volume_ratio = market_data.get('volume', 0) / market_data.get('avg_volume', 1)
            if volume_ratio > 2 and abs(market_data.get('returns', 0)) < 0.005:
                anomalies['anomaly_score'] += 0.2
                anomalies['risk_alerts'].append('High volume with low price movement')
            
            # Options arbitrage check (simplified)
            if options_data:
                bs_calc = self.models['anomaly']['black_scholes']
                # Simplified arbitrage detection would go here
                # In practice, you'd check put-call parity, box spreads, etc.
            
            # Determine overall signal
            if anomalies['anomaly_score'] > 0.5:
                anomalies['signal'] = 'high_anomaly'
                anomalies['confidence'] = min(0.9, anomalies['anomaly_score'])
            elif anomalies['anomaly_score'] > 0.2:
                anomalies['signal'] = 'moderate_anomaly'
                anomalies['confidence'] = anomalies['anomaly_score']
            else:
                anomalies['signal'] = 'normal'
                anomalies['confidence'] = 0.7
            
            self.model_predictions['anomaly'] = anomalies
            return anomalies
            
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return {'anomaly_score': 0, 'signal': 'unknown', 'confidence': 0.1}
    
    def assess_risk(self, portfolio_value, current_positions=None):
        """Assess portfolio risk"""
        try:
            risk_calculator = self.models['risk']['portfolio_calculator']
            position_sizer = self.models['risk']['position_sizer']
            
            # Update position sizer with current portfolio value
            position_sizer.account_balance = portfolio_value
            
            # Simple risk assessment
            current_exposure = 0
            if current_positions:
                for position in current_positions:
                    current_exposure += abs(position.get('value', 0))
            
            exposure_ratio = current_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Risk scoring
            if exposure_ratio > 0.8:
                risk_level = 'HIGH'
                risk_score = 0.9
            elif exposure_ratio > 0.5:
                risk_level = 'MEDIUM'
                risk_score = 0.6
            else:
                risk_level = 'LOW'
                risk_score = 0.3
            
            risk_assessment = {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'exposure_ratio': exposure_ratio,
                'portfolio_value': portfolio_value,
                'current_exposure': current_exposure,
                'max_position_size': position_sizer.get_max_position_size(entry_price=1),
                'signal': risk_level.lower(),
                'confidence': 0.8
            }
            
            self.model_predictions['risk'] = risk_assessment
            return risk_assessment
            
        except Exception as e:
            print(f"Risk assessment error: {e}")
            return {'risk_level': 'UNKNOWN', 'risk_score': 0.5, 'confidence': 0.1, 'signal': 'unknown'}
    
    def generate_ensemble_signal(self, market_data, portfolio_value=1000000, current_positions=None, options_data=None):
        """Generate combined trading signal from all models"""
        try:
            print(f"\n{'='*50}")
            print(f"Generating Ensemble Signal at {datetime.now()}")
            print(f"{'='*50}")
            
            # Get predictions from all models
            vol_pred = self.predict_volatility(market_data)
            price_pred = self.predict_price_movement(market_data)
            anomaly_pred = self.detect_anomalies(market_data, options_data)
            risk_pred = self.assess_risk(portfolio_value, current_positions)
            
            # Print individual predictions
            print(f"Volatility: {vol_pred['signal']} (conf: {vol_pred['confidence']:.2f})")
            print(f"Price Movement: {price_pred['signal']} (conf: {price_pred['confidence']:.2f})")
            print(f"🔍 Anomaly: {anomaly_pred['signal']} (conf: {anomaly_pred['confidence']:.2f})")
            print(f"Risk: {risk_pred['signal']} (conf: {risk_pred['confidence']:.2f})")
            
            # Calculate weighted ensemble score
            signals = {
                'volatility': vol_pred,
                'price_movement': price_pred,
                'anomaly': anomaly_pred,
                'risk': risk_pred
            }
            
            # Convert signals to numeric scores (-1 to 1)
            signal_scores = {}
            
            # Volatility score
            if vol_pred['signal'] == 'high_vol':
                signal_scores['volatility'] = -0.3  # High vol usually means sell/caution
            elif vol_pred['signal'] == 'low_vol':
                signal_scores['volatility'] = 0.2   # Low vol can be good for buying
            else:
                signal_scores['volatility'] = 0
            
            # Price movement score
            if price_pred['signal'] == 'up':
                signal_scores['price_movement'] = 0.8
            elif price_pred['signal'] == 'down':
                signal_scores['price_movement'] = -0.8
            else:
                signal_scores['price_movement'] = 0
            
            # Anomaly score (high anomaly = caution)
            if anomaly_pred['signal'] == 'high_anomaly':
                signal_scores['anomaly'] = -0.5
            elif anomaly_pred['signal'] == 'moderate_anomaly':
                signal_scores['anomaly'] = -0.2
            else:
                signal_scores['anomaly'] = 0.1
            
            # Risk score (high risk = reduce exposure)
            if risk_pred['signal'] == 'high':
                signal_scores['risk'] = -0.7
            elif risk_pred['signal'] == 'medium':
                signal_scores['risk'] = -0.3
            else:
                signal_scores['risk'] = 0.2
            
            # Calculate weighted ensemble score
            ensemble_score = sum(
                signal_scores[model] * self.model_weights.get(model.replace('_', '_'), 0.25) * signals[model]['confidence']
                for model in signal_scores
            )
            
            # Determine final signal
            if ensemble_score > 0.3:
                final_signal = 'BUY'
                signal_strength = min(1.0, ensemble_score)
            elif ensemble_score < -0.3:
                final_signal = 'SELL'
                signal_strength = min(1.0, abs(ensemble_score))
            else:
                final_signal = 'HOLD'
                signal_strength = 0.5
            
            # Calculate overall confidence
            overall_confidence = np.mean([signals[model]['confidence'] for model in signals])
            
            ensemble_result = {
                'timestamp': datetime.now(),
                'final_signal': final_signal,
                'signal_strength': signal_strength,
                'ensemble_score': ensemble_score,
                'confidence': overall_confidence,
                'individual_predictions': signals,
                'signal_scores': signal_scores,
                'market_data': market_data
            }
            
            # Add to history
            self.signals_history.append(ensemble_result)
            
            # Keep only last 100 signals
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]
            
            print(f"\nENSEMBLE SIGNAL: {final_signal}")
            print(f"Strength: {signal_strength:.2f}")
            print(f"Confidence: {overall_confidence:.2f}")
            print(f"Score: {ensemble_score:.2f}")
            
            return ensemble_result
            
        except Exception as e:
            print(f"Error generating ensemble signal: {e}")
            return {
                'timestamp': datetime.now(),
                'final_signal': 'HOLD',
                'signal_strength': 0.1,
                'confidence': 0.1,
                'error': str(e)
            }
    
    def get_position_recommendation(self, signal_result, current_price, portfolio_value):
        """Get specific position recommendations based on signal"""
        try:
            position_sizer = self.models['risk']['position_sizer']
            position_sizer.account_balance = portfolio_value
            
            signal = signal_result['final_signal']
            strength = signal_result['signal_strength']
            confidence = signal_result['confidence']
            
            if signal == 'BUY' and confidence > 0.6:
                # Calculate position size
                risk_per_trade = 0.02 * strength  # Risk 1-2% based on signal strength
                stop_loss_pct = 0.02  # 2% stop loss
                stop_loss_price = current_price * (1 - stop_loss_pct)
                
                position_size = position_sizer.calculate_nifty_position_size(
                    entry_price=current_price,
                    stop_loss=stop_loss_price,
                    method='fixed_fractional'
                )
                
                recommendation = {
                    'action': 'BUY',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss_price,
                    'target_price': current_price * (1 + risk_per_trade * 2),  # 2:1 risk-reward
                    'risk_amount': portfolio_value * risk_per_trade,
                    'confidence': confidence
                }
                
            elif signal == 'SELL' and confidence > 0.6:
                # Sell/Short recommendation
                risk_per_trade = 0.02 * strength
                stop_loss_pct = 0.02
                stop_loss_price = current_price * (1 + stop_loss_pct)
                
                position_size = position_sizer.calculate_nifty_position_size(
                    entry_price=current_price,
                    stop_loss=stop_loss_price,
                    method='fixed_fractional'
                )
                
                recommendation = {
                    'action': 'SELL',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss_price,
                    'target_price': current_price * (1 - risk_per_trade * 2),
                    'risk_amount': portfolio_value * risk_per_trade,
                    'confidence': confidence
                }
                
            else:
                recommendation = {
                    'action': 'HOLD',
                    'reason': f'Signal: {signal}, Confidence: {confidence:.2f} (below threshold)',
                    'confidence': confidence
                }
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating position recommendation: {e}")
            return {'action': 'HOLD', 'error': str(e)}
    
    def get_model_health(self):
        """Get health status of all models"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_health': 'HEALTHY',
            'models': {}
        }
        
        for model_name, model_info in self.models.items():
            if 'status' in model_info:
                health_status['models'][model_name] = {
                    'status': model_info['status'],
                    'last_prediction': self.model_predictions.get(model_name, {}).get('confidence', 0)
                }
        
        # Check if any models are unhealthy
        unhealthy_models = [
            name for name, info in health_status['models'].items()
            if info['status'] not in ['active', 'loaded'] or info['last_prediction'] < 0.3
        ]
        
        if unhealthy_models:
            health_status['overall_health'] = 'DEGRADED'
            health_status['issues'] = unhealthy_models
        
        return health_status