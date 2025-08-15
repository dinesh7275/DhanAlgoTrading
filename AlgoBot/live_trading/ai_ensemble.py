"""
Trading Signal Ensemble for Indian Options Trading
==================================================

Simplified ensemble using technical indicators and risk analysis for ₹10,000 capital
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import AI models including restored learning modules
from aiModels.indicators.technical_indicators import TechnicalIndicators
from aiModels.riskAnalysisModel import (
    PortfolioRiskCalculator, PositionSizer, RealTimeRiskMonitor
)
from aiModels.ivPrediction.live_iv_predictor import LiveIVPredictor
from aiModels.priceMovement.live_price_predictor import LivePricePredictor
from aiModels.optionsAnomaly.live_anomaly_detector import LiveAnomalyDetector


class TradingSignalEnsemble:
    """
    Simplified ensemble for options trading signals using technical indicators
    """
    
    def __init__(self, capital=10000):
        self.capital = capital
        self.technical_indicators = TechnicalIndicators()
        self.risk_calculator = PortfolioRiskCalculator()
        self.position_sizer = PositionSizer(capital, 0.10, 0.15)
        self.risk_monitor = RealTimeRiskMonitor(0.15, 0.10)
        
        # Initialize restored learning modules
        self.iv_predictor = LiveIVPredictor(lookback_days=7)
        self.price_predictor = LivePricePredictor(lookback_days=5)
        self.anomaly_detector = LiveAnomalyDetector(contamination=0.1, lookback_days=3)
        
        self.signals_history = []
        
        print(f"Enhanced Trading Signal Ensemble initialized for Rs.{capital:,} capital")
        print("- Technical Indicators: Enabled")
        print("- IV Prediction with Real Data: Enabled") 
        print("- Price Movement Prediction: Enabled")
        print("- Options Anomaly Detection: Enabled")
    
    def generate_ensemble_signal(self, market_data, portfolio_value=10000, current_positions=None):
        """Generate trading signal based on technical indicators"""
        try:
            # Get current market data
            current_price = market_data.get('nifty', {}).get('ltp', 0)
            if current_price == 0:
                return self._create_hold_signal("No market data")
            
            # Create DataFrame for technical analysis
            df = self._create_dataframe_from_market_data(market_data)
            if df is None or len(df) < 20:
                return self._create_hold_signal("Insufficient data for analysis")
            
            # Calculate technical indicators
            ma_indicators = self.technical_indicators.calculate_moving_averages(df)
            momentum_indicators = self.technical_indicators.calculate_momentum_indicators(df)
            
            # Get latest indicator values
            latest_ma = ma_indicators.iloc[-1]
            latest_momentum = momentum_indicators.iloc[-1]
            
            # Generate signal based on EMA 6/15 crossover and MACD
            signal_result = self._analyze_options_signals(
                latest_ma, latest_momentum, current_price
            )
            
            # Enhance with learning module predictions
            iv_prediction = self.iv_predictor.predict_iv()
            price_prediction = self.price_predictor.predict_price_movement()
            anomaly_result = self.anomaly_detector.detect_anomalies()
            
            # Combine predictions for enhanced signal
            signal_result = self._enhance_signal_with_learning(
                signal_result, iv_prediction, price_prediction, anomaly_result
            )
            
            # Add risk assessment
            risk_score = self._assess_market_risk(market_data, current_positions)
            signal_result['risk_score'] = risk_score
            
            # Adjust confidence based on risk
            if risk_score > 0.7:
                signal_result['confidence'] *= 0.7  # Reduce confidence in high risk
            
            # Store signal in history
            self.signals_history.append(signal_result)
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]
            
            return signal_result
            
        except Exception as e:
            print(f"Signal generation error: {e}")
            return self._create_hold_signal(f"Error: {e}")
    
    def _enhance_signal_with_learning(self, base_signal, iv_prediction, price_prediction, anomaly_result):
        """Enhance signal using learning module predictions"""
        try:
            enhanced_signal = base_signal.copy()
            learning_factors = []
            
            # IV Prediction Enhancement
            if iv_prediction.get('status') == 'success':
                predicted_iv = iv_prediction['predicted_iv']
                iv_confidence = iv_prediction['confidence']
                
                # High IV favors premium selling (but we only buy), so adjust accordingly
                if predicted_iv > 0.25:  # High IV environment
                    enhanced_signal['confidence'] *= (1 + iv_confidence * 0.2)  # Boost confidence
                    learning_factors.append(f"High IV environment ({predicted_iv*100:.1f}%)")
                elif predicted_iv < 0.15:  # Low IV environment  
                    enhanced_signal['confidence'] *= 0.9  # Slightly reduce confidence
                    learning_factors.append(f"Low IV environment ({predicted_iv*100:.1f}%)")
                
                enhanced_signal['predicted_iv'] = predicted_iv
                enhanced_signal['iv_confidence'] = iv_confidence
            
            # Price Movement Enhancement
            if price_prediction.get('status') == 'success':
                ml_signal = price_prediction['signal']
                ml_confidence = price_prediction['confidence']
                option_rec = price_prediction['option_recommendation']
                
                # Align ML prediction with technical signal
                if (enhanced_signal['final_signal'] == 'BUY' and ml_signal == 'BULLISH' and 
                    enhanced_signal['option_type'] == 'CE' and option_rec == 'CE'):
                    # Perfect alignment - boost confidence
                    enhanced_signal['confidence'] *= (1 + ml_confidence * 0.3)
                    learning_factors.append("ML confirms bullish CE signal")
                elif (enhanced_signal['final_signal'] == 'SELL' and ml_signal == 'BEARISH' and 
                      enhanced_signal['option_type'] == 'PE' and option_rec == 'PE'):
                    # Perfect alignment for bearish
                    enhanced_signal['confidence'] *= (1 + ml_confidence * 0.3)
                    learning_factors.append("ML confirms bearish PE signal")
                elif ml_signal == 'NEUTRAL':
                    # ML suggests caution
                    enhanced_signal['confidence'] *= 0.8
                    learning_factors.append("ML suggests neutral market")
                elif ml_confidence > 0.7:
                    # Strong ML signal different from technical - consider override
                    if ml_signal == 'BULLISH' and option_rec == 'CE':
                        enhanced_signal['final_signal'] = 'BUY'
                        enhanced_signal['option_type'] = 'CE'
                        enhanced_signal['confidence'] = ml_confidence * 0.8
                        learning_factors.append("ML override: Strong bullish signal")
                    elif ml_signal == 'BEARISH' and option_rec == 'PE':
                        enhanced_signal['final_signal'] = 'SELL'  # We buy PE
                        enhanced_signal['option_type'] = 'PE'
                        enhanced_signal['confidence'] = ml_confidence * 0.8
                        learning_factors.append("ML override: Strong bearish signal")
                
                enhanced_signal['ml_signal'] = ml_signal
                enhanced_signal['ml_confidence'] = ml_confidence
            
            # Anomaly Detection Enhancement
            if anomaly_result.get('status') == 'success' and anomaly_result.get('anomalies_found', 0) > 0:
                opportunities = self.anomaly_detector.get_trading_opportunities(anomaly_result)
                
                if opportunities:
                    # Found anomaly opportunities
                    best_opportunity = opportunities[0]
                    
                    if best_opportunity['profit_potential'] > 0.1:  # 10%+ profit potential
                        # Strong anomaly signal
                        enhanced_signal['final_signal'] = best_opportunity['action']
                        enhanced_signal['option_type'] = best_opportunity['option_type']
                        enhanced_signal['confidence'] = min(0.9, enhanced_signal['confidence'] + 0.2)
                        enhanced_signal['anomaly_opportunity'] = best_opportunity
                        learning_factors.append(f"Anomaly: {best_opportunity['reason']}")
                
                enhanced_signal['anomalies_found'] = anomaly_result['anomalies_found']
            
            # Final confidence adjustment
            enhanced_signal['confidence'] = min(0.95, enhanced_signal['confidence'])
            
            # Add learning insights
            if learning_factors:
                enhanced_signal['reasons'].extend(learning_factors)
                enhanced_signal['learning_enhanced'] = True
            else:
                enhanced_signal['learning_enhanced'] = False
            
            return enhanced_signal
            
        except Exception as e:
            print(f"Error enhancing signal with learning: {e}")
            base_signal['learning_enhanced'] = False
            return base_signal
    
    def _create_dataframe_from_market_data(self, market_data):
        """Create DataFrame from market data"""
        try:
            nifty_data = market_data.get('nifty', {})
            if not nifty_data:
                return None
                
            # Use current price as base for minimal DataFrame
            current_price = nifty_data.get('ltp', 0)
            change = nifty_data.get('change', 0)
            prev_close = current_price - change
            
            # Create basic OHLCV data (simplified)
            high = current_price * 1.005
            low = current_price * 0.995
            volume = 1000000  # Default volume
            
            # Generate some historical data points for indicator calculation
            dates = pd.date_range(end=datetime.now(), periods=50, freq='15min')
            
            # Simple price series with some randomness for technical indicators
            base_prices = np.linspace(prev_close, current_price, 50)
            noise = np.random.normal(0, current_price * 0.002, 50)
            closes = base_prices + noise
            
            df = pd.DataFrame({
                'Close': closes,
                'High': closes * 1.005,
                'Low': closes * 0.995,
                'Volume': np.random.randint(500000, 1500000, 50)
            }, index=dates)
            
            return df
            
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None
    
    def _analyze_options_signals(self, ma_indicators, momentum_indicators, current_price):
        """Analyze signals specifically for options trading"""
        signal_score = 0
        confidence_factors = []
        reasons = []
        
        # EMA 6/15 crossover analysis (primary signal for options)
        ema_6 = ma_indicators.get('EMA_6', current_price)
        ema_15 = ma_indicators.get('EMA_15', current_price)
        ema_cross = ma_indicators.get('EMA_6_15_Signal', 0)
        
        if ema_6 > ema_15 and current_price > ema_6:
            signal_score += 4  # Strong bullish for CE
            confidence_factors.append(0.4)
            reasons.append("EMA 6/15 bullish crossover with price above EMA 6")
        elif ema_6 < ema_15 and current_price < ema_6:
            signal_score -= 4  # Strong bearish for PE
            confidence_factors.append(0.4)
            reasons.append("EMA 6/15 bearish crossover with price below EMA 6")
        elif ema_cross == 1:
            signal_score += 2  # Mild bullish
            confidence_factors.append(0.2)
            reasons.append("EMA 6 above EMA 15")
        elif ema_cross == -1:
            signal_score -= 2  # Mild bearish
            confidence_factors.append(0.2)
            reasons.append("EMA 6 below EMA 15")
        
        # MACD analysis
        macd = momentum_indicators.get('MACD', 0)
        macd_signal = momentum_indicators.get('MACD_Signal', 0)
        macd_cross = momentum_indicators.get('MACD_Signal_Cross', 0)
        
        if macd > macd_signal and macd > 0:
            signal_score += 3
            confidence_factors.append(0.3)
            reasons.append("MACD bullish above signal line and zero")
        elif macd < macd_signal and macd < 0:
            signal_score -= 3
            confidence_factors.append(0.3)
            reasons.append("MACD bearish below signal line and zero")
        elif macd_cross == 1:
            signal_score += 1
            confidence_factors.append(0.15)
            reasons.append("MACD crossed above signal line")
        elif macd_cross == 0:
            signal_score -= 1
            confidence_factors.append(0.15)
            reasons.append("MACD crossed below signal line")
        
        # RSI analysis (options-optimized levels)
        rsi = momentum_indicators.get('RSI_14', 50)
        if rsi < 25:  # Deep oversold - good for CE
            signal_score += 2
            confidence_factors.append(0.25)
            reasons.append("RSI deeply oversold - CE opportunity")
        elif rsi > 75:  # Deep overbought - good for PE
            signal_score -= 2
            confidence_factors.append(0.25)
            reasons.append("RSI deeply overbought - PE opportunity")
        elif rsi < 35:
            signal_score += 1
            confidence_factors.append(0.15)
            reasons.append("RSI oversold")
        elif rsi > 65:
            signal_score -= 1
            confidence_factors.append(0.15)
            reasons.append("RSI overbought")
        
        # Determine final signal
        if signal_score >= 5:  # Strong bullish
            final_signal = 'BUY'
            option_type = 'CE'
            confidence = min(0.9, 0.6 + len(confidence_factors) * 0.1)
        elif signal_score <= -5:  # Strong bearish
            final_signal = 'SELL'  # But we only buy PE, not sell
            option_type = 'PE'
            confidence = min(0.9, 0.6 + len(confidence_factors) * 0.1)
        elif signal_score >= 3:
            final_signal = 'BUY'
            option_type = 'CE'
            confidence = min(0.7, 0.5 + len(confidence_factors) * 0.08)
        elif signal_score <= -3:
            final_signal = 'SELL'
            option_type = 'PE'  
            confidence = min(0.7, 0.5 + len(confidence_factors) * 0.08)
        else:
            final_signal = 'HOLD'
            option_type = None
            confidence = 0.3
            reasons.append("Mixed signals - no clear direction")
        
        return {
            'timestamp': datetime.now(),
            'final_signal': final_signal,
            'option_type': option_type,
            'confidence': confidence,
            'signal_score': signal_score,
            'reasons': reasons,
            'technical_data': {
                'ema_6': ema_6,
                'ema_15': ema_15,
                'macd': macd,
                'rsi': rsi,
                'current_price': current_price
            }
        }
    
    def _assess_market_risk(self, market_data, current_positions):
        """Assess current market risk"""
        try:
            # Get VIX level
            vix_value = market_data.get('india_vix', {}).get('vix_value', 20)
            
            # Get market change
            nifty_change = market_data.get('nifty', {}).get('change_percent', 0)
            
            # Calculate risk score (0-1)
            risk_score = 0
            
            # VIX risk component
            if vix_value > 30:
                risk_score += 0.4
            elif vix_value > 25:
                risk_score += 0.2
            elif vix_value < 15:
                risk_score += 0.1  # Low volatility can be risky too
            
            # Market movement risk
            if abs(nifty_change) > 2:
                risk_score += 0.3
            elif abs(nifty_change) > 1:
                risk_score += 0.2
            
            # Position concentration risk
            if current_positions and len(current_positions) > 3:
                risk_score += 0.2
            
            return min(1.0, risk_score)
            
        except Exception as e:
            print(f"Risk assessment error: {e}")
            return 0.5  # Medium risk default
    
    def _create_hold_signal(self, reason="Unknown"):
        """Create a HOLD signal"""
        return {
            'timestamp': datetime.now(),
            'final_signal': 'HOLD',
            'option_type': None,
            'confidence': 0.1,
            'signal_score': 0,
            'reasons': [reason],
            'risk_score': 0.5,
            'technical_data': {}
        }
    
    def get_signal_summary(self):
        """Get summary of recent signals"""
        if not self.signals_history:
            return {"status": "No signals generated yet"}
        
        recent_signals = self.signals_history[-10:]
        
        buy_signals = sum(1 for s in recent_signals if s['final_signal'] == 'BUY')
        sell_signals = sum(1 for s in recent_signals if s['final_signal'] == 'SELL')
        hold_signals = sum(1 for s in recent_signals if s['final_signal'] == 'HOLD')
        
        avg_confidence = np.mean([s['confidence'] for s in recent_signals])
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'average_confidence': avg_confidence,
            'latest_signal': recent_signals[-1]['final_signal'] if recent_signals else 'NONE'
        }


if __name__ == "__main__":
    # Test the ensemble
    print("Testing Trading Signal Ensemble...")
    
    ensemble = TradingSignalEnsemble(capital=10000)
    
    # Create mock market data
    mock_data = {
        'nifty': {
            'ltp': 25000,
            'change': 150,
            'change_percent': 0.6
        },
        'india_vix': {
            'vix_value': 18.5
        }
    }
    
    # Generate signal
    signal = ensemble.generate_ensemble_signal(mock_data)
    print(f"Generated signal: {signal['final_signal']} with confidence: {signal['confidence']:.2f}")
    print(f"Reasons: {', '.join(signal['reasons'])}")