"""
Live Price Movement Prediction with Real Market Data
===================================================

Real-time price movement prediction using live market data and pattern recognition
Optimized for Indian options trading with ₹10,000 capital
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class LivePricePredictor:
    """
    Real-time price movement prediction using live market data
    """
    
    def __init__(self, lookback_days=15, prediction_horizon='15min'):
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.training_data = pd.DataFrame()
        self.last_update = None
        self.prediction_accuracy = []
        self.feature_importance = {}
        
        print(f"Live Price Predictor initialized - Lookback: {lookback_days} days")
    
    def fetch_real_market_data(self):
        """Fetch real Nifty market data for prediction"""
        try:
            # Fetch Nifty data with higher frequency
            nifty = yf.Ticker('^NSEI')
            data = nifty.history(period=f'{self.lookback_days}d', interval='5m')
            
            if data.empty:
                print("No market data available")
                return None
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Calculate target (price movement direction)
            data = self.calculate_movement_targets(data)
            
            print(f"Fetched {len(data)} data points for price prediction")
            return data.dropna()
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        # Price-based indicators
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        # EMA 6 and 15 (key for options trading)
        data['ema_6'] = data['Close'].ewm(span=6).mean()
        data['ema_15'] = data['Close'].ewm(span=15).mean()
        data['ema_6_15_diff'] = data['ema_6'] - data['ema_15']
        data['ema_6_15_ratio'] = data['ema_6'] / data['ema_15']
        
        # Price position relative to MAs
        data['price_above_sma20'] = (data['Close'] > data['sma_20']).astype(int)
        data['price_above_ema20'] = (data['Close'] > data['ema_20']).astype(int)
        
        # RSI
        data['rsi'] = self.calculate_rsi(data['Close'])
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        
        # MACD
        data['macd'], data['macd_signal'], data['macd_histogram'] = self.calculate_macd(data['Close'])
        data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = self.calculate_bollinger_bands(data['Close'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        data['volume_spike'] = (data['volume_ratio'] > 2).astype(int)
        
        # Volatility
        data['realized_vol'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        data['high_vol'] = (data['realized_vol'] > data['realized_vol'].rolling(50).mean()).astype(int)
        
        # Price patterns
        data['higher_high'] = ((data['High'] > data['High'].shift(1)) & 
                              (data['High'].shift(1) > data['High'].shift(2))).astype(int)
        data['lower_low'] = ((data['Low'] < data['Low'].shift(1)) & 
                            (data['Low'].shift(1) < data['Low'].shift(2))).astype(int)
        
        # Gap analysis
        data['gap_up'] = (data['Open'] > data['Close'].shift(1) * 1.002).astype(int)
        data['gap_down'] = (data['Open'] < data['Close'].shift(1) * 0.998).astype(int)
        
        # Time-based features
        data['hour'] = data.index.hour
        data['is_opening'] = (data['hour'] == 9).astype(int)
        data['is_closing'] = (data['hour'] >= 15).astype(int)
        data['day_of_week'] = data.index.dayofweek
        
        return data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        return upper, sma, lower
    
    def calculate_movement_targets(self, data):
        """Calculate price movement targets for classification"""
        # Define movement thresholds (optimized for options trading)
        move_threshold = 0.003  # 0.3% movement threshold
        
        # Calculate future returns
        future_periods = 3  # Look ahead 3 periods (15 minutes for 5min data)
        data['future_return'] = data['Close'].shift(-future_periods) / data['Close'] - 1
        
        # Create classification targets
        data['movement_class'] = 0  # No significant movement
        data.loc[data['future_return'] > move_threshold, 'movement_class'] = 1  # Bullish
        data.loc[data['future_return'] < -move_threshold, 'movement_class'] = -1  # Bearish
        
        # Binary classification for simplicity
        data['bullish_target'] = (data['movement_class'] == 1).astype(int)
        data['bearish_target'] = (data['movement_class'] == -1).astype(int)
        data['direction_target'] = np.where(data['future_return'] > 0, 1, 0)  # Up/Down
        
        return data
    
    def prepare_features_for_training(self, data):
        """Prepare feature matrix for training"""
        # Select relevant features
        feature_columns = [
            'returns', 'log_returns',
            'ema_6_15_diff', 'ema_6_15_ratio',
            'price_above_sma20', 'price_above_ema20',
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'macd_histogram', 'macd_bullish',
            'bb_width', 'bb_position',
            'volume_ratio', 'volume_spike',
            'realized_vol', 'high_vol',
            'higher_high', 'lower_low',
            'gap_up', 'gap_down',
            'is_opening', 'is_closing', 'day_of_week'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Create feature matrix
        X = data[available_features].copy()
        
        # Create targets
        y_direction = data['direction_target'].copy()
        y_bullish = data['bullish_target'].copy()
        y_bearish = data['bearish_target'].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y_direction.isnull())
        X = X[valid_mask]
        y_direction = y_direction[valid_mask]
        y_bullish = y_bullish[valid_mask]
        y_bearish = y_bearish[valid_mask]
        
        return X, y_direction, y_bullish, y_bearish, available_features
    
    def train_prediction_model(self, data):
        """Train the price movement prediction model"""
        try:
            X, y_direction, y_bullish, y_bearish, feature_names = self.prepare_features_for_training(data)
            
            if len(X) < 100:  # Need minimum data for training
                print("Insufficient data for training")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model for direction prediction
            self.model.fit(X_scaled, y_direction)
            
            # Calculate feature importance
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            # Evaluate model
            predictions = self.model.predict(X_scaled)
            accuracy = accuracy_score(y_direction, predictions)
            
            # Store training info
            training_info = {
                'timestamp': datetime.now(),
                'data_points': len(X),
                'features_used': feature_names,
                'accuracy': accuracy,
                'feature_importance': self.feature_importance
            }
            
            self.prediction_accuracy.append(training_info)
            self.last_update = datetime.now()
            
            print(f"Model trained - Accuracy: {accuracy:.3f}, Features: {len(feature_names)}")
            print(f"Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_price_movement(self, current_data=None):
        """Predict price movement for current market conditions"""
        try:
            # Get current market data if not provided
            if current_data is None:
                current_data = self.fetch_real_market_data()
                if current_data is None:
                    return self._create_default_prediction("No market data")
            
            # Train model if not available or outdated
            if (self.model is None or 
                self.last_update is None or 
                (datetime.now() - self.last_update).hours > 4):
                
                success = self.train_prediction_model(current_data)
                if not success:
                    return self._create_default_prediction("Model training failed")
            
            # Prepare current features
            X, _, _, _, feature_names = self.prepare_features_for_training(current_data.tail(1))
            
            if len(X) == 0:
                return self._create_default_prediction("No valid features")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            direction_prob = self.model.predict_proba(X_scaled)[0]
            direction_pred = self.model.predict(X_scaled)[0]
            
            # Calculate confidence
            confidence = max(direction_prob)
            
            # Determine signal
            if direction_pred == 1 and confidence > 0.6:
                signal = 'BULLISH'
                option_recommendation = 'CE'
            elif direction_pred == 0 and confidence > 0.6:
                signal = 'BEARISH'
                option_recommendation = 'PE'
            else:
                signal = 'NEUTRAL'
                option_recommendation = None
            
            # Get current market context
            current_row = current_data.iloc[-1]
            
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'option_recommendation': option_recommendation,
                'confidence': confidence,
                'direction_probability': {
                    'up': direction_prob[1] if len(direction_prob) > 1 else 0.5,
                    'down': direction_prob[0] if len(direction_prob) > 1 else 0.5
                },
                'current_price': current_row['Close'],
                'key_indicators': {
                    'ema_6_15_signal': 'bullish' if current_row.get('ema_6_15_diff', 0) > 0 else 'bearish',
                    'rsi': current_row.get('rsi', 50),
                    'macd_signal': 'bullish' if current_row.get('macd_bullish', 0) == 1 else 'bearish',
                    'volume_spike': bool(current_row.get('volume_spike', 0))
                },
                'feature_importance': dict(sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True)[:5]),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error predicting price movement: {e}")
            return self._create_default_prediction(f"Error: {e}")
    
    def _create_default_prediction(self, reason):
        """Create default prediction when model fails"""
        return {
            'timestamp': datetime.now(),
            'signal': 'NEUTRAL',
            'option_recommendation': None,
            'confidence': 0.1,
            'direction_probability': {'up': 0.5, 'down': 0.5},
            'current_price': 0,
            'key_indicators': {},
            'feature_importance': {},
            'status': f'default: {reason}'
        }
    
    def update_model_with_new_data(self):
        """Update model with fresh market data"""
        try:
            new_data = self.fetch_real_market_data()
            if new_data is not None and len(new_data) > 100:
                success = self.train_prediction_model(new_data)
                if success:
                    print(f"Model updated with {len(new_data)} new data points")
                    return True
            return False
        except Exception as e:
            print(f"Error updating model: {e}")
            return False
    
    def get_model_performance(self):
        """Get model performance metrics"""
        if not self.prediction_accuracy:
            return {"status": "No performance data available"}
        
        recent_performance = self.prediction_accuracy[-5:]  # Last 5 training sessions
        
        return {
            'latest_accuracy': recent_performance[-1]['accuracy'],
            'average_accuracy': np.mean([p['accuracy'] for p in recent_performance]),
            'total_trainings': len(self.prediction_accuracy),
            'last_update': self.last_update,
            'features_count': len(recent_performance[-1]['features_used']),
            'top_features': dict(sorted(self.feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:10])
        }


if __name__ == "__main__":
    # Test the price predictor
    print("Testing Live Price Movement Predictor...")
    
    predictor = LivePricePredictor(lookback_days=5)
    
    # Make a prediction
    result = predictor.predict_price_movement()
    print(f"Price Movement Prediction: {result['signal']}")
    print(f"Option Recommendation: {result['option_recommendation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Direction Probabilities: Up: {result['direction_probability']['up']:.2f}, Down: {result['direction_probability']['down']:.2f}")
    print(f"Status: {result['status']}")
    
    # Get performance metrics
    performance = predictor.get_model_performance()
    print(f"Model Performance: {performance}")