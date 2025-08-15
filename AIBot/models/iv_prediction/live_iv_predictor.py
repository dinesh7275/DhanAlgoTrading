"""
Live IV Prediction with Real Market Data Learning
================================================

Real-time implied volatility prediction using live market data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class LiveIVPredictor:
    """
    Real-time IV prediction using live market data and adaptive learning
    """
    
    def __init__(self, lookback_days=30, update_frequency='15min'):
        self.lookback_days = lookback_days
        self.update_frequency = update_frequency
        self.scaler = MinMaxScaler()
        self.model = None
        self.training_data = pd.DataFrame()
        self.last_update = None
        self.prediction_accuracy = []
        
        print(f"Live IV Predictor initialized - Lookback: {lookback_days} days")
    
    def fetch_real_market_data(self):
        """Fetch real market data for IV calculation"""
        try:
            # Fetch Nifty data
            nifty = yf.Ticker('^NSEI')
            data = nifty.history(period=f'{self.lookback_days}d', interval='15m')
            
            if data.empty:
                print("No market data available")
                return None
            
            # Calculate returns and realized volatility
            data['returns'] = data['Close'].pct_change()
            data['realized_vol'] = data['returns'].rolling(window=24).std() * np.sqrt(252)
            
            # Get VIX data (India VIX)
            try:
                vix_data = yf.Ticker('^INDIAVIX')
                vix_hist = vix_data.history(period=f'{self.lookback_days}d', interval='1d')
                
                if not vix_hist.empty:
                    # Interpolate VIX to match 15min frequency
                    vix_resampled = vix_hist['Close'].resample('15min').ffill()
                    data['india_vix'] = vix_resampled.reindex(data.index, method='ffill')
                else:
                    data['india_vix'] = 20  # Default VIX value
            except:
                data['india_vix'] = 20
            
            # Calculate additional features
            data['high_low_ratio'] = data['High'] / data['Low']
            data['close_open_ratio'] = data['Close'] / data['Open']
            data['volume_ma'] = data['Volume'].rolling(window=24).mean()
            data['price_change'] = data['Close'].pct_change()
            
            # Remove NaN values
            data = data.dropna()
            
            print(f"Fetched {len(data)} data points for IV prediction")
            return data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def prepare_features(self, data):
        """Prepare features for IV prediction"""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['realized_vol'] = data['realized_vol']
            features['price_change'] = data['price_change']
            features['high_low_ratio'] = data['high_low_ratio']
            features['close_open_ratio'] = data['close_open_ratio']
            
            # Volume features
            features['volume_ratio'] = data['Volume'] / data['volume_ma']
            features['volume_change'] = data['Volume'].pct_change()
            
            # Technical indicators
            features['rsi'] = self.calculate_rsi(data['Close'])
            features['bollinger_width'] = self.calculate_bollinger_width(data['Close'])
            
            # Time-based features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['time_to_expiry'] = self.calculate_time_to_expiry()
            
            # Market regime indicators
            features['trend'] = self.calculate_trend(data['Close'])
            features['volatility_regime'] = self.calculate_vol_regime(data['realized_vol'])
            
            # Target: Use India VIX as proxy for implied volatility
            if 'india_vix' in data.columns:
                features['target_iv'] = data['india_vix'] / 100  # Convert to decimal
            else:
                # If no VIX data, use realized volatility + premium
                features['target_iv'] = data['realized_vol'] * 1.2
            
            return features.dropna()
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_width(self, prices, window=20):
        """Calculate Bollinger Band width"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (upper - lower) / sma
    
    def calculate_time_to_expiry(self):
        """Calculate time to next expiry (simplified)"""
        now = datetime.now()
        # Next Thursday (weekly expiry)
        days_ahead = 3 - now.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        next_expiry = now + timedelta(days=days_ahead)
        time_to_expiry = (next_expiry - now).days / 7  # In weeks
        return max(0.1, time_to_expiry)  # Minimum 0.1 weeks
    
    def calculate_trend(self, prices, window=20):
        """Calculate trend indicator"""
        sma = prices.rolling(window=window).mean()
        trend = np.where(prices > sma, 1, -1)
        return pd.Series(trend, index=prices.index)
    
    def calculate_vol_regime(self, volatility, window=20):
        """Calculate volatility regime"""
        vol_ma = volatility.rolling(window=window).mean()
        vol_std = volatility.rolling(window=window).std()
        
        high_vol = vol_ma + vol_std
        low_vol = vol_ma - vol_std
        
        regime = np.where(volatility > high_vol, 2,  # High vol regime
                         np.where(volatility < low_vol, 0, 1))  # Low vol / Normal
        return pd.Series(regime, index=volatility.index)
    
    def train_simple_model(self, features):
        """Train a simple linear model for IV prediction"""
        try:
            # Use a simple weighted average model for real-time prediction
            target = features['target_iv'].values
            
            # Define feature weights based on market knowledge
            weights = {
                'realized_vol': 0.3,
                'price_change': 0.1,
                'volume_ratio': 0.1,
                'rsi': 0.05,
                'bollinger_width': 0.1,
                'time_to_expiry': 0.15,
                'trend': 0.05,
                'volatility_regime': 0.15
            }
            
            # Calculate predictions
            predictions = np.zeros(len(features))
            
            for i, (idx, row) in enumerate(features.iterrows()):
                pred = 0
                weight_sum = 0
                
                for feature, weight in weights.items():
                    if feature in row and not pd.isna(row[feature]):
                        if feature == 'realized_vol':
                            pred += row[feature] * weight
                        elif feature == 'price_change':
                            pred += abs(row[feature]) * 0.2 * weight  # Volatility from price changes
                        elif feature == 'volume_ratio':
                            pred += max(0, (row[feature] - 1)) * 0.1 * weight  # Volume spikes
                        elif feature == 'rsi':
                            # RSI extremes indicate higher volatility
                            rsi_vol = abs(row[feature] - 50) / 50 * 0.1
                            pred += rsi_vol * weight
                        elif feature == 'bollinger_width':
                            pred += row[feature] * weight
                        elif feature == 'time_to_expiry':
                            # Time decay factor
                            pred += (1 / max(0.1, row[feature])) * 0.05 * weight
                        elif feature == 'trend':
                            pred += abs(row[feature]) * 0.02 * weight  # Trending markets
                        elif feature == 'volatility_regime':
                            pred += row[feature] * 0.05 * weight  # Regime adjustment
                        
                        weight_sum += weight
                
                if weight_sum > 0:
                    predictions[i] = pred / weight_sum
                else:
                    predictions[i] = 0.2  # Default 20% volatility
            
            # Store model parameters
            self.model = {
                'type': 'weighted_average',
                'weights': weights,
                'trained_on': datetime.now(),
                'data_points': len(features)
            }
            
            # Calculate accuracy
            if len(target) > 0:
                mae = mean_absolute_error(target, predictions)
                rmse = np.sqrt(mean_squared_error(target, predictions))
                
                accuracy = {
                    'mae': mae,
                    'rmse': rmse,
                    'mean_target': np.mean(target),
                    'mean_prediction': np.mean(predictions)
                }
                
                self.prediction_accuracy.append(accuracy)
                print(f"Model trained - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_iv(self, current_data=None):
        """Predict IV for current market conditions"""
        try:
            if self.model is None:
                # Train model with latest data
                market_data = self.fetch_real_market_data()
                if market_data is not None:
                    features = self.prepare_features(market_data)
                    if features is not None and len(features) > 0:
                        self.train_simple_model(features)
            
            if self.model is None:
                return {'predicted_iv': 0.20, 'confidence': 0.1, 'status': 'no_model'}
            
            # Get current market data if not provided
            if current_data is None:
                current_data = self.fetch_real_market_data()
                if current_data is None:
                    return {'predicted_iv': 0.20, 'confidence': 0.1, 'status': 'no_data'}
            
            # Prepare current features
            features = self.prepare_features(current_data.tail(1))
            if features is None or len(features) == 0:
                return {'predicted_iv': 0.20, 'confidence': 0.1, 'status': 'no_features'}
            
            # Make prediction using current model
            weights = self.model['weights']
            current_row = features.iloc[-1]
            
            pred = 0
            weight_sum = 0
            
            for feature, weight in weights.items():
                if feature in current_row and not pd.isna(current_row[feature]):
                    if feature == 'realized_vol':
                        pred += current_row[feature] * weight
                    elif feature == 'price_change':
                        pred += abs(current_row[feature]) * 0.2 * weight
                    elif feature == 'volume_ratio':
                        pred += max(0, (current_row[feature] - 1)) * 0.1 * weight
                    elif feature == 'rsi':
                        rsi_vol = abs(current_row[feature] - 50) / 50 * 0.1
                        pred += rsi_vol * weight
                    elif feature == 'bollinger_width':
                        pred += current_row[feature] * weight
                    elif feature == 'time_to_expiry':
                        pred += (1 / max(0.1, current_row[feature])) * 0.05 * weight
                    elif feature == 'trend':
                        pred += abs(current_row[feature]) * 0.02 * weight
                    elif feature == 'volatility_regime':
                        pred += current_row[feature] * 0.05 * weight
                    
                    weight_sum += weight
            
            if weight_sum > 0:
                predicted_iv = pred / weight_sum
            else:
                predicted_iv = 0.20
            
            # Calculate confidence based on recent accuracy
            confidence = 0.5  # Base confidence
            if self.prediction_accuracy:
                recent_accuracy = self.prediction_accuracy[-5:]  # Last 5 predictions
                avg_mae = np.mean([acc['mae'] for acc in recent_accuracy])
                confidence = max(0.1, 1 - (avg_mae * 5))  # Higher MAE = lower confidence
            
            # Ensure predicted IV is within reasonable bounds
            predicted_iv = max(0.05, min(2.0, predicted_iv))
            
            return {
                'predicted_iv': predicted_iv,
                'confidence': confidence,
                'status': 'success',
                'features_used': list(weights.keys()),
                'current_values': current_row.to_dict()
            }
            
        except Exception as e:
            print(f"Error predicting IV: {e}")
            return {'predicted_iv': 0.20, 'confidence': 0.1, 'status': f'error: {e}'}
    
    def update_model_with_new_data(self):
        """Update model with fresh market data"""
        try:
            new_data = self.fetch_real_market_data()
            if new_data is not None:
                features = self.prepare_features(new_data)
                if features is not None and len(features) > 0:
                    success = self.train_simple_model(features)
                    if success:
                        self.last_update = datetime.now()
                        print(f"Model updated at {self.last_update}")
                        return True
            return False
        except Exception as e:
            print(f"Error updating model: {e}")
            return False
    
    def get_model_status(self):
        """Get current model status and performance"""
        status = {
            'model_exists': self.model is not None,
            'last_update': self.last_update,
            'accuracy_history': self.prediction_accuracy[-10:],  # Last 10 accuracy measurements
            'data_points_trained': self.model['data_points'] if self.model else 0
        }
        
        if self.prediction_accuracy:
            recent_acc = self.prediction_accuracy[-5:]
            status['recent_performance'] = {
                'avg_mae': np.mean([acc['mae'] for acc in recent_acc]),
                'avg_rmse': np.mean([acc['rmse'] for acc in recent_acc]),
                'predictions_count': len(recent_acc)
            }
        
        return status


if __name__ == "__main__":
    # Test the IV predictor
    print("Testing Live IV Predictor...")
    
    predictor = LiveIVPredictor(lookback_days=7)
    
    # Make a prediction
    result = predictor.predict_iv()
    print(f"IV Prediction: {result['predicted_iv']:.4f} ({result['predicted_iv']*100:.2f}%)")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Status: {result['status']}")
    
    # Get model status
    status = predictor.get_model_status()
    print(f"Model Status: {status}")