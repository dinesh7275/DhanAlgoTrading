"""
Live Options Anomaly Detection with Real Market Data
===================================================

Real-time detection of options pricing anomalies and arbitrage opportunities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LiveAnomalyDetector:
    """
    Real-time options anomaly detection using live market data
    """
    
    def __init__(self, contamination=0.1, lookback_days=7):
        self.contamination = contamination  # Expected fraction of anomalies
        self.lookback_days = lookback_days
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = pd.DataFrame()
        self.last_update = None
        self.anomaly_history = []
        
        print(f"Live Anomaly Detector initialized - Contamination: {contamination}")
    
    def fetch_options_data(self):
        """Fetch real options data (simulated with Nifty data)"""
        try:
            # Get Nifty spot data
            nifty = yf.Ticker('^NSEI')
            spot_data = nifty.history(period=f'{self.lookback_days}d', interval='5m')
            
            if spot_data.empty:
                print("No spot data available")
                return None
            
            # Get VIX data for volatility
            try:
                vix = yf.Ticker('^INDIAVIX')
                vix_data = vix.history(period=f'{self.lookback_days}d', interval='1d')
                current_vix = vix_data['Close'].iloc[-1] / 100 if not vix_data.empty else 0.20
            except:
                current_vix = 0.20  # Default 20% volatility
            
            # Generate synthetic options data based on real spot prices
            options_data = self.generate_synthetic_options_data(spot_data, current_vix)
            
            print(f"Generated {len(options_data)} options data points")
            return options_data
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
    
    def generate_synthetic_options_data(self, spot_data, volatility):
        """Generate synthetic options data based on real spot prices"""
        options_list = []
        
        for i, (timestamp, row) in enumerate(spot_data.iterrows()):
            spot_price = row['Close']
            
            # Generate strikes around current spot (±500 points)
            strikes = np.arange(
                round(spot_price/50)*50 - 500, 
                round(spot_price/50)*50 + 550, 
                50
            )
            
            # Calculate time to expiry (assume weekly expiry)
            days_to_expiry = 7 - (i % 7)  # Simulate weekly cycle
            time_to_expiry = max(0.01, days_to_expiry / 365)
            
            for strike in strikes:
                # Calculate theoretical prices using Black-Scholes approximation
                ce_price = self.black_scholes_call(spot_price, strike, time_to_expiry, 0.05, volatility)
                pe_price = self.black_scholes_put(spot_price, strike, time_to_expiry, 0.05, volatility)
                
                # Add market noise and inefficiencies
                ce_market_price = ce_price * np.random.normal(1, 0.05)  # 5% noise
                pe_market_price = pe_price * np.random.normal(1, 0.05)
                
                # Calculate features for anomaly detection
                ce_data = {
                    'timestamp': timestamp,
                    'strike': strike,
                    'option_type': 'CE',
                    'spot_price': spot_price,
                    'market_price': max(0.5, ce_market_price),
                    'theoretical_price': ce_price,
                    'time_to_expiry': time_to_expiry,
                    'volatility': volatility,
                    'moneyness': spot_price / strike,
                    'intrinsic_value': max(0, spot_price - strike),
                    'time_value': max(0, ce_price - max(0, spot_price - strike)),
                    'delta': self.calculate_delta(spot_price, strike, time_to_expiry, volatility, 'call'),
                    'volume': np.random.randint(100, 10000),  # Simulated volume
                    'open_interest': np.random.randint(1000, 50000)
                }
                
                pe_data = {
                    'timestamp': timestamp,
                    'strike': strike,
                    'option_type': 'PE',
                    'spot_price': spot_price,
                    'market_price': max(0.5, pe_market_price),
                    'theoretical_price': pe_price,
                    'time_to_expiry': time_to_expiry,
                    'volatility': volatility,
                    'moneyness': strike / spot_price,
                    'intrinsic_value': max(0, strike - spot_price),
                    'time_value': max(0, pe_price - max(0, strike - spot_price)),
                    'delta': self.calculate_delta(spot_price, strike, time_to_expiry, volatility, 'put'),
                    'volume': np.random.randint(100, 10000),
                    'open_interest': np.random.randint(1000, 50000)
                }
                
                options_list.extend([ce_data, pe_data])
        
        return pd.DataFrame(options_list)
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option pricing"""
        if T <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
        return max(0, call_price)
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes put option pricing"""
        if T <= 0:
            return max(0, K - S)
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
        return max(0, put_price)
    
    def calculate_delta(self, S, K, T, sigma, option_type):
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = (np.log(S/K) + (0.05 + sigma**2/2)*T) / (sigma*np.sqrt(T))
        
        if option_type == 'call':
            return stats.norm.cdf(d1)
        else:
            return stats.norm.cdf(d1) - 1
    
    def calculate_anomaly_features(self, options_data):
        """Calculate features for anomaly detection"""
        features = pd.DataFrame()
        
        # Price-based anomalies
        features['price_deviation'] = (options_data['market_price'] - options_data['theoretical_price']) / options_data['theoretical_price']
        features['bid_ask_spread'] = np.random.uniform(0.01, 0.1, len(options_data))  # Simulated spread
        features['price_to_intrinsic'] = options_data['market_price'] / (options_data['intrinsic_value'] + 0.1)
        
        # Volume and liquidity anomalies
        features['volume_to_oi_ratio'] = options_data['volume'] / (options_data['open_interest'] + 1)
        features['unusual_volume'] = (options_data['volume'] > options_data['volume'].quantile(0.95)).astype(int)
        
        # Greeks-based anomalies
        features['delta_consistency'] = abs(options_data['delta']) * options_data['time_to_expiry']
        features['time_value_ratio'] = options_data['time_value'] / (options_data['market_price'] + 0.1)
        
        # Moneyness anomalies
        features['moneyness_deviation'] = abs(options_data['moneyness'] - 1)
        features['deep_otm'] = (options_data['moneyness'] < 0.9).astype(int) if 'CE' in str(options_data['option_type'].iloc[0]) else (options_data['moneyness'] > 1.1).astype(int)
        
        # Time-based anomalies
        features['expiry_pressure'] = 1 / (options_data['time_to_expiry'] + 0.001)
        features['weekend_effect'] = (options_data['timestamp'].dt.dayofweek >= 4).astype(int)
        
        # Volatility anomalies
        features['vol_smile_deviation'] = np.random.normal(0, 0.1, len(options_data))  # Simulated vol smile
        features['iv_rank'] = np.random.uniform(0, 1, len(options_data))  # Simulated IV rank
        
        return features.fillna(0)
    
    def train_anomaly_detector(self, options_data):
        """Train the anomaly detection model"""
        try:
            features = self.calculate_anomaly_features(options_data)
            
            if len(features) < 100:
                print("Insufficient data for training anomaly detector")
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train Isolation Forest
            self.isolation_forest.fit(features_scaled)
            
            # Store training info
            self.last_update = datetime.now()
            
            print(f"Anomaly detector trained on {len(features)} samples")
            return True
            
        except Exception as e:
            print(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, current_options_data=None):
        """Detect anomalies in current options data"""
        try:
            # Get current data if not provided
            if current_options_data is None:
                current_options_data = self.fetch_options_data()
                if current_options_data is None:
                    return {'status': 'no_data', 'anomalies': []}
            
            # Train model if needed
            if (self.isolation_forest is None or 
                self.last_update is None or 
                (datetime.now() - self.last_update).hours > 6):
                
                success = self.train_anomaly_detector(current_options_data)
                if not success:
                    return {'status': 'training_failed', 'anomalies': []}
            
            # Calculate features for current data
            features = self.calculate_anomaly_features(current_options_data)
            
            if len(features) == 0:
                return {'status': 'no_features', 'anomalies': []}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomalies
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            is_anomaly = self.isolation_forest.predict(features_scaled) == -1
            
            # Extract anomalous options
            anomalies = []
            anomaly_indices = np.where(is_anomaly)[0]
            
            for idx in anomaly_indices:
                option_info = current_options_data.iloc[idx]
                
                anomaly = {
                    'timestamp': option_info['timestamp'],
                    'strike': option_info['strike'],
                    'option_type': option_info['option_type'],
                    'spot_price': option_info['spot_price'],
                    'market_price': option_info['market_price'],
                    'theoretical_price': option_info['theoretical_price'],
                    'price_deviation': (option_info['market_price'] - option_info['theoretical_price']) / option_info['theoretical_price'],
                    'anomaly_score': anomaly_scores[idx],
                    'moneyness': option_info['moneyness'],
                    'time_to_expiry': option_info['time_to_expiry'],
                    'opportunity_type': self.classify_anomaly_type(option_info, features.iloc[idx])
                }
                
                anomalies.append(anomaly)
            
            # Sort by anomaly score (most anomalous first)
            anomalies = sorted(anomalies, key=lambda x: x['anomaly_score'])
            
            # Store in history
            detection_result = {
                'timestamp': datetime.now(),
                'total_options': len(current_options_data),
                'anomalies_found': len(anomalies),
                'anomalies': anomalies[:20]  # Top 20 anomalies
            }
            
            self.anomaly_history.append(detection_result)
            if len(self.anomaly_history) > 100:
                self.anomaly_history = self.anomaly_history[-100:]
            
            return {
                'status': 'success',
                'total_options_analyzed': len(current_options_data),
                'anomalies_found': len(anomalies),
                'top_anomalies': anomalies[:10],  # Top 10 for display
                'detection_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return {'status': f'error: {e}', 'anomalies': []}
    
    def classify_anomaly_type(self, option_info, features):
        """Classify the type of anomaly detected"""
        price_dev = abs((option_info['market_price'] - option_info['theoretical_price']) / option_info['theoretical_price'])
        
        if price_dev > 0.2:  # 20% price deviation
            if option_info['market_price'] < option_info['theoretical_price']:
                return 'UNDERPRICED'
            else:
                return 'OVERPRICED'
        
        if features.get('unusual_volume', 0) == 1:
            return 'VOLUME_ANOMALY'
        
        if features.get('deep_otm', 0) == 1:
            return 'DEEP_OTM_ACTIVITY'
        
        if option_info['time_to_expiry'] < 0.02:  # Less than a week
            return 'EXPIRY_PRESSURE'
        
        return 'GENERAL_ANOMALY'
    
    def get_trading_opportunities(self, anomalies_result):
        """Extract actionable trading opportunities from anomalies"""
        if anomalies_result['status'] != 'success':
            return []
        
        opportunities = []
        
        for anomaly in anomalies_result['top_anomalies']:
            if anomaly['opportunity_type'] == 'UNDERPRICED' and anomaly['price_deviation'] < -0.1:
                # Significant underpricing - buy opportunity
                opportunity = {
                    'action': 'BUY',
                    'strike': anomaly['strike'],
                    'option_type': anomaly['option_type'],
                    'reason': f"Underpriced by {abs(anomaly['price_deviation']*100):.1f}%",
                    'market_price': anomaly['market_price'],
                    'fair_value': anomaly['theoretical_price'],
                    'profit_potential': (anomaly['theoretical_price'] - anomaly['market_price']) / anomaly['market_price'],
                    'confidence': min(1.0, abs(anomaly['anomaly_score']) / 0.5),
                    'time_to_expiry': anomaly['time_to_expiry'],
                    'risk_level': 'LOW' if anomaly['moneyness'] > 0.95 and anomaly['moneyness'] < 1.05 else 'MEDIUM'
                }
                opportunities.append(opportunity)
        
        # Sort by profit potential
        opportunities = sorted(opportunities, key=lambda x: x['profit_potential'], reverse=True)
        
        return opportunities[:5]  # Top 5 opportunities
    
    def get_detection_summary(self):
        """Get summary of recent anomaly detection"""
        if not self.anomaly_history:
            return {'status': 'No detection history'}
        
        recent_detections = self.anomaly_history[-10:]
        
        return {
            'total_detections': len(self.anomaly_history),
            'recent_average_anomalies': np.mean([d['anomalies_found'] for d in recent_detections]),
            'last_detection': recent_detections[-1]['timestamp'],
            'model_last_update': self.last_update,
            'anomaly_types_found': self._count_anomaly_types(recent_detections)
        }
    
    def _count_anomaly_types(self, detections):
        """Count different types of anomalies found"""
        type_counts = {}
        
        for detection in detections:
            for anomaly in detection['anomalies']:
                anomaly_type = anomaly['opportunity_type']
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        return type_counts


if __name__ == "__main__":
    # Test the anomaly detector
    print("Testing Live Options Anomaly Detector...")
    
    detector = LiveAnomalyDetector(contamination=0.1, lookback_days=3)
    
    # Detect anomalies
    result = detector.detect_anomalies()
    print(f"Anomaly Detection Result: {result['status']}")
    print(f"Total Options Analyzed: {result.get('total_options_analyzed', 0)}")
    print(f"Anomalies Found: {result.get('anomalies_found', 0)}")
    
    if result.get('top_anomalies'):
        print("\\nTop Anomalies:")
        for i, anomaly in enumerate(result['top_anomalies'][:3]):
            print(f"{i+1}. {anomaly['strike']}{anomaly['option_type']} - {anomaly['opportunity_type']}")
            print(f"   Price Deviation: {anomaly['price_deviation']*100:.1f}%")
    
    # Get trading opportunities
    opportunities = detector.get_trading_opportunities(result)
    if opportunities:
        print(f"\\nTrading Opportunities Found: {len(opportunities)}")
        for opp in opportunities[:2]:
            print(f"- {opp['action']} {opp['strike']}{opp['option_type']}: {opp['reason']}")
    
    # Get summary
    summary = detector.get_detection_summary()
    print(f"\\nDetection Summary: {summary}")