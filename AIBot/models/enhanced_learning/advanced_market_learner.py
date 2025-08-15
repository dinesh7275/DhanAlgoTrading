#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Market Learning System
==============================

Enhanced ML system that learns from 30-day historical data across multiple timeframes
with real-time adaptation and candlestick pattern recognition.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

class AdvancedMarketLearner:
    """
    Advanced machine learning system for market analysis and prediction
    """
    
    def __init__(self, symbol: str = "^NSEI", lookback_days: int = 30):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.timeframes = ['1m', '5m', '15m', '1h', '1d']
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Create models directory
        self.models_dir = Path("models/enhanced_learning/saved_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model architecture configurations
        self.model_configs = {
            'lstm': {
                'layers': [64, 32],
                'dropout': 0.2,
                'lookback': 20
            },
            'cnn_lstm': {
                'cnn_filters': [32, 16],
                'cnn_kernel': 3,
                'lstm_units': 50,
                'lookback': 30
            },
            'ensemble': {
                'rf_trees': 100,
                'gb_estimators': 100,
                'mlp_layers': [100, 50, 25]
            }
        }
        
        logger.info(f"AdvancedMarketLearner initialized for {symbol}")
    
    def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple timeframes
        """
        logger.info("Fetching multi-timeframe data...")
        
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        for timeframe in self.timeframes:
            try:
                if timeframe == '1d':
                    # Daily data
                    df = yf.download(self.symbol, start=start_date, end=end_date, interval='1d')
                elif timeframe == '1h':
                    # Hourly data
                    df = yf.download(self.symbol, start=start_date, end=end_date, interval='1h')
                elif timeframe == '15m':
                    # 15-minute data
                    df = yf.download(self.symbol, start=start_date, end=end_date, interval='15m')
                elif timeframe == '5m':
                    # 5-minute data (limited to last 7 days due to Yahoo Finance limits)
                    recent_start = end_date - timedelta(days=7)
                    df = yf.download(self.symbol, start=recent_start, end=end_date, interval='5m')
                elif timeframe == '1m':
                    # 1-minute data (limited to last 7 days)
                    recent_start = end_date - timedelta(days=7)
                    df = yf.download(self.symbol, start=recent_start, end=end_date, interval='1m')
                
                if not df.empty:
                    df = self.clean_data(df)
                    data[timeframe] = df
                    logger.info(f"Fetched {len(df)} records for {timeframe} timeframe")
                else:
                    logger.warning(f"No data retrieved for {timeframe} timeframe")
                    
            except Exception as e:
                logger.error(f"Error fetching {timeframe} data: {e}")
                continue
        
        return data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare market data
        """
        # Remove duplicates and sort by index
        df = df.drop_duplicates().sort_index()
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Remove rows with all NaN values
        df = df.dropna()
        
        # Ensure proper column names
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        return df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical analysis features
        """
        logger.info("Creating technical features...")
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Volatility measures
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Returns'].rolling(window=period).std()
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI_14'] = calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume features
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        # Price position relative to recent highs/lows
        for period in [5, 10, 20]:
            df[f'High_{period}'] = df['High'].rolling(window=period).max()
            df[f'Low_{period}'] = df['Low'].rolling(window=period).min()
            df[f'Price_Position_{period}'] = (df['Close'] - df[f'Low_{period}']) / (df[f'High_{period}'] - df[f'Low_{period}'])
        
        # Candlestick pattern features
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        df['Is_Green'] = (df['Close'] > df['Open']).astype(int)
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap'] > 0.005).astype(int)  # 0.5% gap
        df['Gap_Down'] = (df['Gap'] < -0.005).astype(int)
        
        return df
    
    def create_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify candlestick patterns for ML features
        """
        logger.info("Creating candlestick pattern features...")
        
        # Doji pattern
        df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
        
        # Hammer and Hanging Man
        body = abs(df['Close'] - df['Open'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        
        df['Hammer'] = ((lower_shadow >= 2 * body) & (upper_shadow <= 0.1 * body) & 
                       (df['Close'] > df['Open'])).astype(int)
        
        df['Hanging_Man'] = ((lower_shadow >= 2 * body) & (upper_shadow <= 0.1 * body) & 
                            (df['Close'] < df['Open'])).astype(int)
        
        # Engulfing patterns
        df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) &
                                  (df['Open'] < df['Close'].shift(1)) & 
                                  (df['Close'] > df['Open'].shift(1))).astype(int)
        
        df['Bearish_Engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) &
                                  (df['Open'] > df['Close'].shift(1)) & 
                                  (df['Close'] < df['Open'].shift(1))).astype(int)
        
        # Morning Star and Evening Star (simplified)
        df['Morning_Star'] = ((df['Close'].shift(2) < df['Open'].shift(2)) &  # First red candle
                             (abs(df['Close'].shift(1) - df['Open'].shift(1)) < body.shift(1) * 0.3) &  # Small second candle
                             (df['Close'] > df['Open']) &  # Third green candle
                             (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)).astype(int)
        
        df['Evening_Star'] = ((df['Close'].shift(2) > df['Open'].shift(2)) &  # First green candle
                             (abs(df['Close'].shift(1) - df['Open'].shift(1)) < body.shift(1) * 0.3) &  # Small second candle
                             (df['Close'] < df['Open']) &  # Third red candle
                             (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)).astype(int)
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
        """
        Create target labels for machine learning
        """
        # Future price movement
        df['Future_Close'] = df['Close'].shift(-lookahead)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
        
        # Classification labels
        df['Target_Direction'] = 0  # Hold
        df.loc[df['Future_Return'] > 0.005, 'Target_Direction'] = 1  # Buy (>0.5% gain)
        df.loc[df['Future_Return'] < -0.005, 'Target_Direction'] = -1  # Sell (<-0.5% loss)
        
        # Multi-class labels for better granularity
        df['Target_Class'] = 'HOLD'
        df.loc[df['Future_Return'] > 0.01, 'Target_Class'] = 'STRONG_BUY'
        df.loc[(df['Future_Return'] > 0.005) & (df['Future_Return'] <= 0.01), 'Target_Class'] = 'BUY'
        df.loc[(df['Future_Return'] < -0.005) & (df['Future_Return'] >= -0.01), 'Target_Class'] = 'SELL'
        df.loc[df['Future_Return'] < -0.01, 'Target_Class'] = 'STRONG_SELL'
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for machine learning models
        """
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['Future_Close', 'Future_Return', 'Target_Direction', 'Target_Class', 
                       'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['Target_Direction']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['Target_Direction'].values
        
        logger.info(f"Prepared ML features: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X, y, feature_cols
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train ensemble of machine learning models
        """
        logger.info("Training ensemble models...")
        
        # Split data chronologically
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=self.model_configs['ensemble']['rf_trees'],
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'feature_importance': dict(zip(feature_names, rf_model.feature_importances_))
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=self.model_configs['ensemble']['gb_estimators'],
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test, gb_pred),
            'feature_importance': dict(zip(feature_names, gb_model.feature_importances_))
        }
        
        # Neural Network
        mlp_model = MLPClassifier(
            hidden_layer_sizes=tuple(self.model_configs['ensemble']['mlp_layers']),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        mlp_model.fit(X_train_scaled, y_train)
        mlp_pred = mlp_model.predict(X_test_scaled)
        
        models['neural_network'] = mlp_model
        results['neural_network'] = {
            'accuracy': accuracy_score(y_test, mlp_pred)
        }
        
        # Save models and scaler
        self.models.update(models)
        self.scalers['ensemble'] = scaler
        self.performance_metrics.update(results)
        
        # Save to disk
        for model_name, model in models.items():
            joblib.dump(model, self.models_dir / f"{model_name}.pkl")
        joblib.dump(scaler, self.models_dir / "ensemble_scaler.pkl")
        
        logger.info("Ensemble models trained successfully")
        return results
    
    def train_deep_learning_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train deep learning models (LSTM, CNN-LSTM)
        """
        logger.info("Training deep learning models...")
        
        # Prepare sequence data
        feature_cols = [col for col in df.columns if col not in 
                       ['Future_Close', 'Future_Return', 'Target_Direction', 'Target_Class',
                        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        df_clean = df[feature_cols + ['Target_Direction']].dropna()
        
        # Create sequences
        lookback = self.model_configs['lstm']['lookback']
        X_sequences, y_sequences = self.create_sequences(df_clean[feature_cols].values, 
                                                        df_clean['Target_Direction'].values, 
                                                        lookback)
        
        # Split data
        split_point = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_point], X_sequences[split_point:]
        y_train, y_test = y_sequences[:split_point], y_sequences[split_point:]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train + 1, num_classes=num_classes)  # +1 to make labels 0,1,2
        y_test_cat = to_categorical(y_test + 1, num_classes=num_classes)
        
        models = {}
        results = {}
        
        # LSTM Model
        lstm_model = self.build_lstm_model(X_train_scaled.shape[1:], num_classes)
        lstm_model.fit(
            X_train_scaled, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        lstm_pred = lstm_model.predict(X_test_scaled)
        lstm_pred_classes = np.argmax(lstm_pred, axis=1) - 1  # Convert back to -1,0,1
        
        models['lstm'] = lstm_model
        results['lstm'] = {
            'accuracy': accuracy_score(y_test, lstm_pred_classes)
        }
        
        # CNN-LSTM Model
        cnn_lstm_model = self.build_cnn_lstm_model(X_train_scaled.shape[1:], num_classes)
        cnn_lstm_model.fit(
            X_train_scaled, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        cnn_lstm_pred = cnn_lstm_model.predict(X_test_scaled)
        cnn_lstm_pred_classes = np.argmax(cnn_lstm_pred, axis=1) - 1
        
        models['cnn_lstm'] = cnn_lstm_model
        results['cnn_lstm'] = {
            'accuracy': accuracy_score(y_test, cnn_lstm_pred_classes)
        }
        
        # Save models
        self.models.update(models)
        self.scalers['deep_learning'] = scaler
        
        lstm_model.save(self.models_dir / "lstm_model.h5")
        cnn_lstm_model.save(self.models_dir / "cnn_lstm_model.h5")
        joblib.dump(scaler, self.models_dir / "deep_learning_scaler.pkl")
        
        logger.info("Deep learning models trained successfully")
        return results
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(targets[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple, num_classes: int) -> Model:
        """
        Build LSTM model architecture
        """
        model = Sequential([
            LSTM(self.model_configs['lstm']['layers'][0], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.model_configs['lstm']['dropout']),
            LSTM(self.model_configs['lstm']['layers'][1], 
                 return_sequences=False),
            Dropout(self.model_configs['lstm']['dropout']),
            Dense(50, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple, num_classes: int) -> Model:
        """
        Build CNN-LSTM hybrid model
        """
        model = Sequential([
            Conv1D(filters=self.model_configs['cnn_lstm']['cnn_filters'][0], 
                   kernel_size=self.model_configs['cnn_lstm']['cnn_kernel'], 
                   activation='relu', 
                   input_shape=input_shape),
            Conv1D(filters=self.model_configs['cnn_lstm']['cnn_filters'][1], 
                   kernel_size=self.model_configs['cnn_lstm']['cnn_kernel'], 
                   activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(self.model_configs['cnn_lstm']['lstm_units'], 
                 return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Complete training pipeline for all models
        """
        logger.info("Starting comprehensive model training...")
        
        # Fetch multi-timeframe data
        market_data = self.fetch_multi_timeframe_data()
        
        all_results = {}
        
        for timeframe, df in market_data.items():
            logger.info(f"Training models for {timeframe} timeframe...")
            
            try:
                # Create features
                df = self.create_technical_features(df)
                df = self.create_candlestick_patterns(df)
                df = self.create_target_labels(df)
                
                # Train ensemble models
                X, y, feature_names = self.prepare_ml_features(df)
                if len(X) > 100:  # Minimum data requirement
                    ensemble_results = self.train_ensemble_models(X, y, feature_names)
                    all_results[f"{timeframe}_ensemble"] = ensemble_results
                    
                    # Train deep learning models
                    if len(df) > 200:  # More data needed for deep learning
                        dl_results = self.train_deep_learning_models(df)
                        all_results[f"{timeframe}_deep_learning"] = dl_results
                else:
                    logger.warning(f"Insufficient data for {timeframe}: {len(X)} samples")
                    
            except Exception as e:
                logger.error(f"Error training models for {timeframe}: {e}")
                continue
        
        # Save results
        import json
        with open(self.models_dir / "training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info("Model training completed successfully")
        return all_results
    
    def get_ensemble_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get ensemble prediction from all trained models
        """
        if 'ensemble' not in self.scalers:
            logger.error("Ensemble models not trained")
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        # Scale features
        features_scaled = self.scalers['ensemble'].transform(features.reshape(1, -1))
        
        predictions = {}
        confidences = {}
        
        # Get predictions from ensemble models
        for model_name in ['random_forest', 'gradient_boosting', 'neural_network']:
            if model_name in self.models:
                pred = self.models[model_name].predict(features_scaled)[0]
                pred_proba = self.models[model_name].predict_proba(features_scaled)[0]
                
                predictions[model_name] = pred
                confidences[model_name] = max(pred_proba)
        
        # Aggregate predictions
        if predictions:
            pred_values = list(predictions.values())
            final_prediction = max(set(pred_values), key=pred_values.count)  # Majority vote
            avg_confidence = np.mean(list(confidences.values()))
            
            signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            
            return {
                'signal': signal_map.get(final_prediction, 'HOLD'),
                'confidence': avg_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }
        
        return {'signal': 'HOLD', 'confidence': 0.0}
    
    def save_models(self):
        """
        Save all trained models
        """
        logger.info("Saving all models...")
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'lookback_days': self.lookback_days,
            'timeframes': self.timeframes,
            'model_configs': self.model_configs,
            'performance_metrics': self.performance_metrics,
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(self.models_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("All models saved successfully")
    
    def load_models(self):
        """
        Load pre-trained models
        """
        logger.info("Loading pre-trained models...")
        
        try:
            # Load ensemble models
            ensemble_models = ['random_forest', 'gradient_boosting', 'neural_network']
            for model_name in ensemble_models:
                model_path = self.models_dir / f"{model_name}.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
            
            # Load scalers
            scaler_path = self.models_dir / "ensemble_scaler.pkl"
            if scaler_path.exists():
                self.scalers['ensemble'] = joblib.load(scaler_path)
            
            # Load deep learning models
            lstm_path = self.models_dir / "lstm_model.h5"
            if lstm_path.exists():
                self.models['lstm'] = tf.keras.models.load_model(lstm_path)
            
            cnn_lstm_path = self.models_dir / "cnn_lstm_model.h5"
            if cnn_lstm_path.exists():
                self.models['cnn_lstm'] = tf.keras.models.load_model(cnn_lstm_path)
            
            dl_scaler_path = self.models_dir / "deep_learning_scaler.pkl"
            if dl_scaler_path.exists():
                self.scalers['deep_learning'] = joblib.load(dl_scaler_path)
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

if __name__ == "__main__":
    # Example usage
    learner = AdvancedMarketLearner("^NSEI", lookback_days=30)
    results = learner.train_all_models()
    learner.save_models()
    
    print("Training Results:")
    for timeframe, result in results.items():
        print(f"{timeframe}: {result}")