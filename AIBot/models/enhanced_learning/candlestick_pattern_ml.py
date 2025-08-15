#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Candlestick Pattern Recognition with Machine Learning
=============================================================

ML-enhanced candlestick pattern recognition system that learns from historical
patterns and their outcomes to predict future price movements.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Image processing
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)

class CandlestickPatternML:
    """
    Advanced machine learning system for candlestick pattern recognition and prediction
    """
    
    def __init__(self, symbol: str = "^NSEI"):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.pattern_templates = {}
        self.feature_importance = {}
        
        # Create models directory
        self.models_dir = Path("models/enhanced_learning/pattern_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern definitions
        self.pattern_definitions = {
            'doji': {
                'body_ratio_max': 0.1,
                'shadow_ratio_min': 0.3,
                'reliability': 0.65
            },
            'hammer': {
                'lower_shadow_min': 2.0,
                'upper_shadow_max': 0.1,
                'body_ratio_min': 0.1,
                'reliability': 0.72
            },
            'shooting_star': {
                'upper_shadow_min': 2.0,
                'lower_shadow_max': 0.1,
                'body_ratio_min': 0.1,
                'reliability': 0.68
            },
            'engulfing_bullish': {
                'body_ratio_min': 0.6,
                'size_ratio_min': 1.2,
                'reliability': 0.75
            },
            'engulfing_bearish': {
                'body_ratio_min': 0.6,
                'size_ratio_min': 1.2,
                'reliability': 0.73
            },
            'morning_star': {
                'middle_body_max': 0.3,
                'gap_min': 0.1,
                'reliability': 0.78
            },
            'evening_star': {
                'middle_body_max': 0.3,
                'gap_min': 0.1,
                'reliability': 0.76
            },
            'harami_bullish': {
                'containment_ratio': 0.8,
                'reliability': 0.62
            },
            'harami_bearish': {
                'containment_ratio': 0.8,
                'reliability': 0.61
            }
        }
        
        # Image generation parameters
        self.image_config = {
            'width': 64,
            'height': 64,
            'dpi': 100,
            'lookback': 10  # Number of candles to include in image
        }
        
        logger.info(f"CandlestickPatternML initialized for {symbol}")
    
    def fetch_training_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical data for training
        """
        logger.info(f"Fetching {days} days of training data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # Fetch 1-hour data for better pattern recognition
            df = yf.download(self.symbol, start=start_date, end=end_date, interval='1h')
            
            if not df.empty:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                df = df.dropna()
                
                logger.info(f"Fetched {len(df)} hourly candles")
                return df
            else:
                logger.error("No data fetched")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return pd.DataFrame()
    
    def calculate_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive candlestick features
        """
        logger.info("Calculating candle features...")
        
        # Basic candle properties
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Total_Range'] = df['High'] - df['Low']
        
        # Ratios
        df['Body_Ratio'] = df['Body_Size'] / df['Total_Range']
        df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Total_Range']
        df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Total_Range']
        df['Body_Position'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Total_Range']
        
        # Candle direction
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Is_Bearish'] = (df['Close'] < df['Open']).astype(int)
        df['Is_Doji'] = (abs(df['Close'] - df['Open']) <= df['Total_Range'] * 0.1).astype(int)
        
        # Size comparisons
        df['Body_Size_Pct'] = df['Body_Size'] / df['Close'] * 100
        df['Range_Size_Pct'] = df['Total_Range'] / df['Close'] * 100
        
        # Previous candle comparisons
        df['Size_vs_Prev'] = df['Body_Size'] / df['Body_Size'].shift(1)
        df['Range_vs_Prev'] = df['Total_Range'] / df['Total_Range'].shift(1)
        
        # Multi-candle features
        for i in range(1, 4):
            df[f'Body_Size_{i}'] = df['Body_Size'].shift(i)
            df[f'Is_Bullish_{i}'] = df['Is_Bullish'].shift(i)
            df[f'Body_Ratio_{i}'] = df['Body_Ratio'].shift(i)
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)
        
        # Price context
        df['Close_vs_High'] = (df['High'] - df['Close']) / df['Close'] * 100
        df['Close_vs_Low'] = (df['Close'] - df['Low']) / df['Close'] * 100
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap'] > 0.002).astype(int)
        df['Gap_Down'] = (df['Gap'] < -0.002).astype(int)
        
        return df
    
    def detect_traditional_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect traditional candlestick patterns
        """
        logger.info("Detecting traditional patterns...")
        
        patterns = list(self.pattern_definitions.keys())
        for pattern in patterns:
            df[pattern] = 0
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Doji
            if current['Body_Ratio'] <= self.pattern_definitions['doji']['body_ratio_max']:
                df.iloc[i, df.columns.get_loc('doji')] = 1
            
            # Hammer
            if (current['Lower_Shadow_Ratio'] >= current['Body_Ratio'] * self.pattern_definitions['hammer']['lower_shadow_min'] and
                current['Upper_Shadow_Ratio'] <= current['Body_Ratio'] * self.pattern_definitions['hammer']['upper_shadow_max'] and
                current['Body_Ratio'] >= self.pattern_definitions['hammer']['body_ratio_min']):
                df.iloc[i, df.columns.get_loc('hammer')] = 1
            
            # Shooting Star
            if (current['Upper_Shadow_Ratio'] >= current['Body_Ratio'] * self.pattern_definitions['shooting_star']['upper_shadow_min'] and
                current['Lower_Shadow_Ratio'] <= current['Body_Ratio'] * self.pattern_definitions['shooting_star']['lower_shadow_max'] and
                current['Body_Ratio'] >= self.pattern_definitions['shooting_star']['body_ratio_min']):
                df.iloc[i, df.columns.get_loc('shooting_star')] = 1
            
            # Bullish Engulfing
            if (current['Is_Bullish'] == 1 and prev['Is_Bearish'] == 1 and
                current['Open'] < prev['Close'] and current['Close'] > prev['Open'] and
                current['Body_Size'] > prev['Body_Size'] * self.pattern_definitions['engulfing_bullish']['size_ratio_min']):
                df.iloc[i, df.columns.get_loc('engulfing_bullish')] = 1
            
            # Bearish Engulfing
            if (current['Is_Bearish'] == 1 and prev['Is_Bullish'] == 1 and
                current['Open'] > prev['Close'] and current['Close'] < prev['Open'] and
                current['Body_Size'] > prev['Body_Size'] * self.pattern_definitions['engulfing_bearish']['size_ratio_min']):
                df.iloc[i, df.columns.get_loc('engulfing_bearish')] = 1
            
            # Morning Star (3-candle pattern)
            if (prev2['Is_Bearish'] == 1 and  # First candle bearish
                prev['Body_Ratio'] <= self.pattern_definitions['morning_star']['middle_body_max'] and  # Middle candle small
                current['Is_Bullish'] == 1 and  # Third candle bullish
                current['Close'] > (prev2['Open'] + prev2['Close']) / 2):  # Third closes above first midpoint
                df.iloc[i, df.columns.get_loc('morning_star')] = 1
            
            # Evening Star (3-candle pattern)
            if (prev2['Is_Bullish'] == 1 and  # First candle bullish
                prev['Body_Ratio'] <= self.pattern_definitions['evening_star']['middle_body_max'] and  # Middle candle small
                current['Is_Bearish'] == 1 and  # Third candle bearish
                current['Close'] < (prev2['Open'] + prev2['Close']) / 2):  # Third closes below first midpoint
                df.iloc[i, df.columns.get_loc('evening_star')] = 1
            
            # Bullish Harami
            if (prev['Is_Bearish'] == 1 and current['Is_Bullish'] == 1 and
                current['Open'] > prev['Close'] and current['Close'] < prev['Open'] and
                current['Body_Size'] < prev['Body_Size'] * self.pattern_definitions['harami_bullish']['containment_ratio']):
                df.iloc[i, df.columns.get_loc('harami_bullish')] = 1
            
            # Bearish Harami
            if (prev['Is_Bullish'] == 1 and current['Is_Bearish'] == 1 and
                current['Open'] < prev['Close'] and current['Close'] > prev['Open'] and
                current['Body_Size'] < prev['Body_Size'] * self.pattern_definitions['harami_bearish']['containment_ratio']):
                df.iloc[i, df.columns.get_loc('harami_bearish')] = 1
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
        """
        Create target labels for pattern effectiveness
        """
        logger.info("Creating target labels...")
        
        # Calculate future returns
        df['Future_Return_1'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Future_Return_3'] = df['Close'].shift(-3) / df['Close'] - 1
        df['Future_Return_5'] = df['Close'].shift(-5) / df['Close'] - 1
        
        # Calculate future volatility
        df['Future_Volatility'] = df['Close'].pct_change().shift(-lookahead).rolling(lookahead).std()
        
        # Create classification labels
        threshold = 0.005  # 0.5% movement threshold
        
        df['Target_Direction'] = 0  # Hold
        df.loc[df['Future_Return_5'] > threshold, 'Target_Direction'] = 1  # Bull
        df.loc[df['Future_Return_5'] < -threshold, 'Target_Direction'] = -1  # Bear
        
        # Pattern effectiveness labels
        for pattern in self.pattern_definitions.keys():
            df[f'{pattern}_effective'] = 0
            
            # Bullish patterns
            if pattern in ['hammer', 'engulfing_bullish', 'morning_star', 'harami_bullish']:
                pattern_mask = df[pattern] == 1
                effective_mask = pattern_mask & (df['Future_Return_5'] > threshold)
                df.loc[effective_mask, f'{pattern}_effective'] = 1
            
            # Bearish patterns  
            elif pattern in ['shooting_star', 'engulfing_bearish', 'evening_star', 'harami_bearish']:
                pattern_mask = df[pattern] == 1
                effective_mask = pattern_mask & (df['Future_Return_5'] < -threshold)
                df.loc[effective_mask, f'{pattern}_effective'] = 1
            
            # Neutral patterns
            else:
                pattern_mask = df[pattern] == 1
                effective_mask = pattern_mask & (abs(df['Future_Return_5']) > threshold)
                df.loc[effective_mask, f'{pattern}_effective'] = 1
        
        return df
    
    def generate_candle_images(self, df: pd.DataFrame, save_images: bool = False) -> np.ndarray:
        """
        Generate candlestick chart images for CNN training
        """
        logger.info("Generating candlestick images...")
        
        images = []
        lookback = self.image_config['lookback']
        
        for i in range(lookback, len(df)):
            # Get window of candles
            window = df.iloc[i-lookback:i]
            
            # Create figure
            fig = Figure(figsize=(self.image_config['width']/100, self.image_config['height']/100), 
                        dpi=self.image_config['dpi'])
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            # Plot candlesticks
            for j, (idx, candle) in enumerate(window.iterrows()):
                color = 'green' if candle['Close'] > candle['Open'] else 'red'
                
                # Draw candle body
                body_height = abs(candle['Close'] - candle['Open'])
                body_bottom = min(candle['Open'], candle['Close'])
                
                rect = patches.Rectangle((j, body_bottom), 0.8, body_height, 
                                       linewidth=1, edgecolor=color, facecolor=color)
                ax.add_patch(rect)
                
                # Draw shadows
                ax.plot([j+0.4, j+0.4], [candle['Low'], body_bottom], color=color, linewidth=1)
                ax.plot([j+0.4, j+0.4], [body_bottom + body_height, candle['High']], color=color, linewidth=1)
            
            # Set limits and remove axes
            ax.set_xlim(-0.5, lookback-0.5)
            ax.set_ylim(window['Low'].min() * 0.999, window['High'].max() * 1.001)
            ax.axis('off')
            
            # Convert to image array
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.image_config['height'], self.image_config['width'], 3)
            
            # Convert to grayscale
            buf_gray = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
            
            # Normalize
            buf_normalized = buf_gray.astype(np.float32) / 255.0
            
            images.append(buf_normalized)
            
            # Save sample images
            if save_images and i < lookback + 10:
                Image.fromarray((buf_normalized * 255).astype(np.uint8)).save(
                    self.models_dir / f"sample_candle_{i}.png"
                )
            
            plt.close(fig)
        
        return np.array(images)
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for traditional ML models
        """
        # Select feature columns
        feature_cols = [
            'Body_Ratio', 'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio', 'Body_Position',
            'Body_Size_Pct', 'Range_Size_Pct', 'Size_vs_Prev', 'Range_vs_Prev',
            'Volume_Ratio', 'Close_vs_High', 'Close_vs_Low', 'Gap',
            'Is_Bullish', 'Is_Bearish', 'Is_Doji', 'Volume_Spike', 'Gap_Up', 'Gap_Down'
        ]
        
        # Add previous candle features
        for i in range(1, 4):
            feature_cols.extend([f'Body_Size_{i}', f'Is_Bullish_{i}', f'Body_Ratio_{i}'])
        
        # Add pattern features
        pattern_cols = list(self.pattern_definitions.keys())
        feature_cols.extend(pattern_cols)
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['Target_Direction']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['Target_Direction'].values
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        
        return X, y, feature_cols
    
    def train_traditional_ml_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train traditional machine learning models
        """
        logger.info("Training traditional ML models...")
        
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
        
        # Random Forest with hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced', None]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train_scaled, y_train)
        rf_model = rf_grid.best_estimator_
        
        rf_pred = rf_model.predict(X_test_scaled)
        rf_proba = rf_model.predict_proba(X_test_scaled)
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'best_params': rf_grid.best_params_,
            'feature_importance': dict(zip(feature_names, rf_model.feature_importances_)),
            'classification_report': classification_report(y_test, rf_pred, output_dict=True)
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test, gb_pred),
            'feature_importance': dict(zip(feature_names, gb_model.feature_importances_)),
            'classification_report': classification_report(y_test, gb_pred, output_dict=True)
        }
        
        # Neural Network
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp_model.fit(X_train_scaled, y_train)
        mlp_pred = mlp_model.predict(X_test_scaled)
        
        models['neural_network'] = mlp_model
        results['neural_network'] = {
            'accuracy': accuracy_score(y_test, mlp_pred),
            'classification_report': classification_report(y_test, mlp_pred, output_dict=True)
        }
        
        # Save models
        self.models.update(models)
        self.scalers['traditional'] = scaler
        self.feature_importance = results['random_forest']['feature_importance']
        
        # Save to disk
        for model_name, model in models.items():
            joblib.dump(model, self.models_dir / f"{model_name}.pkl")
        joblib.dump(scaler, self.models_dir / "traditional_scaler.pkl")
        
        logger.info("Traditional ML models trained successfully")
        return results
    
    def train_cnn_model(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train CNN model for image-based pattern recognition
        """
        logger.info("Training CNN model...")
        
        # Split data
        split_point = int(len(images) * 0.8)
        X_train, X_test = images[:split_point], images[split_point:]
        y_train, y_test = labels[:split_point], labels[split_point:]
        
        # Convert labels to categorical
        num_classes = len(np.unique(labels))
        y_train_cat = to_categorical(y_train + 1, num_classes=num_classes)  # +1 to make 0,1,2
        y_test_cat = to_categorical(y_test + 1, num_classes=num_classes)
        
        # Build CNN model
        model = Sequential([
            Input(shape=(self.image_config['height'], self.image_config['width'], 1)),
            
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train.reshape(-1, self.image_config['height'], self.image_config['width'], 1),
            y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(
            X_test.reshape(-1, self.image_config['height'], self.image_config['width'], 1),
            y_test_cat,
            verbose=0
        )
        
        # Save model
        model.save(self.models_dir / "cnn_pattern_model.h5")
        self.models['cnn'] = model
        
        logger.info(f"CNN model trained with accuracy: {test_acc:.4f}")
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'history': history.history
        }
    
    def train_hybrid_model(self, df: pd.DataFrame, images: np.ndarray) -> Dict[str, Any]:
        """
        Train hybrid model combining traditional features and CNN
        """
        logger.info("Training hybrid model...")
        
        # Prepare traditional features
        X_traditional, y, feature_names = self.prepare_ml_features(df)
        
        # Align data lengths
        min_length = min(len(X_traditional), len(images))
        X_traditional = X_traditional[-min_length:]
        images = images[-min_length:]
        y = y[-min_length:]
        
        # Split data
        split_point = int(min_length * 0.8)
        
        X_trad_train = X_traditional[:split_point]
        X_trad_test = X_traditional[split_point:]
        X_img_train = images[:split_point]
        X_img_test = images[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        # Scale traditional features
        scaler = StandardScaler()
        X_trad_train_scaled = scaler.fit_transform(X_trad_train)
        X_trad_test_scaled = scaler.transform(X_trad_test)
        
        # Convert labels
        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train + 1, num_classes=num_classes)
        y_test_cat = to_categorical(y_test + 1, num_classes=num_classes)
        
        # Build hybrid model
        # CNN branch
        cnn_input = Input(shape=(self.image_config['height'], self.image_config['width'], 1))
        x1 = Conv2D(32, (3, 3), activation='relu')(cnn_input)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = Flatten()(x1)
        x1 = Dense(64, activation='relu')(x1)
        
        # Traditional features branch
        trad_input = Input(shape=(X_trad_train_scaled.shape[1],))
        x2 = Dense(128, activation='relu')(trad_input)
        x2 = Dropout(0.3)(x2)
        x2 = Dense(64, activation='relu')(x2)
        
        # Combine branches
        combined = concatenate([x1, x2])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
        combined = Dense(64, activation='relu')(combined)
        output = Dense(num_classes, activation='softmax')(combined)
        
        model = Model(inputs=[cnn_input, trad_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train hybrid model
        history = model.fit(
            [X_img_train.reshape(-1, self.image_config['height'], self.image_config['width'], 1),
             X_trad_train_scaled],
            y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(
            [X_img_test.reshape(-1, self.image_config['height'], self.image_config['width'], 1),
             X_trad_test_scaled],
            y_test_cat,
            verbose=0
        )
        
        # Save model
        model.save(self.models_dir / "hybrid_pattern_model.h5")
        self.models['hybrid'] = model
        self.scalers['hybrid'] = scaler
        
        logger.info(f"Hybrid model trained with accuracy: {test_acc:.4f}")
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'history': history.history
        }
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Complete training pipeline for all pattern recognition models
        """
        logger.info("Starting comprehensive pattern recognition training...")
        
        # Fetch training data
        df = self.fetch_training_data(days=365)
        if df.empty:
            logger.error("No training data available")
            return {}
        
        # Calculate features and detect patterns
        df = self.calculate_candle_features(df)
        df = self.detect_traditional_patterns(df)
        df = self.create_target_labels(df)
        
        all_results = {}
        
        try:
            # Train traditional ML models
            X, y, feature_names = self.prepare_ml_features(df)
            if len(X) > 100:
                traditional_results = self.train_traditional_ml_models(X, y, feature_names)
                all_results['traditional'] = traditional_results
            
            # Generate images and train CNN
            if len(df) > 200:
                images = self.generate_candle_images(df, save_images=True)
                
                # Align labels with images
                aligned_labels = y[-(len(images)):]
                
                if len(images) > 100:
                    cnn_results = self.train_cnn_model(images, aligned_labels)
                    all_results['cnn'] = cnn_results
                    
                    # Train hybrid model
                    hybrid_results = self.train_hybrid_model(df, images)
                    all_results['hybrid'] = hybrid_results
        
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
        
        # Save results
        import json
        with open(self.models_dir / "pattern_training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'training_date': datetime.now().isoformat(),
            'data_points': len(df),
            'pattern_definitions': self.pattern_definitions,
            'results_summary': {
                model_type: {
                    'best_accuracy': max([model_results.get('accuracy', 0) 
                                        if isinstance(model_results, dict) 
                                        else max(model_results.values(), key=lambda x: x.get('accuracy', 0))['accuracy']
                                        for model_results in results.values()])
                } for model_type, results in all_results.items()
            }
        }
        
        with open(self.models_dir / "pattern_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Pattern recognition training completed successfully")
        return all_results
    
    def predict_pattern_outcome(self, current_data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Predict pattern outcome using ensemble of trained models
        """
        if not self.models:
            logger.warning("No models loaded for prediction")
            return {'prediction': 'HOLD', 'confidence': 0.0}
        
        predictions = {}
        confidences = {}
        
        try:
            # Prepare traditional features
            if isinstance(current_data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame([current_data])
            else:
                df = current_data.copy()
            
            # Calculate features
            df = self.calculate_candle_features(df)
            df = self.detect_traditional_patterns(df)
            
            # Get predictions from traditional models
            if 'random_forest' in self.models and 'traditional' in self.scalers:
                X, _, feature_names = self.prepare_ml_features(df)
                if len(X) > 0:
                    X_scaled = self.scalers['traditional'].transform(X[-1:])
                    
                    for model_name in ['random_forest', 'gradient_boosting', 'neural_network']:
                        if model_name in self.models:
                            pred = self.models[model_name].predict(X_scaled)[0]
                            pred_proba = self.models[model_name].predict_proba(X_scaled)[0]
                            
                            predictions[model_name] = pred
                            confidences[model_name] = max(pred_proba)
            
            # Ensemble prediction
            if predictions:
                pred_values = list(predictions.values())
                final_prediction = max(set(pred_values), key=pred_values.count)
                avg_confidence = np.mean(list(confidences.values()))
                
                signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                
                return {
                    'prediction': signal_map.get(final_prediction, 'HOLD'),
                    'confidence': avg_confidence,
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'detected_patterns': self.get_detected_patterns(df),
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error in pattern prediction: {e}")
        
        return {'prediction': 'HOLD', 'confidence': 0.0}
    
    def get_detected_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of detected patterns in current data
        """
        patterns = []
        latest = df.iloc[-1]
        
        for pattern in self.pattern_definitions.keys():
            if pattern in df.columns and latest[pattern] == 1:
                patterns.append(pattern)
        
        return patterns
    
    def load_models(self):
        """
        Load pre-trained models
        """
        logger.info("Loading pre-trained pattern recognition models...")
        
        try:
            # Load traditional models
            traditional_models = ['random_forest', 'gradient_boosting', 'neural_network']
            for model_name in traditional_models:
                model_path = self.models_dir / f"{model_name}.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
            
            # Load scalers
            scaler_path = self.models_dir / "traditional_scaler.pkl"
            if scaler_path.exists():
                self.scalers['traditional'] = joblib.load(scaler_path)
            
            # Load deep learning models
            cnn_path = self.models_dir / "cnn_pattern_model.h5"
            if cnn_path.exists():
                self.models['cnn'] = tf.keras.models.load_model(cnn_path)
            
            hybrid_path = self.models_dir / "hybrid_pattern_model.h5"
            if hybrid_path.exists():
                self.models['hybrid'] = tf.keras.models.load_model(hybrid_path)
            
            hybrid_scaler_path = self.models_dir / "hybrid_scaler.pkl"
            if hybrid_scaler_path.exists():
                self.scalers['hybrid'] = joblib.load(hybrid_scaler_path)
            
            logger.info(f"Loaded {len(self.models)} pattern recognition models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

if __name__ == "__main__":
    # Example usage
    pattern_ml = CandlestickPatternML("^NSEI")
    results = pattern_ml.train_all_models()
    
    print("Pattern Recognition Training Results:")
    for model_type, result in results.items():
        print(f"{model_type}: {result}")
    
    # Test prediction
    test_data = {
        'Open': 25000, 'High': 25100, 'Low': 24950, 'Close': 25050, 'Volume': 1000000
    }
    prediction = pattern_ml.predict_pattern_outcome(test_data)
    print(f"Pattern Prediction: {prediction}")