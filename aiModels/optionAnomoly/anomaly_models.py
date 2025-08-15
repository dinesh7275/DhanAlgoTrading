"""
Anomaly Detection Models for Options
===================================

Autoencoder and other ML models for detecting option pricing anomalies
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class OptionsAnomalyAutoencoder:
    """
    Autoencoder for detecting option pricing anomalies
    """
    
    def __init__(self, input_dim, encoding_dim=10, hidden_layers=[20, 15]):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def build_autoencoder(self):
        """Build autoencoder architecture"""
        print("Building Options Anomaly Autoencoder...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)
            encoded = layers.Dropout(0.1)(encoded)
        
        # Bottleneck layer
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, activation='relu')(decoded)
            decoded = layers.Dropout(0.1)(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        self.model = keras.Model(input_layer, decoded)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(self.model.summary())
        return self.model
    
    def train_autoencoder(self, X_train, X_val=None, epochs=100, batch_size=32):
        """Train the autoencoder"""
        print("Training Options Anomaly Autoencoder...")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.7,
                patience=8,
                min_lr=0.00001
            )
        ]
        
        # Train model
        validation_data = (X_val_scaled, X_val_scaled) if X_val_scaled is not None else None
        
        history = self.model.fit(
            X_train_scaled, X_train_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate reconstruction threshold
        train_reconstructions = self.model.predict(X_train_scaled)
        train_mse = np.mean(np.square(X_train_scaled - train_reconstructions), axis=1)
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile as threshold
        
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        return history
    
    def detect_anomalies(self, X, return_scores=False):
        """Detect anomalies in new data"""
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Identify anomalies
        anomalies = mse > self.threshold
        
        if return_scores:
            return anomalies, mse
        return anomalies
    
    def get_feature_importance(self, X):
        """Get feature importance based on reconstruction errors"""
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate per-feature reconstruction errors
        feature_errors = np.mean(np.square(X_scaled - reconstructions), axis=0)
        
        # Normalize to get importance scores
        importance_scores = feature_errors / np.sum(feature_errors)
        
        return importance_scores


class OptionsIsolationForest:
    """
    Isolation Forest for option anomaly detection
    """
    
    def __init__(self, contamination=0.05, n_estimators=100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X):
        """Fit the Isolation Forest model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self
    
    def predict_anomalies(self, X):
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions == -1  # Convert to boolean (True for anomaly)
    
    def get_anomaly_scores(self, X):
        """Get anomaly scores (lower scores indicate anomalies)"""
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        return scores


class HybridAnomalyDetector:
    """
    Hybrid model combining autoencoder and isolation forest
    """
    
    def __init__(self, input_dim, autoencoder_weight=0.6):
        self.autoencoder = OptionsAnomalyAutoencoder(input_dim)
        self.isolation_forest = OptionsIsolationForest()
        self.autoencoder_weight = autoencoder_weight
        self.isolation_weight = 1 - autoencoder_weight
    
    def fit(self, X_train, X_val=None, epochs=100):
        """Fit both models"""
        print("Training Hybrid Anomaly Detector...")
        
        # Train autoencoder
        self.autoencoder.build_autoencoder()
        ae_history = self.autoencoder.train_autoencoder(X_train, X_val, epochs)
        
        # Train isolation forest
        self.isolation_forest.fit(X_train)
        
        return ae_history
    
    def predict_anomalies(self, X, return_scores=False):
        """Predict anomalies using hybrid approach"""
        # Get autoencoder anomalies and scores
        ae_anomalies, ae_scores = self.autoencoder.detect_anomalies(X, return_scores=True)
        
        # Get isolation forest anomalies and scores
        if_anomalies = self.isolation_forest.predict_anomalies(X)
        if_scores = self.isolation_forest.get_anomaly_scores(X)
        
        # Normalize scores to [0, 1] range
        ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
        if_scores_norm = (-if_scores - (-if_scores).min()) / ((-if_scores).max() - (-if_scores).min())
        
        # Combine scores
        combined_scores = (self.autoencoder_weight * ae_scores_norm + 
                         self.isolation_weight * if_scores_norm)
        
        # Set threshold for combined scores
        threshold = np.percentile(combined_scores, 95)
        combined_anomalies = combined_scores > threshold
        
        if return_scores:
            return combined_anomalies, {
                'combined_scores': combined_scores,
                'autoencoder_scores': ae_scores,
                'isolation_forest_scores': if_scores,
                'threshold': threshold
            }
        
        return combined_anomalies
    
    def get_model_agreement(self, X):
        """Check agreement between the two models"""
        ae_anomalies, _ = self.autoencoder.detect_anomalies(X, return_scores=True)
        if_anomalies = self.isolation_forest.predict_anomalies(X)
        
        agreement = ae_anomalies == if_anomalies
        agreement_rate = np.mean(agreement)
        
        return {
            'agreement_rate': agreement_rate,
            'autoencoder_anomalies': ae_anomalies,
            'isolation_forest_anomalies': if_anomalies,
            'agreement_mask': agreement
        }