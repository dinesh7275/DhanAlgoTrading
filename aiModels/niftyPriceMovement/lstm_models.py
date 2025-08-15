"""
LSTM Models for Nifty Price Movement Prediction
===============================================

LSTM architectures optimized for Nifty price movement classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class NiftyLSTMClassifier:
    """
    LSTM model for Nifty price movement classification
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_lstm_model(self):
        """
        Build LSTM architecture for Nifty movement prediction
        """
        print("Building Nifty LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # First LSTM layer
            layers.LSTM(
                128, 
                return_sequences=True, 
                dropout=0.2, 
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(
                100, 
                return_sequences=True, 
                dropout=0.2, 
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(
                75, 
                dropout=0.2, 
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax' if self.n_classes > 2 else 'sigmoid')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        loss = 'sparse_categorical_crossentropy' if self.n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        print(model.summary())
        return model
    
    def create_sequences(self, X, y):
        """
        Create sequences for LSTM input
        """
        print(f"Creating LSTM sequences of length {self.sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Created {len(X_sequences)} sequences")
        print(f"Sequence shape: {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def train_model(self, X, y, validation_split=0.2, epochs=150, batch_size=32):
        """
        Train the LSTM model
        """
        print("Training Nifty LSTM model...")
        
        # Set number of features if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_lstm_model()
        
        # Split data (time-series aware)
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"Training set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=12,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_nifty_movement_lstm.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger('nifty_lstm_training_log.csv')
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_seq, _ = self.create_sequences(X, pd.Series(range(len(X))))
        predictions = self.model.predict(X_seq, verbose=0)
        
        if self.n_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_seq, _ = self.create_sequences(X, pd.Series(range(len(X))))
        return self.model.predict(X_seq, verbose=0)


class NiftyBidirectionalLSTM:
    """
    Bidirectional LSTM for Nifty price movement prediction
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
    
    def build_bidirectional_lstm(self):
        """
        Build bidirectional LSTM architecture
        """
        print("Building Nifty Bidirectional LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # First Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    100, 
                    return_sequences=True, 
                    dropout=0.2, 
                    recurrent_dropout=0.1
                )
            ),
            layers.BatchNormalization(),
            
            # Second Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    75, 
                    return_sequences=True, 
                    dropout=0.2, 
                    recurrent_dropout=0.1
                )
            ),
            layers.BatchNormalization(),
            
            # Third Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    50, 
                    dropout=0.2, 
                    recurrent_dropout=0.1
                )
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax' if self.n_classes > 2 else 'sigmoid')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if self.n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        print(model.summary())
        return model


class NiftyAttentionLSTM:
    """
    LSTM with attention mechanism for Nifty price movement prediction
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
    
    def attention_layer(self, inputs):
        """
        Custom attention mechanism
        """
        # Calculate attention weights
        attention = layers.Dense(1, activation='tanh')(inputs)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(inputs.shape[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention weights
        sent_representation = layers.Multiply()([inputs, attention])
        sent_representation = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(sent_representation)
        
        return sent_representation
    
    def build_attention_lstm(self):
        """
        Build LSTM with attention mechanism
        """
        print("Building Nifty Attention LSTM model...")
        
        # Input
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        lstm_out = layers.LSTM(100, return_sequences=True, dropout=0.2)(inputs)
        lstm_out = layers.BatchNormalization()(lstm_out)
        
        lstm_out = layers.LSTM(75, return_sequences=True, dropout=0.2)(lstm_out)
        lstm_out = layers.BatchNormalization()(lstm_out)
        
        # Apply attention
        attended_out = self.attention_layer(lstm_out)
        
        # Dense layers
        dense_out = layers.Dense(128, activation='relu')(attended_out)
        dense_out = layers.BatchNormalization()(dense_out)
        dense_out = layers.Dropout(0.5)(dense_out)
        
        dense_out = layers.Dense(64, activation='relu')(dense_out)
        dense_out = layers.Dropout(0.3)(dense_out)
        
        # Output layer
        outputs = layers.Dense(self.n_classes, activation='softmax' if self.n_classes > 2 else 'sigmoid')(dense_out)
        
        # Create model
        model = keras.Model(inputs, outputs)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if self.n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        print(model.summary())
        return model


class NiftyGRUClassifier:
    """
    GRU model for Nifty price movement prediction (alternative to LSTM)
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
    
    def build_gru_model(self):
        """
        Build GRU architecture
        """
        print("Building Nifty GRU model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # First GRU layer
            layers.GRU(
                128, 
                return_sequences=True, 
                dropout=0.2, 
                recurrent_dropout=0.1
            ),
            layers.BatchNormalization(),
            
            # Second GRU layer
            layers.GRU(
                100, 
                return_sequences=True, 
                dropout=0.2, 
                recurrent_dropout=0.1
            ),
            layers.BatchNormalization(),
            
            # Third GRU layer
            layers.GRU(
                75, 
                dropout=0.2, 
                recurrent_dropout=0.1
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax' if self.n_classes > 2 else 'sigmoid')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if self.n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        print(model.summary())
        return model
    
    def create_sequences(self, X, y):
        """
        Create sequences for GRU input
        """
        print(f"Creating GRU sequences of length {self.sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        return X_sequences, y_sequences