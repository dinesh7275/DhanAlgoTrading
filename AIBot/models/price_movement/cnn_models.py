"""
CNN Models for Nifty Price Movement Prediction
==============================================

Convolutional Neural Network architectures for price movement classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class NiftyCNNClassifier:
    """
    CNN model for Nifty price movement classification
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """
        Build 1D CNN architecture for time series classification
        """
        print("Building Nifty CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # First Conv1D block
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Second Conv1D block
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Third Conv1D block
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth Conv1D block
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
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
    
    def create_sequences(self, X, y):
        """
        Create sequences for CNN input
        """
        print(f"Creating sequences of length {self.sequence_length}...")
        
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
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the CNN model
        """
        print("Training Nifty CNN model...")
        
        # Set number of features if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_cnn_model()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, random_state=42, stratify=y_seq
        )
        
        print(f"Training set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_nifty_movement_cnn.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
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


class NiftyCNNLSTM:
    """
    Hybrid CNN-LSTM model for Nifty price movement prediction
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
    
    def build_cnn_lstm_model(self):
        """
        Build CNN-LSTM hybrid architecture
        """
        print("Building Nifty CNN-LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # CNN layers for feature extraction
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # LSTM layers for sequence modeling
            layers.LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            layers.BatchNormalization(),
            
            layers.LSTM(50, dropout=0.2, recurrent_dropout=0.1),
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
    
    def create_sequences(self, X, y):
        """
        Create sequences for CNN-LSTM input
        """
        print(f"Creating sequences of length {self.sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Created {len(X_sequences)} sequences")
        return X_sequences, y_sequences
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the CNN-LSTM model
        """
        print("Training Nifty CNN-LSTM model...")
        
        # Set number of features if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_cnn_lstm_model()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, random_state=42, stratify=y_seq
        )
        
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
                factor=0.6,
                patience=10,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_nifty_movement_cnn_lstm.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history


class NiftyResidualCNN:
    """
    Residual CNN architecture for Nifty price movement prediction
    """
    
    def __init__(self, sequence_length=20, n_features=None, n_classes=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
    def residual_block(self, x, filters, kernel_size=3):
        """
        Create a residual block
        """
        # Main path
        fx = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Dropout(0.1)(fx)
        fx = layers.Conv1D(filters, kernel_size, padding='same')(fx)
        fx = layers.BatchNormalization()(fx)
        
        # Skip connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        # Add skip connection
        out = layers.Add()([x, fx])
        out = layers.Activation('relu')(out)
        return out
    
    def build_residual_cnn(self):
        """
        Build residual CNN architecture
        """
        print("Building Nifty Residual CNN model...")
        
        # Input
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Initial conv layer
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual blocks
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)
        x = layers.MaxPooling1D(2)(x)
        
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = layers.MaxPooling1D(2)(x)
        
        x = self.residual_block(x, 256)
        x = self.residual_block(x, 256)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.n_classes, activation='softmax' if self.n_classes > 2 else 'sigmoid')(x)
        
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