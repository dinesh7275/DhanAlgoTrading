"""
LSTM Model for Nifty Volatility Prediction
==========================================

Optimized LSTM architecture for Indian market volatility prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


class NiftyVolatilityLSTM:
    """
    LSTM model optimized for Nifty 50 volatility prediction
    """

    def __init__(self, input_shape, lstm_units=[100, 75, 50], dense_units=[32, 16], dropout_rate=0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None

    def build_nifty_model(self):
        """Build LSTM architecture optimized for Nifty 50"""
        print("Building Nifty 50 LSTM model...")

        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # First LSTM layer with higher capacity for complex patterns
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001),
                name='lstm_1'
            ),
            layers.BatchNormalization(),

            # Second LSTM layer
            layers.LSTM(
                self.lstm_units[1],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001),
                name='lstm_2'
            ),
            layers.BatchNormalization(),

            # Third LSTM layer for capturing longer patterns
            layers.LSTM(
                self.lstm_units[2],
                dropout=self.dropout_rate,
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(0.001),
                name='lstm_3'
            ),

            # Dense layers with residual connections concept
            layers.Dense(self.dense_units[0], activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            layers.Dense(self.dense_units[1], activation='relu', name='dense_2'),
            layers.Dropout(self.dropout_rate * 0.5),

            # Output layer
            layers.Dense(1, activation='linear', name='output')
        ])

        # Compile with optimized settings for volatility prediction
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )

        self.model = model

        # Print model summary
        print(model.summary())

        return model

    def get_nifty_callbacks(self, model_save_path='best_nifty_volatility_model.h5'):
        """Get callbacks optimized for Nifty training"""
        return [
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
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Custom callback for volatility-specific metrics
            keras.callbacks.CSVLogger('nifty_training_log.csv')
        ]

    def train_nifty_model(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """Train the model with Nifty-specific parameters"""
        print("Starting Nifty 50 LSTM training...")

        callbacks = self.get_nifty_callbacks()

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )

        self.history = history
        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def predict_with_uncertainty(self, X, n_samples=10):
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout
        """
        # Enable dropout during inference for uncertainty estimation
        predictions = []
        
        for _ in range(n_samples):
            # Make prediction with dropout enabled
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


class AttentionLSTM:
    """
    LSTM with attention mechanism for better long-term dependencies
    """
    
    def __init__(self, input_shape, lstm_units=[100, 75], dense_units=[32, 16], dropout_rate=0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None

    def attention_layer(self, inputs):
        """Custom attention layer"""
        # inputs shape: (batch_size, timesteps, features)
        attention_weights = layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
        attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
        
        # Apply attention weights
        attended_inputs = layers.Multiply()([inputs, attention_weights])
        
        # Sum over timesteps
        output = layers.GlobalAveragePooling1D()(attended_inputs)
        
        return output

    def build_attention_model(self):
        """Build LSTM model with attention mechanism"""
        print("Building Attention-based LSTM model...")

        # Input
        input_layer = layers.Input(shape=self.input_shape)

        # LSTM layers
        lstm_out = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=0.1,
            name='lstm_1'
        )(input_layer)
        
        lstm_out = layers.BatchNormalization()(lstm_out)
        
        lstm_out = layers.LSTM(
            self.lstm_units[1],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=0.1,
            name='lstm_2'
        )(lstm_out)

        # Apply attention
        attended_out = self.attention_layer(lstm_out)

        # Dense layers
        dense_out = layers.Dense(self.dense_units[0], activation='relu')(attended_out)
        dense_out = layers.BatchNormalization()(dense_out)
        dense_out = layers.Dropout(self.dropout_rate)(dense_out)

        dense_out = layers.Dense(self.dense_units[1], activation='relu')(dense_out)
        dense_out = layers.Dropout(self.dropout_rate * 0.5)(dense_out)

        # Output
        output = layers.Dense(1, activation='linear')(dense_out)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )

        self.model = model
        print(model.summary())
        
        return model