"""
Model Evaluation and Performance Analysis
========================================

Comprehensive evaluation metrics for volatility prediction models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class NiftyModelEvaluator:
    """
    Evaluate model performance with Nifty 50 specific metrics
    """

    def __init__(self, model, target_scaler):
        self.model = model
        self.target_scaler = target_scaler

    def evaluate_nifty_model(self, X_test, y_test):
        """Comprehensive evaluation for Nifty volatility prediction"""
        print("Evaluating Nifty 50 model performance...")

        # Make predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Standard metrics
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)

        # Volatility-specific metrics
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

        # Directional accuracy (important for trading)
        y_actual_direction = np.diff(y_actual) > 0
        y_pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(y_actual_direction == y_pred_direction)

        # Volatility regime accuracy (low, medium, high vol)
        vol_percentiles = np.percentile(y_actual, [33, 67])
        actual_regime = np.digitize(y_actual, vol_percentiles)
        pred_regime = np.digitize(y_pred, vol_percentiles)
        regime_accuracy = np.mean(actual_regime == pred_regime)

        # Trading-relevant metrics
        # Forecast accuracy for different volatility levels
        high_vol_mask = y_actual > np.percentile(y_actual, 75)
        low_vol_mask = y_actual < np.percentile(y_actual, 25)

        high_vol_mae = mean_absolute_error(y_actual[high_vol_mask], y_pred[high_vol_mask])
        low_vol_mae = mean_absolute_error(y_actual[low_vol_mask], y_pred[low_vol_mask])

        # Persistence benchmark (naive forecast using last value)
        y_persistence = np.roll(y_actual, 1)[1:]
        y_actual_bench = y_actual[1:]
        persistence_mae = mean_absolute_error(y_actual_bench, y_persistence)
        
        # Skill score (improvement over persistence)
        skill_score = (persistence_mae - mae) / persistence_mae

        print(f"\nNifty 50 Volatility Model Performance:")
        print(f"{'='*50}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.1f}%)")
        print(f"Volatility Regime Accuracy: {regime_accuracy:.4f} ({regime_accuracy*100:.1f}%)")
        print(f"High Volatility MAE: {high_vol_mae:.6f}")
        print(f"Low Volatility MAE: {low_vol_mae:.6f}")
        print(f"Skill Score vs Persistence: {skill_score:.4f}")

        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'directional_accuracy': directional_accuracy,
            'regime_accuracy': regime_accuracy,
            'high_vol_mae': high_vol_mae, 'low_vol_mae': low_vol_mae,
            'skill_score': skill_score,
            'y_actual': y_actual, 'y_pred': y_pred
        }

    def plot_nifty_results(self, results, history):
        """Plot comprehensive results for Nifty model"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Training history
        axes[0, 0].plot(history.history['loss'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Nifty Model Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Predictions vs Actual
        axes[0, 1].plot(results['y_actual'], label='Actual Volatility', alpha=0.7, linewidth=1)
        axes[0, 1].plot(results['y_pred'], label='Predicted Volatility', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('Nifty 50 Volatility Predictions vs Actual')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Scatter plot
        axes[0, 2].scatter(results['y_actual'], results['y_pred'], alpha=0.6, s=20)
        axes[0, 2].plot([results['y_actual'].min(), results['y_actual'].max()],
                        [results['y_actual'].min(), results['y_actual'].max()], 'r--', linewidth=2)
        axes[0, 2].set_title('Predicted vs Actual Volatility')
        axes[0, 2].set_xlabel('Actual Volatility')
        axes[0, 2].set_ylabel('Predicted Volatility')
        axes[0, 2].grid(True)

        # Residuals
        residuals = results['y_actual'] - results['y_pred']
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Prediction Residuals Distribution')
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)

        # Volatility regimes
        vol_percentiles = np.percentile(results['y_actual'], [33, 67])
        actual_regime = np.digitize(results['y_actual'], vol_percentiles)
        pred_regime = np.digitize(results['y_pred'], vol_percentiles)

        regime_names = ['Low Vol', 'Medium Vol', 'High Vol']
        regime_accuracy = []

        for i in range(3):
            mask = actual_regime == i
            if np.sum(mask) > 0:
                accuracy = np.mean(pred_regime[mask] == i)
                regime_accuracy.append(accuracy)
            else:
                regime_accuracy.append(0)

        axes[1, 1].bar(regime_names, regime_accuracy, color=['green', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_title('Accuracy by Volatility Regime')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, axis='y')

        # Error analysis over time
        rolling_mae = pd.Series(np.abs(residuals)).rolling(window=50).mean()
        axes[1, 2].plot(rolling_mae, color='red', alpha=0.8)
        axes[1, 2].set_title('Rolling MAE Over Time (50-period window)')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Rolling MAE')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.show()

    def generate_performance_report(self, results):
        """Generate a detailed performance report"""
        report = {
            'model_performance': {
                'mse': results['mse'],
                'mae': results['mae'],
                'rmse': results['rmse'],
                'r2': results['r2'],
                'mape': results['mape']
            },
            'trading_metrics': {
                'directional_accuracy': results['directional_accuracy'],
                'regime_accuracy': results['regime_accuracy'],
                'skill_score': results['skill_score']
            },
            'volatility_analysis': {
                'high_vol_mae': results['high_vol_mae'],
                'low_vol_mae': results['low_vol_mae'],
                'avg_predicted_vol': np.mean(results['y_pred']),
                'avg_actual_vol': np.mean(results['y_actual'])
            }
        }
        
        return report


class ModelComparator:
    """
    Compare multiple models' performance
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model, target_scaler):
        """Add a model for comparison"""
        self.models[name] = {
            'model': model,
            'scaler': target_scaler
        }
    
    def compare_models(self, X_test, y_test):
        """Compare all added models"""
        comparison_results = {}
        
        for name, model_info in self.models.items():
            evaluator = NiftyModelEvaluator(model_info['model'], model_info['scaler'])
            results = evaluator.evaluate_nifty_model(X_test, y_test)
            comparison_results[name] = results
        
        # Create comparison dataframe
        metrics = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy', 'skill_score']
        comparison_df = pd.DataFrame({
            name: [results[metric] for metric in metrics]
            for name, results in comparison_results.items()
        }, index=metrics)
        
        print("\nModel Comparison:")
        print("="*50)
        print(comparison_df.round(4))
        
        return comparison_df, comparison_results