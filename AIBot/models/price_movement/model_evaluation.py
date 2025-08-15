"""
Model Evaluation for Nifty Price Movement Prediction
===================================================

Comprehensive evaluation metrics and analysis for classification models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class NiftyMovementEvaluator:
    """
    Evaluate Nifty price movement prediction models
    """
    
    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        
    def evaluate_classification(self, X_test, y_test, class_names=None):
        """
        Comprehensive classification evaluation
        """
        print(f"Evaluating {self.model_name} performance...")
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC (for binary classification)
        roc_auc = None
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n{self.model_name} Performance Metrics:")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return self.evaluation_results
    
    def calculate_trading_metrics(self, returns_data=None):
        """
        Calculate trading-specific metrics
        """
        if 'y_true' not in self.evaluation_results:
            print("Run evaluate_classification first")
            return None
        
        y_true = self.evaluation_results['y_true']
        y_pred = self.evaluation_results['y_pred']
        
        # For binary classification (Up/Down)
        if len(np.unique(y_true)) == 2:
            # Directional accuracy
            directional_accuracy = accuracy_score(y_true, y_pred)
            
            # True positives (correctly predicted up moves)
            tp_mask = (y_true == 1) & (y_pred == 1)
            tn_mask = (y_true == 0) & (y_pred == 0)
            fp_mask = (y_true == 0) & (y_pred == 1)
            fn_mask = (y_true == 1) & (y_pred == 0)
            
            # Trading metrics
            true_positives = np.sum(tp_mask)
            true_negatives = np.sum(tn_mask)
            false_positives = np.sum(fp_mask)
            false_negatives = np.sum(fn_mask)
            
            # Hit rates
            up_hit_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            down_hit_rate = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            
            trading_metrics = {
                'directional_accuracy': directional_accuracy,
                'up_prediction_accuracy': up_hit_rate,
                'down_prediction_accuracy': down_hit_rate,
                'true_positives': true_positives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'total_up_predictions': true_positives + false_positives,
                'total_down_predictions': true_negatives + false_negatives
            }
            
            # If returns data is provided, calculate P&L metrics
            if returns_data is not None and len(returns_data) == len(y_pred):
                strategy_returns = []
                
                for i in range(len(y_pred)):
                    if y_pred[i] == 1:  # Predicted up
                        strategy_returns.append(returns_data[i])
                    else:  # Predicted down - stay in cash or short
                        strategy_returns.append(0)  # Assuming cash position
                
                strategy_returns = np.array(strategy_returns)
                
                # Calculate performance metrics
                total_return = np.sum(strategy_returns)
                winning_trades = np.sum(strategy_returns > 0)
                losing_trades = np.sum(strategy_returns < 0)
                
                if winning_trades > 0:
                    avg_win = np.mean(strategy_returns[strategy_returns > 0])
                else:
                    avg_win = 0
                
                if losing_trades > 0:
                    avg_loss = np.mean(strategy_returns[strategy_returns < 0])
                else:
                    avg_loss = 0
                
                win_rate = winning_trades / len(strategy_returns) if len(strategy_returns) > 0 else 0
                
                trading_metrics.update({
                    'total_strategy_return': total_return,
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'win_rate': win_rate,
                    'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf
                })
            
            return trading_metrics
        
        return None
    
    def plot_confusion_matrix(self, class_names=None, figsize=(8, 6)):
        """
        Plot confusion matrix
        """
        if 'confusion_matrix' not in self.evaluation_results:
            print("Run evaluate_classification first")
            return
        
        cm = self.evaluation_results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0])
        )
        
        plt.title(f'{self.model_name} - Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plot ROC curve for binary classification
        """
        if 'y_pred_proba' not in self.evaluation_results or self.evaluation_results['y_pred_proba'] is None:
            print("Prediction probabilities not available")
            return
        
        y_true = self.evaluation_results['y_true']
        y_pred_proba = self.evaluation_results['y_pred_proba']
        
        if len(np.unique(y_true)) != 2:
            print("ROC curve only available for binary classification")
            return
        
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba_pos = y_pred_proba[:, 1]
        else:
            proba_pos = y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_true, proba_pos)
        roc_auc = roc_auc_score(y_true, proba_pos)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'{self.model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, figsize=(8, 6)):
        """
        Plot precision-recall curve
        """
        if 'y_pred_proba' not in self.evaluation_results or self.evaluation_results['y_pred_proba'] is None:
            print("Prediction probabilities not available")
            return
        
        y_true = self.evaluation_results['y_true']
        y_pred_proba = self.evaluation_results['y_pred_proba']
        
        if len(np.unique(y_true)) != 2:
            print("PR curve only available for binary classification")
            return
        
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba_pos = y_pred_proba[:, 1]
        else:
            proba_pos = y_pred_proba
        
        precision, recall, _ = precision_recall_curve(y_true, proba_pos)
        avg_precision = average_precision_score(y_true, proba_pos)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'{self.model_name} (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_distribution(self, figsize=(12, 4)):
        """
        Plot distribution of predictions vs actual
        """
        if 'y_pred_proba' not in self.evaluation_results:
            print("Run evaluate_classification first")
            return
        
        y_true = self.evaluation_results['y_true']
        y_pred = self.evaluation_results['y_pred']
        y_pred_proba = self.evaluation_results['y_pred_proba']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Actual vs Predicted distribution
        unique_classes = np.unique(y_true)
        x_pos = np.arange(len(unique_classes))
        
        actual_counts = [np.sum(y_true == cls) for cls in unique_classes]
        pred_counts = [np.sum(y_pred == cls) for cls in unique_classes]
        
        width = 0.35
        axes[0].bar(x_pos - width/2, actual_counts, width, label='Actual', alpha=0.8)
        axes[0].bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Actual vs Predicted Class Distribution')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(unique_classes)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction confidence distribution
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                max_proba = np.max(y_pred_proba, axis=1)
            else:
                max_proba = np.abs(y_pred_proba - 0.5) + 0.5  # Convert to confidence
            
            axes[1].hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Prediction Confidence')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Prediction Confidence Distribution')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self):
        """
        Generate comprehensive evaluation report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_classification first."
        
        report = {
            'model_name': self.model_name,
            'basic_metrics': {
                'accuracy': self.evaluation_results['accuracy'],
                'precision': self.evaluation_results['precision'],
                'recall': self.evaluation_results['recall'],
                'f1_score': self.evaluation_results['f1_score']
            },
            'confusion_matrix': self.evaluation_results['confusion_matrix'].tolist(),
            'classification_report': self.evaluation_results['classification_report']
        }
        
        if self.evaluation_results['roc_auc']:
            report['roc_auc'] = self.evaluation_results['roc_auc']
        
        # Add trading metrics if available
        trading_metrics = self.calculate_trading_metrics()
        if trading_metrics:
            report['trading_metrics'] = trading_metrics
        
        return report


class ModelComparison:
    """
    Compare multiple models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        """
        Add a model for comparison
        """
        self.models[name] = model
    
    def compare_models(self, X_test, y_test, class_names=None):
        """
        Compare all added models
        """
        comparison_results = {}
        
        for name, model in self.models.items():
            evaluator = NiftyMovementEvaluator(model, name)
            results = evaluator.evaluate_classification(X_test, y_test, class_names)
            comparison_results[name] = results
        
        # Create comparison dataframe
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        comparison_df = pd.DataFrame({
            name: [results[metric] for metric in metrics]
            for name, results in comparison_results.items()
        }, index=metrics)
        
        print("\nModel Comparison:")
        print("="*50)
        print(comparison_df.round(4))
        
        self.results = comparison_results
        return comparison_df, comparison_results
    
    def plot_model_comparison(self, figsize=(10, 6)):
        """
        Plot model comparison
        """
        if not self.results:
            print("Run compare_models first")
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][metric] for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()