# src/evaluation/metrics.py
# Comprehensive evaluation metrics for GARCH-GRU model assessment
# Implements metrics from Section 2.3 of the paper
# RELEVANT FILES: hybrid.py, simulator.py, main.py

import numpy as np
import pandas as pd
from scipy import stats

class ModelEvaluator:
    """Evaluation metrics for volatility forecasting models"""
    
    @staticmethod
    def calculate_mse(actual, predicted):
        """Mean Squared Error"""
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        return np.mean((actual - predicted) ** 2)
    
    @staticmethod
    def calculate_mae(actual, predicted):
        """Mean Absolute Error"""
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def calculate_hmse(actual, predicted):
        """Heteroscedasticity-adjusted Mean Squared Error
        
        HMSE = mean((actual - predicted)² / actual²)
        Penalizes errors more when actual volatility is low
        """
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.nan
            
        return np.mean(((actual[mask] - predicted[mask]) / actual[mask]) ** 2)
    
    @staticmethod
    def calculate_r_squared(actual, predicted):
        """R-squared (coefficient of determination)"""
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0 if ss_res == 0 else -np.inf
            
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_all_metrics(actual, predicted):
        """Calculate all metrics at once"""
        return {
            'MSE': ModelEvaluator.calculate_mse(actual, predicted),
            'MAE': ModelEvaluator.calculate_mae(actual, predicted),
            'HMSE': ModelEvaluator.calculate_hmse(actual, predicted),
            'R²': ModelEvaluator.calculate_r_squared(actual, predicted)
        }
    
    @staticmethod
    def compare_models(actual, predictions_dict):
        """Compare multiple models
        
        Args:
            actual: Actual volatility values
            predictions_dict: Dict of model_name -> predictions
        
        Returns:
            DataFrame with metrics for each model
        """
        results = {}
        for model_name, predictions in predictions_dict.items():
            results[model_name] = ModelEvaluator.calculate_all_metrics(actual, predictions)
        
        return pd.DataFrame(results).T
    
    @staticmethod
    def print_evaluation_report(actual, predicted, model_name="Model"):
        """Print formatted evaluation report"""
        metrics = ModelEvaluator.calculate_all_metrics(actual, predicted)
        
        print(f"\n{'='*50}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'='*50}")
        print(f"MSE:  {metrics['MSE']:.6f}")
        print(f"MAE:  {metrics['MAE']:.6f}")
        print(f"HMSE: {metrics['HMSE']:.6f}")
        print(f"R²:   {metrics['R²']:.4f}")
        print(f"{'='*50}\n")
        
        return metrics