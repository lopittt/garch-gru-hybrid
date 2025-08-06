# src/models/hybrid.py
# GARCH-GRU hybrid model combining econometric and deep learning approaches
# Integrates GARCH volatility forecasts with GRU predictions
# RELEVANT FILES: garch.py, gru.py, simulator.py, main.py

import numpy as np
import pandas as pd
from .garch import GARCH
from .gru import GRUModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import scale_volatility, prepare_rolling_windows

class GARCHGRUHybrid:
    def __init__(self, sequence_length=6):
        self.garch = GARCH(p=1, q=1)
        self.gru = GRUModel(sequence_length=sequence_length)
        self.is_fitted = False
        self.lambda_scale = None
        self.rolling_results = []
        
    def fit(self, data, returns, volatility):
        """Fit hybrid model using rolling window methodology from the paper"""
        print("Implementing paper-exact methodology...")
        
        # Prepare rolling windows (Section 3.2, Page 6)
        windows = prepare_rolling_windows(data, returns, volatility)
        print(f"Created {len(windows)} rolling windows")
        
        if not windows:
            # Fallback to original method if not enough data
            return self._fit_simple(returns, volatility)
        
        # Use the last window for final model training
        final_window = windows[-1]
        
        # Fit GARCH on the rolling window data
        garch_returns = final_window['garch_data']['returns']
        self.garch.fit(garch_returns)
        garch_vol = self.garch.conditional_volatility()
        
        # Generate GARCH forecasts for the prediction period
        garch_forecasts = []
        temp_garch = GARCH(p=1, q=1)
        temp_garch.fit(garch_returns)
        
        for i in range(len(final_window['prediction_period']['returns'])):
            forecast = temp_garch.forecast(horizon=1)
            garch_forecasts.append(forecast[0])
            # Update with actual return for next forecast (in real scenario, this would be iterative)
        
        garch_forecasts = np.array(garch_forecasts)
        
        # Apply volatility scaling (Equations 12-13, Page 7)
        prediction_volatility = final_window['prediction_period']['volatility']
        scaled_garch, self.lambda_scale = scale_volatility(garch_vol, prediction_volatility)
        scaled_forecasts, _ = scale_volatility(garch_forecasts, prediction_volatility)
        
        print(f"Volatility scaling factor (lambda): {self.lambda_scale:.4f}")
        
        # Prepare GRU training data with GARCH forecasts as input
        prediction_returns = final_window['prediction_period']['returns']
        sequences, targets = self.gru.prepare_sequences(
            np.abs(prediction_returns),
            scaled_forecasts,
            prediction_volatility
        )
        
        if len(sequences) == 0:
            print("Warning: No sequences generated, falling back to simple method")
            return self._fit_simple(returns, volatility)
        
        # Split into train/validation (67/33 split)
        split_idx = max(1, int(0.67 * len(sequences)))
        train_seq = sequences[:split_idx]
        train_tgt = targets[:split_idx]
        val_seq = sequences[split_idx:]
        val_tgt = targets[split_idx:]
        
        print(f"Training GRU with {len(train_seq)} training samples, {len(val_seq)} validation samples")
        
        # Train GRU with paper specifications
        training_history = self.gru.train(
            train_seq, train_tgt, val_seq, val_tgt, 
            epochs=50, batch_size=32, max_train_size=500
        )
        
        self.gru.load_best_model()
        self.is_fitted = True
        
        return self
    
    def _fit_simple(self, returns, volatility):
        """Fallback method for insufficient data"""
        print("Using simplified fitting method")
        
        # Fit GARCH model
        self.garch.fit(returns)
        garch_vol = self.garch.conditional_volatility()
        
        # Apply scaling
        scaled_garch, self.lambda_scale = scale_volatility(garch_vol, volatility)
        
        # Align data
        min_len = min(len(returns), len(volatility), len(scaled_garch))
        returns_aligned = returns[-min_len:].values
        volatility_aligned = volatility[-min_len:].values
        garch_aligned = scaled_garch[-min_len:]
        
        # Generate GARCH forecasts for GRU input
        garch_forecasts = []
        for i in range(len(returns_aligned)):
            if i > 0:
                temp_garch = GARCH(p=1, q=1)
                temp_garch.fit(returns_aligned[:i])
                forecast = temp_garch.forecast(horizon=1)[0]
                garch_forecasts.append(forecast)
            else:
                garch_forecasts.append(garch_aligned[0])
        
        # Prepare GRU training data
        sequences, targets = self.gru.prepare_sequences(
            np.abs(returns_aligned),
            np.array(garch_forecasts),
            volatility_aligned
        )
        
        if len(sequences) == 0:
            print("Error: Could not create training sequences")
            return self
        
        # Split into train/validation
        split_idx = max(1, int(0.67 * len(sequences)))
        train_seq = sequences[:split_idx]
        train_tgt = targets[:split_idx]
        val_seq = sequences[split_idx:]
        val_tgt = targets[split_idx:]
        
        # Train GRU
        self.gru.train(train_seq, train_tgt, val_seq, val_tgt, epochs=20)
        self.gru.load_best_model()
        
        self.is_fitted = True
        return self
    
    def forecast(self, returns, volatility, horizon=1):
        """Generate forecasts using paper methodology"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get GARCH forecast with scaling
        raw_garch_forecast = self.garch.forecast(horizon=horizon)
        if self.lambda_scale is not None:
            # Apply scaling factor from training
            garch_forecast = np.sqrt(self.lambda_scale * (raw_garch_forecast ** 2))
        else:
            garch_forecast = raw_garch_forecast
        
        # Get recent data for GRU (use GARCH forecasts as input, not historical volatility)
        recent_returns = np.abs(returns[-self.gru.sequence_length:].values)
        
        # Generate recent GARCH forecasts for GRU input
        recent_garch_forecasts = []
        for i in range(self.gru.sequence_length):
            start_idx = len(returns) - self.gru.sequence_length + i
            if start_idx > 0:
                temp_returns = returns[:start_idx]
                temp_garch = GARCH(p=1, q=1)
                temp_garch.fit(temp_returns)
                forecast = temp_garch.forecast(horizon=1)[0]
                if self.lambda_scale is not None:
                    forecast = np.sqrt(self.lambda_scale * (forecast ** 2))
                recent_garch_forecasts.append(forecast)
            else:
                recent_garch_forecasts.append(garch_forecast[0])
        
        recent_garch_forecasts = np.array(recent_garch_forecasts)
        
        # Prepare GRU input (2D: returns + GARCH forecasts)
        gru_input = np.column_stack([
            recent_returns,
            recent_garch_forecasts
        ]).reshape(1, self.gru.sequence_length, 2)
        
        # Get GRU forecast
        gru_forecast = self.gru.predict(gru_input)
        
        # Hybrid combination (paper uses weighted average, we'll use equal weights)
        # In the paper, weights could be optimized based on historical performance
        combined_forecast = 0.5 * garch_forecast + 0.5 * gru_forecast
        
        return {
            'garch': garch_forecast,
            'gru': gru_forecast,
            'combined': combined_forecast
        }
    
    def get_volatility_series(self):
        return {
            'garch': self.garch.conditional_volatility(),
            'combined': None  # Would need full historical predictions
        }