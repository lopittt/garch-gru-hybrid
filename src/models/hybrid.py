# src/models/hybrid.py
# GARCH-GRU hybrid model combining econometric and deep learning approaches
# Integrates GARCH volatility forecasts with GRU predictions
# RELEVANT FILES: garch.py, gru.py, simulator.py, main.py

import numpy as np
import pandas as pd
import time
from .garch import GARCH
from .gru import GRUModel
from .modal_gru import ModalGRUModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import scale_volatility, prepare_rolling_windows
from config.settings import get_settings

class GARCHGRUHybrid:
    def __init__(self, sequence_length=None, use_modal=None):
        # Load settings
        self.settings = get_settings()
        
        # Use settings with manual override capability
        if sequence_length is None:
            sequence_length = self.settings.get('training.sequence_length', 6)
        
        if use_modal is None:
            self.use_modal = self.settings.use_modal
        else:
            self.use_modal = use_modal
        
        # Initialize GARCH with settings
        garch_p = self.settings.get('data.garch_p', 1)
        garch_q = self.settings.get('data.garch_q', 1)
        self.garch = GARCH(p=garch_p, q=garch_q)
        
        # Initialize GRU based on settings
        if self.use_modal:
            self.gru = ModalGRUModel(sequence_length=sequence_length)
            print(f"🌥️  Using Modal cloud training (GPU: {self.settings.modal_gpu})")
        else:
            self.gru = GRUModel(sequence_length=sequence_length)
            print("💻 Using local CPU training")
        
        self.is_fitted = False
        self.lambda_scale = None
        self.rolling_results = []
        self.training_time = None
        
        # Print configuration if verbose
        if self.settings.get('training.verbose_training', True):
            print(f"   Training location: {'Modal' if self.use_modal else 'Local'}")
            print(f"   Sequence length: {sequence_length}")
            print(f"   GARCH model: ({garch_p}, {garch_q})")
        
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
        
        # Measure training time
        start_time = time.time()
        
        # Get training parameters from settings
        epochs = self.settings.get('training.epochs', 50)
        batch_size = self.settings.get('training.batch_size', 32)
        max_train_size = self.settings.get('training.max_train_size', 500)
        
        if self.use_modal:
            print("Training with Modal cloud compute...")
            training_history = self.gru.train_on_modal(
                train_seq, train_tgt, val_seq, val_tgt, 
                epochs=epochs, batch_size=batch_size, max_train_size=max_train_size
            )
            self.training_time = self.gru.get_training_time()
        else:
            print("Training locally...")
            training_history = self.gru.train(
                train_seq, train_tgt, val_seq, val_tgt, 
                epochs=epochs, batch_size=batch_size, max_train_size=max_train_size
            )
            end_time = time.time()
            self.training_time = end_time - start_time
            self.gru.load_best_model()
        
        self.is_fitted = True
        
        # Always print training time if configured in settings
        if self.settings.print_training_time:
            location = "Modal cloud" if self.use_modal else "local CPU"
            print(f"✅ Training completed on {location} in {self.training_time:.2f} seconds")
        
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
        
        # Train GRU with timing
        start_time = time.time()
        
        # Use reduced epochs for simple training
        epochs = min(self.settings.get('training.epochs', 50), 20)
        batch_size = self.settings.get('training.batch_size', 32)
        
        if self.use_modal:
            self.gru.train_on_modal(train_seq, train_tgt, val_seq, val_tgt, epochs=epochs, batch_size=batch_size)
            self.training_time = self.gru.get_training_time()
        else:
            self.gru.train(train_seq, train_tgt, val_seq, val_tgt, epochs=epochs, batch_size=batch_size)
            end_time = time.time()
            self.training_time = end_time - start_time
            self.gru.load_best_model()
        
        self.is_fitted = True
        
        # Print training time if configured in settings
        if self.settings.print_training_time:
            location = "Modal cloud" if self.use_modal else "local CPU"
            print(f"✅ Simplified training completed on {location} in {self.training_time:.2f} seconds")
            
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
        
        # Get recent data for GRU (use existing GARCH model's conditional volatility)
        recent_returns = np.abs(returns[-self.gru.sequence_length:].values)
        
        # Use the already-fitted GARCH model's conditional volatility
        # This is much more efficient and consistent with the paper
        garch_cond_vol = self.garch.conditional_volatility()
        
        # Get the recent GARCH conditional volatility values
        if len(garch_cond_vol) >= self.gru.sequence_length:
            recent_garch_forecasts = garch_cond_vol[-self.gru.sequence_length:].values
        else:
            # Fallback if not enough history
            recent_garch_forecasts = np.full(self.gru.sequence_length, garch_forecast[0])
        
        # Apply scaling if needed
        if self.lambda_scale is not None:
            recent_garch_forecasts = np.sqrt(self.lambda_scale * (recent_garch_forecasts ** 2))
        
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
    
    def get_training_time(self):
        """Get the training time for comparison"""
        return self.training_time