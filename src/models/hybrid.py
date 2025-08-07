# src/models/hybrid.py
# GARCH-GRU hybrid model combining econometric and deep learning approaches
# Integrates GARCH volatility forecasts with GRU predictions
# RELEVANT FILES: garch.py, gru.py, simulator.py, main.py

import numpy as np
import pandas as pd
import time
from .garch import GARCH
from .normalized_gru import NormalizedGRUModel as GRUModel
try:
    from .modal_gru import ModalGRUModel
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    class ModalGRUModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("Modal not available. Use local training instead.")
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
        if self.use_modal and MODAL_AVAILABLE:
            self.gru = ModalGRUModel(sequence_length=sequence_length)
            print(f"🌥️  Using Modal cloud training (GPU: {self.settings.modal_gpu})")
        else:
            if self.use_modal and not MODAL_AVAILABLE:
                print("⚠️  Modal not available, falling back to local training")
            self.gru = GRUModel(sequence_length=sequence_length)
            print("💻 Using local CPU training")
            self.use_modal = False  # Update flag for consistency
        
        self.is_fitted = False
        self.lambda_scale = None
        self.rolling_results = []
        self.training_time = None
        
        # Add weight tracking for adaptive weighting
        self.garch_weight = 0.5  # Initial weight
        self.gru_weight = 0.5
        self.weight_optimization_history = []
        
        # Print configuration if verbose
        if self.settings.get('training.verbose_training', True):
            print(f"   Training location: {'Modal' if self.use_modal else 'Local'}")
            print(f"   Sequence length: {sequence_length}")
            print(f"   GARCH model: ({garch_p}, {garch_q})")
        
    def fit(self, data, returns, volatility):
        """Fit hybrid model using ALL available data for GRU training"""
        print("Training with all available data...")
        
        # Prepare rolling windows for methodology validation
        windows = prepare_rolling_windows(data, returns, volatility)
        print(f"Created {len(windows)} rolling windows for validation")
        
        if not windows:
            # Fallback to original method if not enough data
            return self._fit_simple(returns, volatility)
        
        # For GARCH: Use the last window approach as in paper
        final_window = windows[-1]
        
        # Fit GARCH on ALL available data for better estimates
        self.garch.fit(returns)
        garch_vol = self.garch.conditional_volatility()
        
        # Apply volatility scaling using all data
        scaled_garch_vol, self.lambda_scale = scale_volatility(garch_vol, volatility)
        
        print(f"Volatility scaling factor (lambda): {self.lambda_scale:.4f}")
        
        # Prepare GRU training data using ALL available data
        sequences, targets = self.gru.prepare_sequences(
            np.abs(returns.values),
            scaled_garch_vol,
            volatility
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
        epochs = self.settings.get('training.epochs', 100)
        # For Ultra-Simple GRU, use appropriate batch size
        batch_size = 32  # Fixed batch size that works well
        max_train_size = None  # Use all available data for Ultra-Simple GRU
        
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
        
        # Optimize combination weights on validation data
        if len(val_seq) > 0:
            print("Optimizing hybrid weights on validation data...")
            self.optimize_weights(val_seq, val_tgt)
        else:
            # Set default weights if no validation data
            self.garch_weight = 0.3
            self.gru_weight = 0.7
            print(f"Using default weights - GARCH: {self.garch_weight:.3f}, GRU: {self.gru_weight:.3f}")
        
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
        
        # Use historical conditional volatility instead of forecasts
        # This is the key fix for look-ahead bias
        garch_conditional_vol = self.garch.conditional_volatility()
        
        # Align with the available data
        if len(garch_conditional_vol) >= len(returns_aligned):
            aligned_conditional_vol = garch_conditional_vol[-len(returns_aligned):].values
        else:
            # Pad if necessary
            padding_size = len(returns_aligned) - len(garch_conditional_vol)
            aligned_conditional_vol = np.concatenate([
                np.full(padding_size, garch_conditional_vol.iloc[0] if len(garch_conditional_vol) > 0 else garch_aligned[0]),
                garch_conditional_vol.values
            ])
        
        # Apply scaling to conditional volatility
        scaled_conditional_vol = np.sqrt(self.lambda_scale * (aligned_conditional_vol ** 2)) if self.lambda_scale else aligned_conditional_vol
        
        # Prepare GRU training data with historical conditional volatility
        sequences, targets = self.gru.prepare_sequences(
            np.abs(returns_aligned),
            scaled_conditional_vol,  # Historical values, not forecasts
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
        epochs = min(self.settings.get('training.epochs', 150), 20)
        batch_size = self.settings.get('training.batch_size', 500)
        
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
    
    def optimize_weights(self, validation_data, validation_targets):
        """Optimize combination weights based on validation performance"""
        
        # Get GRU predictions for validation data
        gru_predictions = self.gru.predict(validation_data).flatten()
        
        # Get scaled GARCH predictions for the same period
        # We need to align GARCH with the validation targets
        garch_cond_vol = self.garch.conditional_volatility()
        val_size = len(gru_predictions)
        
        # Get the corresponding GARCH values
        # Since sequences use historical data, we need to get the right alignment
        if len(garch_cond_vol) >= val_size:
            # Get GARCH values that correspond to validation targets
            # Accounting for sequence_length offset
            seq_len = self.gru.sequence_length if hasattr(self.gru, 'sequence_length') else 6
            start_idx = len(garch_cond_vol) - len(validation_targets) - seq_len + 1
            end_idx = start_idx + len(validation_targets)
            
            if start_idx >= 0 and end_idx <= len(garch_cond_vol):
                garch_predictions = garch_cond_vol.values[start_idx:end_idx]
            else:
                # Fallback to last values
                garch_predictions = garch_cond_vol.values[-val_size:]
        else:
            # Use forecast if not enough historical data
            garch_predictions = np.full(val_size, self.garch.forecast(horizon=1)[0])
        
        # Apply scaling to GARCH to match target scale
        if self.lambda_scale is not None:
            scaled_garch_predictions = np.sqrt(self.lambda_scale * (garch_predictions ** 2))
        else:
            scaled_garch_predictions = garch_predictions
        
        # Ensure arrays have same length
        min_len = min(len(scaled_garch_predictions), len(gru_predictions), len(validation_targets))
        scaled_garch_predictions = scaled_garch_predictions[:min_len]
        gru_predictions = gru_predictions[:min_len]
        validation_targets = validation_targets[:min_len]
        
        # Grid search for optimal weights
        best_mse = float('inf')
        best_weight = 0.5
        best_r2 = -float('inf')
        
        # Test more granular weights
        for w_garch in np.linspace(0, 1, 21):
            w_gru = 1 - w_garch
            
            # Calculate combined predictions
            combined_predictions = w_garch * scaled_garch_predictions + w_gru * gru_predictions
            
            # Calculate MSE and R²
            mse = np.mean((combined_predictions - validation_targets) ** 2)
            ss_res = np.sum((combined_predictions - validation_targets) ** 2)
            ss_tot = np.sum((validation_targets - np.mean(validation_targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
            
            if r2 > best_r2:  # Optimize for R² instead of MSE
                best_r2 = r2
                best_mse = mse
                best_weight = w_garch
        
        # Set the optimal weights
        self.garch_weight = best_weight
        self.gru_weight = 1 - best_weight
        
        # Calculate individual R² for comparison
        garch_r2 = 1 - (np.sum((scaled_garch_predictions - validation_targets) ** 2) / 
                        np.sum((validation_targets - np.mean(validation_targets)) ** 2))
        gru_r2 = 1 - (np.sum((gru_predictions - validation_targets) ** 2) / 
                     np.sum((validation_targets - np.mean(validation_targets)) ** 2))
        
        print(f"Weight optimization results:")
        print(f"  GARCH R²: {garch_r2:.4f}")
        print(f"  GRU R²: {gru_r2:.4f}")
        print(f"  Best combined R²: {best_r2:.4f}")
        print(f"Optimized weights - GARCH: {self.garch_weight:.3f}, GRU: {self.gru_weight:.3f}")
        
        return self
    
    def forecast(self, returns, volatility, horizon=1):
        """Generate forecasts using paper methodology"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get GARCH forecast for next period
        garch_forecast = self.garch.forecast(horizon=horizon)
        
        # Get recent data for GRU
        recent_returns = np.abs(returns[-self.gru.sequence_length:].values)
        
        # For GRU input, we need GARCH conditional volatility values
        # If we have enough history from training, use those
        # Otherwise, use the rolling forecast to generate them
        garch_cond_vol = self.garch.conditional_volatility()
        
        if len(returns) <= len(self.garch.returns):
            # We're still within training data range
            recent_garch_values = garch_cond_vol[-self.gru.sequence_length:].values
        else:
            # We're in test data, need to generate GARCH values
            # Get the new returns since training
            new_returns = returns[len(self.garch.returns):]
            # Generate rolling forecasts for these new returns
            new_garch_values = self.garch.rolling_forecast(new_returns)
            # Combine with historical values
            all_garch_values = np.concatenate([garch_cond_vol.values, new_garch_values])
            # Get the most recent sequence
            recent_garch_values = all_garch_values[-self.gru.sequence_length:]
        
        # Apply the same scaling that was used during training
        # During training, we scale GARCH to match target volatility
        # For forecasting, we just apply the learned lambda scale
        if self.lambda_scale is not None:
            recent_garch_scaled = np.sqrt(self.lambda_scale * (recent_garch_values ** 2))
        else:
            recent_garch_scaled = recent_garch_values
        
        # Prepare GRU input (2D: returns + scaled GARCH values)
        gru_input = np.column_stack([
            recent_returns,
            recent_garch_scaled
        ]).reshape(1, self.gru.sequence_length, 2)
        
        # Get GRU forecast
        gru_forecast = self.gru.predict(gru_input)
        
        # Ensure consistent shapes - flatten all to 1D
        garch_forecast = np.array(garch_forecast).flatten()
        gru_forecast = np.array(gru_forecast).flatten()
        
        # IMPORTANT: No scaling adjustment needed here
        # GRU was trained to predict actual GKYZ volatility directly
        # The scaling was only applied to GARCH inputs during training
        
        # Use optimized weights for hybrid combination
        combined_forecast = self.garch_weight * garch_forecast + self.gru_weight * gru_forecast
        
        return {
            'garch': garch_forecast,
            'gru': gru_forecast,
            'combined': combined_forecast,
            'weights': {'garch': self.garch_weight, 'gru': self.gru_weight}
        }
    
    def get_volatility_series(self):
        return {
            'garch': self.garch.conditional_volatility(),
            'combined': None  # Would need full historical predictions
        }
    
    def get_training_time(self):
        """Get the training time for comparison"""
        return self.training_time