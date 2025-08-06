# src/data/preprocessor.py
# Calculates Garman-Klass-Yang-Zhang volatility estimator from OHLC data
# Prepares rolling windows for GARCH and GRU model training
# RELEVANT FILES: fetcher.py, garch.py, gru.py, main.py

import numpy as np
import pandas as pd

def calculate_gkyz_volatility(data, window=10):
    """Garman-Klass-Yang-Zhang volatility estimator - Equation 11 from paper"""
    
    # Following exact formula from Equation 11, Page 7
    c_prev = np.log(data['Close'].shift(1))  # C_{t-1}
    o = np.log(data['Open'])                 # O_t
    h = np.log(data['High'])                 # H_t
    l = np.log(data['Low'])                  # L_t
    c = np.log(data['Close'])                # C_t
    
    # Equation 11: GKYZ = (c_{t-1} - o_t)^2 + 0.5*(h_t - l_t)^2 + (2*ln(2) - 1)*(c_t - o_t)^2
    term1 = (c_prev - o) ** 2
    term2 = 0.5 * (h - l) ** 2
    term3 = (2 * np.log(2) - 1) * (c - o) ** 2
    
    gkyz_var = term1 + term2 + term3  # This gives variance
    gkyz_vol = np.sqrt(gkyz_var)      # Convert to volatility
    
    # Apply rolling window average
    gkyz_rolling = gkyz_vol.rolling(window=window).mean()
    
    return gkyz_rolling.dropna()

def prepare_rolling_windows(data, returns, volatility, garch_window=504, prediction_window=126):
    """Prepare rolling windows as described in Section 3.2, Page 6
    
    Uses 504 days (2 years) for GARCH estimation and 126 days (6 months) for prediction
    Creates rolling windows that move forward one day at a time
    """
    windows = []
    
    # Start from garch_window + prediction_window to ensure we have enough data
    start_idx = garch_window + prediction_window
    
    for i in range(start_idx, len(returns)):
        window = {
            'garch_data': {
                'returns': returns[i-garch_window-prediction_window:i-prediction_window],
                'start_date': data.index[i-garch_window-prediction_window] if hasattr(data, 'index') else i-garch_window-prediction_window,
                'end_date': data.index[i-prediction_window] if hasattr(data, 'index') else i-prediction_window
            },
            'prediction_period': {
                'returns': returns[i-prediction_window:i],
                'volatility': volatility[i-prediction_window:i],
                'start_date': data.index[i-prediction_window] if hasattr(data, 'index') else i-prediction_window,
                'end_date': data.index[i] if hasattr(data, 'index') else i
            },
            'window_id': len(windows)
        }
        windows.append(window)
    
    return windows

def scale_volatility(garch_vol, gkyz_vol):
    """Scale volatility according to Equations 12-13, Page 7
    
    Equation 12: σ̃²_{GARCH,t} = λ * σ²_{GARCH,t}
    Equation 13: λ = (1/T) * Σ(GKYZ²_t) / (1/T) * Σ(σ²_{GARCH,t})
    """
    # Calculate scaling factor λ (lambda) from Equation 13
    gkyz_var_mean = np.mean(gkyz_vol ** 2)  # Mean of GKYZ² 
    garch_var_mean = np.mean(garch_vol ** 2)  # Mean of σ²_GARCH
    
    lambda_scale = gkyz_var_mean / garch_var_mean if garch_var_mean > 0 else 1.0
    
    # Apply scaling from Equation 12: σ̃²_{GARCH,t} = λ * σ²_{GARCH,t}
    scaled_garch_var = lambda_scale * (garch_vol ** 2)
    scaled_garch_vol = np.sqrt(scaled_garch_var)
    
    return scaled_garch_vol, lambda_scale

def prepare_gru_sequences(returns, volatility, garch_forecasts, sequence_length=6):
    """Prepare sequences for GRU training with proper input structure
    
    Input should be GARCH forecasts, not historical volatility as in original implementation
    """
    sequences = []
    targets = []
    
    # Ensure all inputs have the same length
    min_len = min(len(returns), len(volatility), len(garch_forecasts))
    returns = returns[:min_len]
    volatility = volatility[:min_len] 
    garch_forecasts = garch_forecasts[:min_len]
    
    for i in range(sequence_length, min_len):
        # Create sequence with returns and GARCH forecasts (not historical volatility)
        seq = np.column_stack([
            returns[i-sequence_length:i],
            garch_forecasts[i-sequence_length:i]  # Key fix: use GARCH forecasts
        ])
        sequences.append(seq)
        targets.append(volatility[i])  # Target is still realized volatility
    
    return np.array(sequences), np.array(targets)