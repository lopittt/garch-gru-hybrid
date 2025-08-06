# src/data/preprocessor.py
# Calculates Garman-Klass-Yang-Zhang volatility estimator from OHLC data
# Prepares rolling windows for GARCH and GRU model training
# RELEVANT FILES: fetcher.py, garch.py, gru.py, main.py

import numpy as np
import pandas as pd

def calculate_gkyz_volatility(data, window=10):
    """Garman-Klass-Yang-Zhang volatility estimator"""
    
    open_close_prev = np.log(data['Open'] / data['Close'].shift(1))
    high_low = np.log(data['High'] / data['Low'])
    close_open = np.log(data['Close'] / data['Open'])
    
    term1 = open_close_prev ** 2
    term2 = 0.5 * (high_low ** 2)
    term3 = (2 * np.log(2) - 1) * (close_open ** 2)
    
    gkyz = np.sqrt(term1 + term2 - term3)
    gkyz_rolling = gkyz.rolling(window=window).mean()
    
    return gkyz_rolling.dropna()

def prepare_rolling_windows(data, returns, volatility, garch_window=504, gru_window=1008):
    windows = {
        'garch': [],
        'gru_train': [],
        'targets': []
    }
    
    for i in range(gru_window, len(returns)):
        windows['garch'].append(returns[i-garch_window:i])
        windows['gru_train'].append({
            'returns': returns[i-gru_window:i],
            'volatility': volatility[i-gru_window:i]
        })
        windows['targets'].append(volatility[i])
    
    return windows

def scale_volatility(garch_vol, gkyz_vol):
    """Scale GARCH volatility to match GKYZ magnitude"""
    scale = gkyz_vol.mean() / garch_vol.mean()
    return garch_vol * scale