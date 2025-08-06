# src/models/hybrid.py
# GARCH-GRU hybrid model combining econometric and deep learning approaches
# Integrates GARCH volatility forecasts with GRU predictions
# RELEVANT FILES: garch.py, gru.py, simulator.py, main.py

import numpy as np
import pandas as pd
from .garch import GARCH
from .gru import GRUModel

class GARCHGRUHybrid:
    def __init__(self):
        self.garch = GARCH(p=1, q=1)
        self.gru = GRUModel(sequence_length=6)
        self.is_fitted = False
        
    def fit(self, returns, volatility):
        # Fit GARCH model
        self.garch.fit(returns)
        garch_vol = self.garch.conditional_volatility()
        
        # Align data
        min_len = min(len(returns), len(volatility), len(garch_vol))
        returns_aligned = returns[-min_len:].values
        volatility_aligned = volatility[-min_len:].values
        garch_vol_aligned = garch_vol[-min_len:].values
        
        # Prepare GRU training data
        sequences, targets = self.gru.prepare_sequences(
            np.abs(returns_aligned),
            volatility_aligned,
            garch_vol_aligned
        )
        
        # Split into train/validation
        split_idx = int(0.67 * len(sequences))
        train_seq = sequences[:split_idx]
        train_tgt = targets[:split_idx]
        val_seq = sequences[split_idx:]
        val_tgt = targets[split_idx:]
        
        # Train GRU
        self.gru.train(train_seq, train_tgt, val_seq, val_tgt)
        self.gru.load_best_model()
        
        self.is_fitted = True
        return self
    
    def forecast(self, returns, volatility, horizon=1):
        # Get GARCH forecast
        garch_forecast = self.garch.forecast(horizon=horizon)
        
        # Get recent data for GRU
        recent_returns = np.abs(returns[-self.gru.sequence_length:].values)
        recent_volatility = volatility[-self.gru.sequence_length:].values
        recent_garch = self.garch.conditional_volatility()[-self.gru.sequence_length:].values
        
        # Prepare GRU input
        gru_input = np.column_stack([
            recent_returns,
            recent_volatility,
            recent_garch
        ]).reshape(1, self.gru.sequence_length, 3)
        
        # Get GRU forecast
        gru_forecast = self.gru.predict(gru_input)[0]
        
        # Combine forecasts (simple average for now)
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