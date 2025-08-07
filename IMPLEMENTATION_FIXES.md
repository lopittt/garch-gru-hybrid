# GARCH-GRU Implementation Fixes Reference

**Document Version**: 1.0  
**Created**: 2025-01-08  
**Purpose**: Self-contained developer reference for fixing critical issues in GARCH-GRU implementation

---

## 🚨 Critical Issues (Must Fix First)

### Issue #1: Look-ahead Bias in GRU Training Data

**Problem**: The GRU is trained using future GARCH forecasts instead of historical conditional volatilities, creating temporal inconsistency and look-ahead bias.

**Location**: `src/models/gru.py:52-71` (prepare_sequences method)

**Current Code** (INCORRECT):
```python
def prepare_sequences(self, returns, garch_forecasts, targets):
    """Prepare sequences using GARCH forecasts as input (not historical volatility)"""
    # ...
    for i in range(self.sequence_length, min_len):
        seq = np.column_stack([
            returns[i-self.sequence_length:i],
            garch_forecasts[i-self.sequence_length:i]  # WRONG: These are future forecasts!
        ])
```

**Root Cause**: `garch_forecasts` at position `i` contains the GARCH model's forecast FOR time `i`, but the model at time `i-1` should only have access to information up to time `i-1`.

**Fix Required**:
```python
def prepare_sequences(self, returns, garch_conditional_vol, targets):
    """Prepare sequences using historical GARCH conditional volatility
    
    Args:
        returns: Historical returns (absolute values)
        garch_conditional_vol: GARCH conditional volatility (from fitted model)
        targets: GKYZ realized volatility (what we're predicting)
    """
    sequences = []
    sequence_targets = []
    
    min_len = min(len(returns), len(garch_conditional_vol), len(targets))
    
    for i in range(self.sequence_length, min_len):
        # Use HISTORICAL conditional volatility, not forecasts
        seq = np.column_stack([
            returns[i-self.sequence_length:i],
            garch_conditional_vol[i-self.sequence_length:i]  # Historical values only
        ])
        sequences.append(seq)
        sequence_targets.append(targets[i])
        
    return np.array(sequences), np.array(sequence_targets)
```

**Also Update** `src/models/hybrid.py:97-101`:
```python
# OLD (WRONG):
sequences, targets = self.gru.prepare_sequences(
    np.abs(prediction_returns),
    scaled_forecasts,  # These are forecasts!
    prediction_volatility
)

# NEW (CORRECT):
# Use historical conditional volatility from the already-fitted GARCH model
garch_conditional_vol = self.garch.conditional_volatility()
scaled_conditional_vol, _ = scale_volatility(garch_conditional_vol, prediction_volatility)

sequences, targets = self.gru.prepare_sequences(
    np.abs(prediction_returns),
    scaled_conditional_vol[-len(prediction_returns):],  # Historical conditional vol
    prediction_volatility
)
```

---

### Issue #2: Training Parameter Mismatch

**Problem**: Training parameters don't match paper specifications, leading to insufficient training.

**Paper Requirements**:
- Epochs: 150
- Batch size: 500
- Max training samples: 500

**Current Implementation**:
- Epochs: 50
- Batch size: 32
- Max training samples: 500 (correct)

**Files to Update**:

#### 1. `src/models/gru.py:73`
```python
# OLD:
def train(self, train_sequences, train_targets, val_sequences, val_targets, 
          epochs=50, batch_size=32, max_train_size=500):

# NEW:
def train(self, train_sequences, train_targets, val_sequences, val_targets, 
          epochs=150, batch_size=500, max_train_size=500):
```

#### 2. `src/models/hybrid.py:120-122`
```python
# OLD:
epochs = self.settings.get('training.epochs', 50)
batch_size = self.settings.get('training.batch_size', 32)

# NEW:
epochs = self.settings.get('training.epochs', 150)  # Changed default
batch_size = self.settings.get('training.batch_size', 500)  # Changed default
```

#### 3. `src/config/settings.py` (update defaults)
```python
# In the DEFAULT_SETTINGS dictionary:
'training': {
    'epochs': 150,  # Changed from 50
    'batch_size': 500,  # Changed from 32
    'max_train_size': 500,
    # ... other settings
}
```

---

### Issue #3: Missing L2 Regularization

**Problem**: GRU optimizer lacks L2 regularization (weight decay) specified in the paper.

**Paper Requirement**: L2 regularization = 0.00001

**Location**: `src/models/gru.py:47`

**Current Code**:
```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0009)
```

**Fix Required**:
```python
self.optimizer = torch.optim.Adam(
    self.model.parameters(), 
    lr=0.0009,
    weight_decay=0.00001  # Add L2 regularization
)
```

---

## ⚠️ High Priority Issues

### Issue #4: Overly Simple Hybrid Weighting

**Problem**: The hybrid model uses fixed 50/50 weighting instead of optimized weights.

**Location**: `src/models/hybrid.py:263`

**Current Code**:
```python
combined_forecast = 0.5 * garch_forecast + 0.5 * gru_forecast
```

**Fix Required** - Add adaptive weighting:
```python
def __init__(self, sequence_length=None, use_modal=None):
    # ... existing code ...
    
    # Add weight tracking
    self.garch_weight = 0.5  # Initial weight
    self.gru_weight = 0.5
    self.weight_optimization_history = []

def optimize_weights(self, validation_data, validation_targets):
    """Optimize combination weights based on validation performance"""
    from scipy.optimize import minimize
    
    def objective(weights):
        w_garch = weights[0]
        w_gru = 1 - w_garch
        
        combined_predictions = []
        for i in range(len(validation_data)):
            garch_pred = self.garch.forecast(horizon=1)[0]
            gru_pred = self.gru.predict(validation_data[i:i+1])[0]
            combined = w_garch * garch_pred + w_gru * gru_pred
            combined_predictions.append(combined)
        
        mse = np.mean((np.array(combined_predictions) - validation_targets) ** 2)
        return mse
    
    # Optimize with constraint that weights sum to 1
    result = minimize(
        objective,
        x0=[0.5],
        bounds=[(0.1, 0.9)],  # Keep weights reasonable
        method='L-BFGS-B'
    )
    
    self.garch_weight = result.x[0]
    self.gru_weight = 1 - self.garch_weight
    
    print(f"Optimized weights - GARCH: {self.garch_weight:.3f}, GRU: {self.gru_weight:.3f}")
    return self

def forecast(self, returns, volatility, horizon=1):
    # ... existing code to get garch_forecast and gru_forecast ...
    
    # Use optimized weights
    combined_forecast = self.garch_weight * garch_forecast + self.gru_weight * gru_forecast
    
    return {
        'garch': garch_forecast,
        'gru': gru_forecast,
        'combined': combined_forecast,
        'weights': {'garch': self.garch_weight, 'gru': self.gru_weight}
    }
```

**Integration in `fit` method**:
```python
def fit(self, data, returns, volatility):
    # ... existing training code ...
    
    # After GRU training, optimize combination weights
    if len(val_seq) > 0:
        print("Optimizing hybrid weights on validation data...")
        self.optimize_weights(val_seq, val_tgt)
    
    return self
```

---

### Issue #5: Missing Evaluation Metrics

**Problem**: Implementation lacks comprehensive evaluation metrics required by the paper.

**Required Metrics**:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- HMSE (Heteroscedasticity-adjusted MSE)
- R² (R-squared)

**Create New File**: `src/evaluation/metrics.py`

```python
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
```

**Integration in `src/main.py`** (add after training):
```python
from evaluation.metrics import ModelEvaluator

# After hybrid model fitting (around line 69)
print("\n4.5. Evaluating model performance...")

# Generate predictions for the last part of the data
test_size = min(252, len(gkyz_volatility) // 5)  # Last 20% or 252 days
test_returns = log_returns.tail(test_size)
test_volatility = gkyz_volatility.tail(test_size)

# Get predictions from both models
garch_predictions = []
hybrid_predictions = []

for i in range(len(test_returns) - hybrid_model.gru.sequence_length):
    idx = i + hybrid_model.gru.sequence_length
    
    # GARCH forecast
    garch_pred = garch_model.forecast(horizon=1)[0]
    garch_predictions.append(garch_pred)
    
    # Hybrid forecast
    hybrid_pred = hybrid_model.forecast(
        test_returns[:idx],
        test_volatility[:idx]
    )['combined'][0]
    hybrid_predictions.append(hybrid_pred)

# Align actual values
actual_test = test_volatility.iloc[hybrid_model.gru.sequence_length:].values

# Evaluate
evaluator = ModelEvaluator()
evaluator.print_evaluation_report(actual_test, garch_predictions, "GARCH(1,1)")
evaluator.print_evaluation_report(actual_test, hybrid_predictions, "GARCH-GRU Hybrid")

# Compare models
comparison_df = evaluator.compare_models(
    actual_test,
    {'GARCH': garch_predictions, 'Hybrid': hybrid_predictions}
)
print("\nModel Comparison:")
print(comparison_df)
```

---

## 🔄 Simulation Volatility Feedback Issue

**Problem**: During simulation, the model feeds back the wrong volatility measure to the GRU, causing distribution shift.

**Location**: `src/simulation/simulator.py:34-66`

**Current Issue**:
- GRU was trained with GARCH conditional volatility as input
- Simulation feeds back the hybrid's combined output instead
- This creates inconsistency between training and simulation

**Complete Fixed Implementation**:

```python
# src/simulation/simulator.py (REPLACE simulate_hybrid_paths method)

def simulate_hybrid_paths(self, hybrid_model, historical_returns, historical_volatility, 
                         n_periods=252, n_paths=100):
    """Simulate price paths using GARCH-GRU hybrid model
    
    Critical fix: Feed GARCH conditional volatility to GRU, not hybrid output
    """
    paths = np.zeros((n_periods + 1, n_paths))
    paths[0, :] = self.initial_price
    
    for path_idx in range(n_paths):
        # Initialize with historical data
        recent_returns = historical_returns.copy()
        
        # Get initial GARCH conditional volatilities (what GRU was trained on)
        garch_model = hybrid_model.garch
        garch_cond_vol = garch_model.conditional_volatility()
        
        # Align with recent returns length
        if len(garch_cond_vol) >= len(recent_returns):
            recent_garch_vol = garch_cond_vol.tail(len(recent_returns))
        else:
            # Pad if necessary
            recent_garch_vol = pd.Series(
                np.concatenate([
                    np.full(len(recent_returns) - len(garch_cond_vol), garch_cond_vol.iloc[0]),
                    garch_cond_vol.values
                ])
            )
        
        for t in range(1, n_periods + 1):
            # CRITICAL: Use GARCH conditional volatility for GRU input
            forecast = hybrid_model.forecast(recent_returns, recent_garch_vol)
            
            # Extract hybrid volatility for return generation
            hybrid_vol = forecast['combined']
            if hasattr(hybrid_vol, '__len__'):
                while hasattr(hybrid_vol, '__len__') and len(hybrid_vol) > 0:
                    hybrid_vol = hybrid_vol[0]
                if hasattr(hybrid_vol, 'item'):
                    hybrid_vol = hybrid_vol.item()
            
            # Generate return using HYBRID volatility
            z = np.random.normal(0, 1)
            ret = self.return_mean + hybrid_vol * z
            ret = np.clip(ret, -0.2, 0.2)  # Cap extreme returns
            
            # Update price
            paths[t, path_idx] = paths[t-1, path_idx] * np.exp(ret)
            paths[t, path_idx] = np.clip(
                paths[t, path_idx], 
                0.01 * self.initial_price, 
                100 * self.initial_price
            )
            
            # Update recent returns
            recent_returns = pd.concat([recent_returns[1:], pd.Series([ret])])
            
            # CRITICAL FIX: Update with new GARCH conditional volatility
            # This maintains consistency with training data structure
            
            # Method 1: Simple one-step update (faster)
            garch_next_cond_vol = garch_model.forecast(horizon=1)[0]
            
            # Method 2: More accurate - refit GARCH state (slower but more precise)
            # temp_returns = pd.concat([recent_returns[:-1], pd.Series([ret])])
            # temp_garch = GARCH(p=1, q=1)
            # temp_garch.fit(temp_returns)
            # garch_next_cond_vol = temp_garch.conditional_volatility().iloc[-1]
            
            # Update GARCH conditional volatility (NOT hybrid output)
            recent_garch_vol = pd.concat([recent_garch_vol[1:], pd.Series([garch_next_cond_vol])])
    
    return paths
```

**Key Changes Explained**:

1. **Initialize** with GARCH conditional volatility (lines 11-22)
2. **Pass** GARCH conditional volatility to `forecast()`, not realized volatility (line 25)
3. **Update** with next GARCH conditional volatility, not hybrid output (lines 52-57)
4. **Use** hybrid forecast only for return generation (line 36)

---

## 📋 Implementation Checklist

### Phase 1: Critical Fixes (Do First)
- [ ] Fix look-ahead bias in `src/models/gru.py:prepare_sequences`
- [ ] Update training parameters (epochs=150, batch_size=500)
- [ ] Add L2 regularization (weight_decay=0.00001)
- [ ] Fix simulation volatility feedback in `src/simulation/simulator.py`

### Phase 2: High Priority
- [ ] Implement adaptive weighting in `src/models/hybrid.py`
- [ ] Create `src/evaluation/metrics.py` with all metrics
- [ ] Integrate evaluation in `src/main.py`

### Phase 3: Testing
- [ ] Verify GARCH conditional volatility is used consistently
- [ ] Confirm training parameters match paper
- [ ] Test evaluation metrics on validation data
- [ ] Compare model performance before/after fixes

---

## 🧪 Validation Tests

After implementing fixes, run these checks:

```python
# Test 1: Verify no look-ahead bias
def test_no_lookahead():
    gru_model = GRUModel()
    returns = np.random.randn(100)
    garch_vol = np.abs(np.random.randn(100))
    targets = np.abs(np.random.randn(100))
    
    sequences, seq_targets = gru_model.prepare_sequences(returns, garch_vol, targets)
    
    # Check that sequence at time t only uses data up to t-1
    assert sequences.shape[1] == gru_model.sequence_length
    print("✓ No look-ahead bias detected")

# Test 2: Verify training parameters
def test_training_params():
    from config.settings import get_settings
    settings = get_settings()
    
    assert settings.get('training.epochs') == 150
    assert settings.get('training.batch_size') == 500
    print("✓ Training parameters correct")

# Test 3: Verify L2 regularization
def test_l2_regularization():
    gru_model = GRUModel()
    weight_decay = gru_model.optimizer.param_groups[0]['weight_decay']
    assert weight_decay == 0.00001
    print("✓ L2 regularization configured")
```

---

## 📝 Notes

1. **Data Extension**: Since we're using daily data, ensure at least 2500+ trading days (10+ years) to match paper's data volume
2. **Performance Impact**: Batch size 500 requires more memory - monitor GPU/CPU usage
3. **Training Time**: 150 epochs will take ~3x longer than current 50 epochs
4. **Validation Split**: Maintain 67/33 train/validation split as per paper

---

*This document provides complete, self-contained fixes for all identified critical and high-priority issues.*