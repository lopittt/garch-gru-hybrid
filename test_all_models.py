#!/usr/bin/env python3
"""Test all GRU models to find optimal configuration"""

import numpy as np
import pandas as pd
from src.data.fetcher import fetch_spy_data
from src.data.preprocessor import calculate_gkyz_volatility, scale_volatility
from src.models.garch import GARCH
from src.models.balanced_gru import BalancedGRUModel
from src.models.ultra_simple_gru import UltraSimpleGRUModel

def test_model(model_class, model_name, data_dict):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print('='*60)
    
    # Create model
    gru = model_class(sequence_length=6)
    
    # Prepare sequences
    sequences, targets = gru.prepare_sequences(
        data_dict['returns_abs'],
        data_dict['scaled_garch'],
        data_dict['volatility']
    )
    
    print(f"Total sequences: {len(sequences)}")
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_seq = sequences[:split_idx]
    train_tgt = targets[:split_idx]
    val_seq = sequences[split_idx:]
    val_tgt = targets[split_idx:]
    
    # Calculate parameter efficiency
    if hasattr(gru.model, 'parameters'):
        model_params = sum(p.numel() for p in gru.model.parameters())
        samples_per_param = len(train_seq) / model_params
        print(f"Model parameters: {model_params}")
        print(f"Training samples: {len(train_seq)}")
        print(f"Samples per parameter: {samples_per_param:.2f}")
        
        if samples_per_param < 10:
            status = "⚠️ Risk of overfitting"
        elif samples_per_param < 20:
            status = "⚠️ Marginal"
        else:
            status = "✅ Good ratio"
        print(status)
    
    # Train
    print(f"\nTraining {model_name}...")
    history = gru.train(
        train_seq, train_tgt,
        val_seq, val_tgt,
        epochs=150,
        batch_size=32
    )
    
    # Evaluate
    predictions = gru.predict(val_seq).flatten()
    
    # Calculate metrics
    mse = np.mean((val_tgt - predictions) ** 2)
    mae = np.mean(np.abs(val_tgt - predictions))
    ss_res = np.sum((val_tgt - predictions) ** 2)
    ss_tot = np.sum((val_tgt - np.mean(val_tgt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
    
    # Correlation
    if np.std(predictions) > 1e-6:
        correlation = np.corrcoef(val_tgt, predictions)[0, 1]
    else:
        correlation = 0
    
    results = {
        'name': model_name,
        'params': model_params if 'model_params' in locals() else 0,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'pred_std': predictions.std(),
        'actual_std': val_tgt.std()
    }
    
    print(f"\nResults for {model_name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.8f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Prediction std: {predictions.std():.6f}")
    
    return results

def main():
    print("="*80)
    print("COMPREHENSIVE GRU MODEL COMPARISON")
    print("="*80)
    
    # Get data (10 years worked best)
    print("\nLoading 10 years of SPY data...")
    data = fetch_spy_data(years_back=10)
    volatility = calculate_gkyz_volatility(data, window=10)
    returns = pd.Series(
        np.log(data['Close'] / data['Close'].shift(1)).dropna().values,
        index=data.index[1:]
    )
    
    # Align
    common_index = volatility.index.intersection(returns.index)
    volatility = volatility[common_index]
    returns = returns[common_index]
    
    print(f"Data shape: {len(returns)} samples")
    print(f"Volatility - Mean: {volatility.mean():.6f}, Std: {volatility.std():.6f}")
    
    # Fit GARCH
    print("\nFitting GARCH model...")
    garch = GARCH()
    garch.fit(returns)
    garch_vol = garch.conditional_volatility()
    
    # Scale volatility
    scaled_garch, lambda_scale = scale_volatility(garch_vol, volatility)
    print(f"Lambda scaling factor: {lambda_scale:.4f}")
    
    # Prepare data dict
    data_dict = {
        'returns_abs': np.abs(returns.values),
        'scaled_garch': scaled_garch,
        'volatility': volatility
    }
    
    # Test models
    models_to_test = [
        (BalancedGRUModel, "Balanced GRU"),
        (UltraSimpleGRUModel, "Ultra-Simple GRU")
    ]
    
    results = []
    for model_class, model_name in models_to_test:
        try:
            result = test_model(model_class, model_name, data_dict)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Params':<10} {'R²':<10} {'MSE':<12} {'MAE':<10} {'Corr':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<20} {r['params']:<10} {r['r2']:<10.4f} "
              f"{r['mse']:<12.8f} {r['mae']:<10.6f} {r['correlation']:<10.4f}")
    
    # Find best model
    best_model = max(results, key=lambda x: x['r2'])
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"✅ Best model: {best_model['name']}")
    print(f"   - R²: {best_model['r2']:.4f}")
    print(f"   - Parameters: {best_model['params']}")
    print(f"   - Correlation: {best_model['correlation']:.4f}")
    
    if best_model['r2'] > 0.7:
        print("\n🎉 Excellent performance achieved!")
    elif best_model['r2'] > 0.5:
        print("\n✅ Good performance achieved!")
    elif best_model['r2'] > 0:
        print("\n⚠️ Positive but marginal performance")
    else:
        print("\n❌ Model failed to achieve positive R²")
    
    return best_model

if __name__ == "__main__":
    best = main()