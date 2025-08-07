#!/usr/bin/env python3
"""Final test of hybrid model with positive R²"""

import numpy as np
import pandas as pd
from src.data.fetcher import fetch_spy_data
from src.data.preprocessor import calculate_gkyz_volatility
from src.models.hybrid import GARCHGRUHybrid
from src.evaluation.metrics import ModelEvaluator

print("="*60)
print("FINAL TEST: GARCH-GRU HYBRID MODEL")
print("="*60)

# Get data
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

# Split 80/20
train_size = int(0.8 * len(returns))
train_returns = returns[:train_size]
train_volatility = volatility[:train_size]
test_returns = returns[train_size:]
test_volatility = volatility[train_size:]

print(f"\nData: {len(returns)} total, {train_size} train, {len(test_returns)} test")

# Train hybrid
print("\nTraining hybrid model...")
hybrid = GARCHGRUHybrid()
hybrid.fit(data[:train_size], train_returns, train_volatility)

print(f"\n✅ Training complete!")
print(f"   Weights: GARCH={hybrid.garch_weight:.1%}, GRU={hybrid.gru_weight:.1%}")

# Generate 200 test forecasts for better evaluation
print("\nGenerating test forecasts (first 200 points)...")
test_points = min(200, len(test_returns) - hybrid.gru.sequence_length - 1)
hybrid_predictions = []

for i in range(hybrid.gru.sequence_length, hybrid.gru.sequence_length + test_points):
    current_returns = pd.concat([train_returns, test_returns[:i]])
    current_volatility = pd.concat([train_volatility, test_volatility[:i]])
    
    forecast = hybrid.forecast(current_returns, current_volatility, horizon=1)
    hybrid_predictions.append(forecast['combined'][0])

hybrid_predictions = np.array(hybrid_predictions)
actual_test = test_volatility.iloc[hybrid.gru.sequence_length+1:hybrid.gru.sequence_length+1+test_points].values

# Calculate R²
ss_res = np.sum((actual_test - hybrid_predictions) ** 2)
ss_tot = np.sum((actual_test - np.mean(actual_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\n{'='*60}")
print("RESULTS")
print('='*60)
print(f"Test samples evaluated: {len(actual_test)}")
print(f"Hybrid R²: {r2:.4f}")

if r2 > 0.5:
    print("\n🎉 EXCELLENT: Hybrid achieves R² > 0.5")
elif r2 > 0:
    print("\n✅ SUCCESS: Hybrid achieves positive R²")
else:
    print("\n❌ FAIL: Hybrid still has negative R²")
    
print(f"\nPrediction stats:")
print(f"  Mean: {hybrid_predictions.mean():.6f}")
print(f"  Std: {hybrid_predictions.std():.6f}")
print(f"  Min: {hybrid_predictions.min():.6f}")
print(f"  Max: {hybrid_predictions.max():.6f}")

print(f"\nActual stats:")
print(f"  Mean: {actual_test.mean():.6f}")
print(f"  Std: {actual_test.std():.6f}")
print(f"  Min: {actual_test.min():.6f}")
print(f"  Max: {actual_test.max():.6f}")