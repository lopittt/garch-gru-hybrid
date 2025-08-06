# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of GARCH-GRU hybrid simulation model for financial volatility forecasting. Based on research paper 2310.01063v1.pdf. Focus on SPY daily data simulation with benchmark GARCH(1,1) comparison.

## Development Environment

**Primary Stack:**
- Python 3.9+
- PyTorch (preferred for ML components)
- pandas (data handling)
- numpy (numerical operations)
- yfinance (SPY data)
- matplotlib/seaborn (plotting)
- arch (GARCH modeling)

**Alternative Libraries:**
- TensorFlow/Keras (if PyTorch insufficient)
- scipy.optimize (GARCH parameter estimation)

## Data Specifications

- **Asset**: SPY (SPDR S&P 500 ETF)
- **Frequency**: Daily
- **End Date**: 2024-12-31
- **Target**: Garman-Klass-Yang-Zhang volatility estimator
- **Windows**: 504 days (GARCH), 1008 days (GRU training)

## Code Architecture

**Modular Structure:**
```
src/
├── data/
│   ├── fetcher.py      # SPY data download
│   └── preprocessor.py # GKYZ volatility calculation
├── models/
│   ├── garch.py        # Standard GARCH(1,1)
│   ├── gru.py          # PyTorch GRU network
│   └── hybrid.py       # GARCH-GRU integration
├── simulation/
│   └── simulator.py    # Path generation
├── visualization/
│   └── plotter.py      # Result plotting
└── main.py             # Linear workflow
```

## Implementation Modules

### 1. Data Module (src/data/)
**fetcher.py:**
- Download SPY data via yfinance
- Extract OHLC prices
- Handle missing data

**preprocessor.py:**
- Calculate GKYZ volatility estimator
- Compute log returns
- Rolling window preparation

### 2. Models Module (src/models/)
**garch.py:**
- GARCH(1,1) parameter estimation
- Conditional variance forecasting
- Maximum likelihood fitting

**gru.py:**
- PyTorch GRU network (512/256/128 architecture)
- Adam optimizer (lr=0.0009)
- MSE loss function
- Training/validation loops

**hybrid.py:**
- GARCH-GRU integration
- Combined volatility forecasting
- Model switching logic

### 3. Simulation Module (src/simulation/)
**simulator.py:**
- Monte Carlo path generation
- Volatility scenario simulation
- Benchmark comparison

### 4. Visualization Module (src/visualization/)
**plotter.py:**
- Simulated path visualization
- Volatility forecast comparison
- Model performance metrics

## Coding Standards

**Priorities:**
1. **Simplicity**: Minimal complexity, avoid over-engineering
2. **Readability**: Self-explanatory code, clear variable names
3. **Modularity**: Each function/class has single responsibility
4. **Linear Flow**: Avoid nested complexity, error handling only when essential

**Header Comments (Required for ALL files):**
```python
# src/models/garch.py
# Standard GARCH(1,1) model implementation
# Provides benchmark volatility forecasting for comparison with hybrid approach
# RELEVANT FILES: hybrid.py, preprocessor.py, simulator.py, main.py
```

**Style Guidelines:**
- Short functions (<20 lines)
- Minimal comments (code should be self-explanatory)
- Clear variable names (no abbreviations)
- Single return statement per function
- No verbose docstrings in exploratory phase

## Model Implementation Details

### GARCH(1,1) Benchmark:
- Conditional variance: h_t = ω + α·ε²_{t-1} + β·h_{t-1}
- Maximum likelihood estimation
- Rolling window re-estimation

### GARCH-GRU Hybrid:
- GARCH volatility as GRU input
- Target: GKYZ volatility estimates
- 3-layer GRU: [512, 256, 128] neurons
- ReLU activation, dropout=0.3
- L2 regularization=0.00001

### Training Setup:
- Batch size: 500
- Sequence length: 6 days
- Training epochs: 150
- Validation split: 33%

## Simulation Workflow

**Linear Process:**
1. Download SPY data (fetcher.py)
2. Calculate GKYZ volatility (preprocessor.py)
3. Fit GARCH(1,1) model (garch.py)
4. Train GRU network (gru.py)
5. Create hybrid model (hybrid.py)
6. Generate simulated paths (simulator.py)
7. Plot results (plotter.py)

## Expected Outputs

- Volatility forecast comparison plots
- Simulated price paths (GARCH vs GARCH-GRU)
- MSE performance metrics
- Model parameter summaries

## Development Commands

```bash
# Run full simulation
python src/main.py

# Individual module testing
python -m src.models.garch
python -m src.models.gru
python -m src.simulation.simulator
```