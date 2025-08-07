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

## Post-Run Analysis Protocol

**Claude must ALWAYS review the following after each model run:**

### 🔍 Mandatory Review Checklist

After any model execution, Claude must systematically review all diagnostic outputs and flag any concerning patterns. This is not optional.

#### 1. Training Phase Review

**Review the HTML diagnostic report's Training section:**
- **Loss curves**: Check for convergence, overfitting, or oscillations
- **Parameter evolution**: Verify GARCH parameters are economically reasonable  
- **Weight optimization**: Ensure hybrid weights converge to stable values (not stuck at 0.1 or 0.9)
- **Training time**: Flag unusually long/short training as potential issues

**✅ Expected patterns:**
- Loss curves should decrease and plateau
- Validation loss should track training loss without major divergence
- GARCH parameters should satisfy: ω > 0, 0 < α < 1, 0 < β < 1, α + β < 1
- Hybrid weights should converge away from boundaries

#### 2. Model Performance Review

**Review the HTML diagnostic report's Statistical section:**
- **R² validation**: Must be positive and reasonable (>0.3 for GARCH, >0.5 for hybrid)
- **Residual patterns**: Check for systematic biases, autocorrelation, or heteroscedasticity
- **Forecast accuracy**: Compare against paper benchmarks and persistence baseline
- **Statistical significance**: Verify improvements are statistically meaningful

**✅ Expected patterns:**
- R² > 0.3 for GARCH, R² > 0.5 for hybrid models
- Residuals should appear random (no patterns in time series plots)
- Q-Q plots should roughly follow diagonal line
- Statistical tests should mostly pass (p-values > 0.05 for residual tests)

#### 3. Simulation Quality Review

**Review the HTML diagnostic report's Simulation section:**
- **Path visualization**: Inspect for unrealistic jumps, trends, or artifacts
- **Distribution matching**: Compare return distributions (histograms, Q-Q plots)
- **Stylized facts**: Verify volatility clustering, fat tails are preserved
- **Monte Carlo stability**: Check simulation statistics converge with sample size

**✅ Expected patterns:**
- Simulated paths should look realistic (no extreme jumps > 10σ)
- Return distributions should match historical patterns
- Volatility clustering should be present in simulated data
- No single path should dominate (good diversity)

#### 4. Red Flag Indicators (Immediate Investigation Required)

**🚨 Critical Issues - Stop and investigate immediately:**
- Negative R² values for any model
- GARCH persistence > 0.99 or < 0.8
- Training loss not decreasing after 50 epochs
- Simulation paths with >20% daily moves
- Hybrid weights stuck at boundaries (0.1 or 0.9)
- Statistical tests all failing (p-values all < 0.01)
- Overall health score < 0.4

**⚠️ Warning Indicators - Investigate and document:**
- R² between 0.1-0.3 for any model
- High residual autocorrelation
- Simulation jump frequency > 1%
- Poor Monte Carlo convergence
- Model training time > 10x expected
- Overall health score 0.4-0.6

#### 5. Visual Analysis Requirements

**Must examine these diagnostic plots:**
- Training loss curves (convergence, overfitting)
- Residual plots (randomness, patterns)  
- Path simulations (realism, diversity)
- Distribution comparisons (Q-Q plots, histograms)
- Statistical test summaries (pass/fail status)

**Flag any:**
- Unusual patterns or systematic deviations
- Outliers or extreme values
- Asymmetries or skewness that seems unrealistic
- Distribution mismatches between simulated and historical data

#### 6. Benchmark Comparisons

**Always compare against:**
- Paper reported metrics (from research paper)
- Consistency with previous successful runs
- Improvement over naive benchmarks (persistence, historical mean)
- Expected ranges for financial volatility models

**Expected benchmarks:**
- GARCH(1,1) R²: 0.3-0.7 for daily volatility
- Hybrid model improvement: 10-30% over GARCH alone
- GARCH persistence: 0.85-0.98 for financial data
- Simulation realism score: >0.6

#### 7. Executive Summary Assessment

**After reviewing all diagnostics, provide:**
- Overall health assessment (pass/fail with reasoning)
- Top 3 concerns (if any) requiring attention
- Recommendations for improvement
- Confidence level in results (high/medium/low)

### 🎯 Review Documentation Template

For each run, Claude should provide:

```
## Post-Run Analysis Summary
**Run ID**: [timestamp]
**Overall Status**: ✅ PASS / ⚠️ WARNING / 🚨 FAIL

### Key Metrics Review
- GARCH R²: [value] ([expected range])
- Hybrid R²: [value] ([expected range])  
- Training convergence: [converged/failed]
- Simulation quality: [score]

### Issues Identified
**Critical**: [list or "None"]
**Warnings**: [list or "None"] 

### Validation Status
- Statistical tests: [X/Y passed]
- Model health score: [score]
- Benchmark comparison: [above/below expectations]

### Recommendations
[List of actionable items for improvement]

### Confidence Assessment
[High/Medium/Low] confidence in results because [reasoning]
```

This systematic review ensures that issues like negative R² values, poor convergence, or unrealistic simulations are caught immediately and addressed before considering results valid.