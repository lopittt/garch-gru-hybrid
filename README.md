# GARCH-GRU Simulation Model

Implementation of hybrid GARCH-GRU model for financial volatility forecasting based on research paper 2310.01063v1.pdf.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python src/main.py
```

## Structure

- `src/data/` - Data fetching and preprocessing
- `src/models/` - GARCH, GRU, and hybrid models
- `src/simulation/` - Monte Carlo path simulation
- `src/visualization/` - Plotting utilities

## Output

The simulation generates:
- Simulated price paths comparison
- Volatility forecast analysis
- Return distribution plots
- Model performance statistics