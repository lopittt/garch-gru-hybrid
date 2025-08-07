# src/simulation/simulator.py
# Monte Carlo simulation for price paths using GARCH and hybrid models
# Generates multiple scenarios for volatility and price evolution
# RELEVANT FILES: garch.py, hybrid.py, plotter.py, main.py

import numpy as np
import pandas as pd

class PathSimulator:
    def __init__(self, initial_price, initial_return_mean=0.0005):
        self.initial_price = initial_price
        self.return_mean = initial_return_mean
        
    def simulate_garch_paths(self, garch_model, n_periods=252, n_paths=100):
        paths = np.zeros((n_periods + 1, n_paths))
        paths[0, :] = self.initial_price
        
        # Simulate returns using GARCH with proper scaling
        simulated_returns = garch_model.simulate(n_periods, n_paths)
        
        # Cap extreme returns for stability
        simulated_returns = np.clip(simulated_returns, -0.2, 0.2)
        
        for t in range(1, n_periods + 1):
            # Use geometric Brownian motion with drift
            returns = simulated_returns[t-1, :] + self.return_mean
            paths[t, :] = paths[t-1, :] * np.exp(returns)
            
            # Additional safety check
            paths[t, :] = np.clip(paths[t, :], 0.01 * self.initial_price, 100 * self.initial_price)
        
        return paths
    
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
                
                # Simple one-step update (faster)
                garch_next_cond_vol = garch_model.forecast(horizon=1)[0]
                
                # Update GARCH conditional volatility (NOT hybrid output)
                recent_garch_vol = pd.concat([recent_garch_vol[1:], pd.Series([garch_next_cond_vol])])
        
        return paths
    
    def calculate_statistics(self, paths):
        final_prices = paths[-1, :]
        returns = (final_prices / paths[0, :]) - 1
        
        stats = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95),
            'median': np.median(final_prices)
        }
        
        return stats
    
    def compare_models(self, garch_paths, hybrid_paths):
        garch_stats = self.calculate_statistics(garch_paths)
        hybrid_stats = self.calculate_statistics(hybrid_paths)
        
        comparison = pd.DataFrame({
            'GARCH': garch_stats,
            'Hybrid': hybrid_stats
        })
        
        return comparison