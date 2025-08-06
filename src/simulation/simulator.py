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
        paths = np.zeros((n_periods + 1, n_paths))
        paths[0, :] = self.initial_price
        
        for path_idx in range(n_paths):
            recent_returns = historical_returns.copy()
            recent_volatility = historical_volatility.copy()
            
            for t in range(1, n_periods + 1):
                # Get volatility forecast
                forecast = hybrid_model.forecast(recent_returns, recent_volatility)
                vol = forecast['combined'][0] if hasattr(forecast['combined'], '__len__') else forecast['combined']
                
                # Generate return
                z = np.random.normal(0, 1)
                ret = self.return_mean + vol * z
                
                # Update price
                paths[t, path_idx] = paths[t-1, path_idx] * np.exp(ret)
                
                # Update rolling windows (simplified)
                recent_returns = pd.concat([recent_returns[1:], pd.Series([ret])])
                recent_volatility = pd.concat([recent_volatility[1:], pd.Series([vol])])
        
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