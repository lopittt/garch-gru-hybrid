# src/models/garch.py
# Standard GARCH(1,1) model implementation using simple maximum likelihood
# Provides benchmark volatility forecasting for comparison with hybrid approach
# RELEVANT FILES: hybrid.py, preprocessor.py, simulator.py, main.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class GARCH:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.params = {}
        self.volatility = None
        self.returns = None
    
    def _garch_likelihood(self, params, returns):
        omega, alpha, beta = params
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        likelihood = np.sum(np.log(sigma2) + returns**2 / sigma2)
        return likelihood
    
    def fit(self, returns):
        self.returns = returns.values if hasattr(returns, 'values') else returns
        
        # Scale returns for numerical stability
        self.returns = self.returns * 100
        
        # Better initial parameters based on unconditional variance
        unconditional_var = np.var(self.returns)
        initial_params = [0.1 * unconditional_var, 0.05, 0.9]
        
        # Tighter constraints for stability
        bounds = [(1e-6, unconditional_var), (0.01, 0.3), (0.1, 0.95)]
        constraints = {'type': 'ineq', 'fun': lambda x: 0.98 - x[1] - x[2]}
        
        # Multiple optimization attempts for robustness
        best_result = None
        best_likelihood = np.inf
        
        for attempt in range(3):
            try:
                result = minimize(
                    self._garch_likelihood,
                    initial_params,
                    args=(self.returns,),
                    bounds=bounds,
                    constraints=constraints,
                    method='SLSQP'
                )
                
                if result.fun < best_likelihood and result.success:
                    best_result = result
                    best_likelihood = result.fun
                    
                # Try different starting points
                initial_params = [np.random.uniform(1e-6, unconditional_var),
                                np.random.uniform(0.01, 0.2),
                                np.random.uniform(0.7, 0.95)]
            except:
                continue
        
        if best_result is None:
            # Fallback to simple parameters
            omega, alpha, beta = 0.1 * unconditional_var, 0.05, 0.9
        else:
            omega, alpha, beta = best_result.x
            
        self.params = {'omega': omega, 'alpha': alpha, 'beta': beta}
        
        # Rescale parameters back
        self.params['omega'] = omega / (100 ** 2)
        self.returns = self.returns / 100
        
        # Calculate conditional volatility
        self._calculate_volatility()
        return self
    
    def _calculate_volatility(self):
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        T = len(self.returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(self.returns)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * sigma2[t-1]
        
        self.volatility = np.sqrt(sigma2)
    
    def forecast(self, horizon=1):
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        last_return = self.returns[-1]
        last_var = self.volatility[-1]**2
        
        forecast_var = omega + alpha * last_return**2 + beta * last_var
        return np.array([np.sqrt(forecast_var)])
    
    def conditional_volatility(self):
        return pd.Series(self.volatility, index=range(len(self.volatility)))
    
    def rolling_forecast(self, new_returns):
        """Generate rolling one-step-ahead forecasts for new data"""
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        
        # Convert new returns to numpy array
        if isinstance(new_returns, pd.Series):
            new_returns = new_returns.values
        
        # No scaling needed - returns are already in correct scale
        # (The model params were already rescaled in fit())
        
        # Initialize with last known values from training
        forecasts = []
        last_return = self.returns[-1]  # Last return from training
        last_var = self.volatility[-1]**2  # Last variance from training
        
        # First forecast uses last training data
        forecast_var = omega + alpha * last_return**2 + beta * last_var
        forecasts.append(np.sqrt(forecast_var))
        
        # Subsequent forecasts use new returns
        for t in range(len(new_returns) - 1):
            # Use return at t to forecast volatility at t+1
            forecast_var = omega + alpha * new_returns[t]**2 + beta * forecast_var
            forecasts.append(np.sqrt(forecast_var))
        
        return np.array(forecasts)
    
    def simulate(self, n_periods, n_paths=1):
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        
        simulations = np.zeros((n_periods, n_paths))
        sigma2 = np.full(n_paths, self.volatility[-1]**2)
        
        for t in range(n_periods):
            z = np.random.normal(0, 1, n_paths)
            returns = np.sqrt(sigma2) * z
            simulations[t, :] = returns
            
            # Update variance for next period
            sigma2 = omega + alpha * returns**2 + beta * sigma2
        
        return simulations