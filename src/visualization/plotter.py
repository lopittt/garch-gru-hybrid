# src/visualization/plotter.py
# Visualization of simulated paths and model comparisons
# Creates plots for price paths, volatility forecasts, and distribution analysis
# RELEVANT FILES: simulator.py, main.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('whitegrid')

def plot_simulated_paths(garch_paths, hybrid_paths, title='Simulated Price Paths'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GARCH paths
    ax1.plot(garch_paths[:, :10], alpha=0.5, linewidth=0.8)
    ax1.plot(np.mean(garch_paths, axis=1), 'k-', linewidth=2, label='Mean Path')
    ax1.fill_between(range(len(garch_paths)), 
                     np.percentile(garch_paths, 5, axis=1),
                     np.percentile(garch_paths, 95, axis=1),
                     alpha=0.2, color='gray', label='5-95 Percentile')
    ax1.set_title('GARCH(1,1) Model')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hybrid paths
    ax2.plot(hybrid_paths[:, :10], alpha=0.5, linewidth=0.8)
    ax2.plot(np.mean(hybrid_paths, axis=1), 'k-', linewidth=2, label='Mean Path')
    ax2.fill_between(range(len(hybrid_paths)),
                     np.percentile(hybrid_paths, 5, axis=1),
                     np.percentile(hybrid_paths, 95, axis=1),
                     alpha=0.2, color='gray', label='5-95 Percentile')
    ax2.set_title('GARCH-GRU Hybrid Model')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_volatility_comparison(garch_vol, gkyz_vol, forecast_comparison=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Align the data properly
    min_len = min(len(garch_vol), len(gkyz_vol))
    
    # Create proper time index
    time_index = range(min_len)
    
    # Historical volatility with proper indexing
    axes[0].plot(time_index[-252:], garch_vol.values[-252:], 
                label='GARCH Conditional Volatility', linewidth=1.5)
    axes[0].plot(time_index[-252:], gkyz_vol.values[-252:], 
                label='GKYZ Realized Volatility', linewidth=1.5, alpha=0.7)
    axes[0].set_title('Historical Volatility Comparison (Last 252 Days)')
    axes[0].set_ylabel('Volatility')
    axes[0].set_xlabel('Days')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Forecast comparison if provided
    if forecast_comparison:
        x = range(len(forecast_comparison['garch']))
        axes[1].plot(x, forecast_comparison['garch'], label='GARCH Forecast', linewidth=1.5)
        axes[1].plot(x, forecast_comparison['gru'], label='GRU Forecast', linewidth=1.5)
        axes[1].plot(x, forecast_comparison['combined'], label='Combined Forecast', 
                    linewidth=2, linestyle='--')
        axes[1].set_title('Volatility Forecasts')
        axes[1].set_xlabel('Forecast Horizon')
        axes[1].set_ylabel('Volatility')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_return_distributions(garch_paths, hybrid_paths):
    garch_returns = (garch_paths[-1, :] / garch_paths[0, :]) - 1
    hybrid_returns = (hybrid_paths[-1, :] / hybrid_paths[0, :]) - 1
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram comparison
    axes[0].hist(garch_returns, bins=30, alpha=0.6, label='GARCH', density=True)
    axes[0].hist(hybrid_returns, bins=30, alpha=0.6, label='Hybrid', density=True)
    axes[0].axvline(np.mean(garch_returns), color='blue', linestyle='--', 
                   label=f'GARCH Mean: {np.mean(garch_returns):.3f}')
    axes[0].axvline(np.mean(hybrid_returns), color='orange', linestyle='--',
                   label=f'Hybrid Mean: {np.mean(hybrid_returns):.3f}')
    axes[0].set_xlabel('Return')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Return Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    axes[1].scatter(np.sort(garch_returns), np.sort(hybrid_returns), alpha=0.5)
    axes[1].plot([min(garch_returns), max(garch_returns)], 
                [min(garch_returns), max(garch_returns)], 'r--', linewidth=2)
    axes[1].set_xlabel('GARCH Returns')
    axes[1].set_ylabel('Hybrid Returns')
    axes[1].set_title('Q-Q Plot: GARCH vs Hybrid')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_comparison_table(comparison_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=comparison_df.round(4).values,
                    rowLabels=comparison_df.index,
                    colLabels=comparison_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    return fig