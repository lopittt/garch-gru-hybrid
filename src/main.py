# src/main.py
# Main workflow orchestrating GARCH-GRU simulation pipeline
# Linear execution from data fetching to visualization
# RELEVANT FILES: fetcher.py, garch.py, hybrid.py, simulator.py, plotter.py

import matplotlib.pyplot as plt
from data.fetcher import fetch_spy_data, get_log_returns
from data.preprocessor import calculate_gkyz_volatility
from models.garch import GARCH
from models.hybrid import GARCHGRUHybrid
from simulation.simulator import PathSimulator
from visualization.plotter import (
    plot_simulated_paths, 
    plot_volatility_comparison, 
    plot_return_distributions,
    plot_model_comparison_table
)

def main():
    print("=" * 50)
    print("GARCH-GRU Simulation Pipeline")
    print("=" * 50)
    
    # Step 1: Fetch Data
    print("\n1. Fetching SPY data...")
    spy_data = fetch_spy_data(end_date='2024-12-31', years_back=10)
    log_returns = get_log_returns(spy_data)
    print(f"   Data shape: {spy_data.shape}")
    print(f"   Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
    
    # Step 2: Calculate Volatility
    print("\n2. Calculating GKYZ volatility...")
    gkyz_volatility = calculate_gkyz_volatility(spy_data, window=10)
    print(f"   Volatility series length: {len(gkyz_volatility)}")
    
    # Step 3: Fit GARCH Model
    print("\n3. Fitting GARCH(1,1) model...")
    garch_model = GARCH(p=1, q=1)
    garch_model.fit(log_returns)
    print(f"   Parameters: ω={garch_model.params['omega']:.6f}, "
          f"α={garch_model.params['alpha']:.6f}, "
          f"β={garch_model.params['beta']:.6f}")
    
    # Step 4: Fit Hybrid Model
    print("\n4. Training GARCH-GRU hybrid model...")
    print("   This may take a few minutes...")
    hybrid_model = GARCHGRUHybrid()
    hybrid_model.fit(log_returns, gkyz_volatility)
    print("   Training complete!")
    
    # Step 5: Simulate Paths
    print("\n5. Simulating price paths...")
    current_price = spy_data['Close'].iloc[-1]
    simulator = PathSimulator(initial_price=current_price)
    
    print(f"   Initial price: ${current_price:.2f}")
    print("   Simulating 252 days (1 year), 100 paths...")
    
    garch_paths = simulator.simulate_garch_paths(garch_model, n_periods=252, n_paths=100)
    hybrid_paths = simulator.simulate_hybrid_paths(
        hybrid_model, 
        log_returns.tail(504), 
        gkyz_volatility.tail(504),
        n_periods=252, 
        n_paths=100
    )
    
    # Step 6: Calculate Statistics
    print("\n6. Calculating statistics...")
    comparison = simulator.compare_models(garch_paths, hybrid_paths)
    print("\nModel Comparison:")
    print(comparison)
    
    # Step 7: Generate Plots
    print("\n7. Generating visualizations...")
    
    plot_simulated_paths(garch_paths, hybrid_paths)
    plt.savefig('simulated_paths.png', dpi=300, bbox_inches='tight')
    
    plot_volatility_comparison(
        garch_model.conditional_volatility(),
        gkyz_volatility
    )
    plt.savefig('volatility_comparison.png', dpi=300, bbox_inches='tight')
    
    plot_return_distributions(garch_paths, hybrid_paths)
    plt.savefig('return_distributions.png', dpi=300, bbox_inches='tight')
    
    plot_model_comparison_table(comparison)
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    print("   Plots saved!")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 50)
    print("Simulation Complete!")
    print("=" * 50)
    
    return {
        'data': spy_data,
        'returns': log_returns,
        'volatility': gkyz_volatility,
        'garch_model': garch_model,
        'hybrid_model': hybrid_model,
        'garch_paths': garch_paths,
        'hybrid_paths': hybrid_paths,
        'comparison': comparison
    }

if __name__ == "__main__":
    results = main()