# src/main_demo.py
# Demo version of GARCH-GRU simulation with faster execution
# Reduced data size and training epochs for quick demonstration
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
    print("GARCH-GRU Simulation Demo")
    print("=" * 50)
    
    # Step 1: Fetch Data (smaller dataset for demo)
    print("\n1. Fetching SPY data...")
    spy_data = fetch_spy_data(end_date='2024-12-31', years_back=3)
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
    
    # Step 4: Fit Hybrid Model (reduced training)
    print("\n4. Training GARCH-GRU hybrid model...")
    print("   Using 10 epochs for quick demo...")
    hybrid_model = GARCHGRUHybrid()
    hybrid_model.fit(log_returns, gkyz_volatility)
    print("   Training complete!")
    
    # Step 5: Simulate Paths (smaller simulation)
    print("\n5. Simulating price paths...")
    current_price = spy_data['Close'].iloc[-1]
    simulator = PathSimulator(initial_price=current_price)
    
    print(f"   Initial price: ${current_price:.2f}")
    print("   Simulating 126 days (6 months), 50 paths...")
    
    garch_paths = simulator.simulate_garch_paths(garch_model, n_periods=126, n_paths=50)
    hybrid_paths = simulator.simulate_hybrid_paths(
        hybrid_model, 
        log_returns.tail(252), 
        gkyz_volatility.tail(252),
        n_periods=126, 
        n_paths=50
    )
    
    # Step 6: Calculate Statistics
    print("\n6. Calculating statistics...")
    comparison = simulator.compare_models(garch_paths, hybrid_paths)
    print("\nModel Comparison:")
    print(comparison)
    
    # Step 7: Generate Plots
    print("\n7. Generating visualizations...")
    
    plot_simulated_paths(garch_paths, hybrid_paths, title='GARCH-GRU Demo Simulation')
    plt.savefig('demo_simulated_paths.png', dpi=300, bbox_inches='tight')
    
    plot_volatility_comparison(
        garch_model.conditional_volatility(),
        gkyz_volatility
    )
    plt.savefig('demo_volatility_comparison.png', dpi=300, bbox_inches='tight')
    
    plot_return_distributions(garch_paths, hybrid_paths)
    plt.savefig('demo_return_distributions.png', dpi=300, bbox_inches='tight')
    
    plot_model_comparison_table(comparison)
    plt.savefig('demo_model_comparison.png', dpi=300, bbox_inches='tight')
    
    print("   Plots saved!")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("Files saved: demo_*.png")
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