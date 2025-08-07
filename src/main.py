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
from config.settings import get_settings

def main():
    # Load settings
    settings = get_settings()
    
    print("=" * 50)
    print("GARCH-GRU Simulation Pipeline")
    print("=" * 50)
    
    # Print current configuration
    if settings.get('training.verbose_training', True):
        settings.print_current_config()
    
    # Step 1: Fetch Data
    print("\n1. Fetching SPY data...")
    years_back = settings.get('data.years_back', 10)
    end_date = settings.get('data.end_date', '2024-12-31')
    spy_data = fetch_spy_data(end_date=end_date, years_back=years_back)
    log_returns = get_log_returns(spy_data)
    print(f"   Data shape: {spy_data.shape}")
    print(f"   Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
    print(f"   Configuration: {years_back} years ending {end_date}")
    
    # Step 2: Calculate Volatility
    print("\n2. Calculating GKYZ volatility...")
    gkyz_window = settings.get('data.gkyz_window', 10)
    gkyz_volatility = calculate_gkyz_volatility(spy_data, window=gkyz_window)
    print(f"   Volatility series length: {len(gkyz_volatility)}")
    print(f"   GKYZ window: {gkyz_window} days")
    
    # Step 3: Fit GARCH Model
    print("\n3. Fitting GARCH model...")
    garch_p = settings.get('data.garch_p', 1)
    garch_q = settings.get('data.garch_q', 1)
    garch_model = GARCH(p=garch_p, q=garch_q)
    garch_model.fit(log_returns)
    print(f"   GARCH({garch_p},{garch_q}) Parameters:")
    print(f"   ω={garch_model.params['omega']:.6f}, "
          f"α={garch_model.params['alpha']:.6f}, "
          f"β={garch_model.params['beta']:.6f}")
    
    # Step 4: Fit Hybrid Model (now automatically uses settings)
    print(f"\n4. Training GARCH-GRU hybrid model...")
    training_location = "Modal cloud" if settings.use_modal else "local CPU"
    print(f"   Training on: {training_location}")
    if settings.use_modal:
        print(f"   GPU type: {settings.modal_gpu}")
    print("   This may take a few minutes...")
    
    hybrid_model = GARCHGRUHybrid()  # Now automatically reads from settings
    hybrid_model.fit(spy_data, log_returns, gkyz_volatility)
    
    # Step 5: Simulate Paths
    print("\n5. Simulating price paths...")
    current_price = spy_data['Close'].iloc[-1]
    simulator = PathSimulator(initial_price=current_price)
    
    # Use simulation settings
    n_periods = settings.get('simulation.n_periods', 252)
    n_paths = settings.get('simulation.n_paths', 100)
    
    print(f"   Initial price: ${current_price:.2f}")
    print(f"   Simulating {n_periods} days, {n_paths} paths...")
    
    garch_paths = simulator.simulate_garch_paths(garch_model, n_periods=n_periods, n_paths=n_paths)
    hybrid_paths = simulator.simulate_hybrid_paths(
        hybrid_model, 
        log_returns.tail(504), 
        gkyz_volatility.tail(504),
        n_periods=n_periods, 
        n_paths=n_paths
    )
    
    # Step 6: Calculate Statistics
    print("\n6. Calculating statistics...")
    comparison = simulator.compare_models(garch_paths, hybrid_paths)
    print("\nModel Comparison:")
    print(comparison)
    
    # Step 7: Generate Plots
    print("\n7. Generating visualizations...")
    
    save_plots = settings.get('simulation.save_plots', True)
    plot_dpi = settings.get('simulation.plot_dpi', 300)
    
    plot_simulated_paths(garch_paths, hybrid_paths)
    if save_plots:
        plt.savefig('simulated_paths.png', dpi=plot_dpi, bbox_inches='tight')
    
    plot_volatility_comparison(
        garch_model.conditional_volatility(),
        gkyz_volatility
    )
    if save_plots:
        plt.savefig('volatility_comparison.png', dpi=plot_dpi, bbox_inches='tight')
    
    plot_return_distributions(garch_paths, hybrid_paths)
    if save_plots:
        plt.savefig('return_distributions.png', dpi=plot_dpi, bbox_inches='tight')
    
    plot_model_comparison_table(comparison)
    if save_plots:
        plt.savefig('model_comparison.png', dpi=plot_dpi, bbox_inches='tight')
    
    if save_plots:
        print(f"   Plots saved at {plot_dpi} DPI!")
    else:
        print("   Plot saving disabled in settings")
    
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