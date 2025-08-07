# src/main.py
# Main workflow orchestrating GARCH-GRU simulation pipeline
# Linear execution from data fetching to visualization
# RELEVANT FILES: fetcher.py, garch.py, hybrid.py, simulator.py, plotter.py

import numpy as np
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
from evaluation.metrics import ModelEvaluator
from diagnostics.report_generator import ReportGenerator

def main():
    # Load settings and start timing
    import time
    start_time = time.time()
    main._start_time = start_time  # Store for later access
    
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
    
    # Step 4.5: Evaluate model performance
    print("\n4.5. Evaluating model performance...")
    
    # Generate predictions for the last part of the data
    test_size = min(252, len(gkyz_volatility) // 5)  # Last 20% or 252 days
    test_returns = log_returns.tail(test_size)
    test_volatility = gkyz_volatility.tail(test_size)
    
    # Get predictions from both models
    print("   Generating out-of-sample predictions...")
    
    # For GARCH: Use rolling one-step-ahead forecasts on test data
    # This properly updates the model with new information for accurate predictions
    garch_predictions = garch_model.rolling_forecast(test_returns)
    
    # For Hybrid: Generate rolling predictions with proper data updates
    hybrid_predictions = []
    for i in range(hybrid_model.gru.sequence_length, len(test_returns) - 1):
        # Combine train and test data up to current point
        current_returns = pd.concat([log_returns[:train_size], test_returns[:i]])
        current_volatility = pd.concat([gkyz_volatility[:train_size], test_volatility[:i]])
        
        # Get hybrid forecast
        hybrid_pred = hybrid_model.forecast(
            current_returns,
            current_volatility
        )['combined'][0]
        hybrid_predictions.append(hybrid_pred)
    
    # Convert list to array
    hybrid_predictions = np.array(hybrid_predictions)
    
    # Align predictions with actuals
    # Forecasts predict next period, so align accordingly
    # Hybrid predictions start from sequence_length and predict next period
    actual_test = test_volatility.iloc[hybrid_model.gru.sequence_length+1:].values
    # GARCH predictions need same alignment
    garch_predictions = garch_predictions[hybrid_model.gru.sequence_length:-1]
    
    # Ensure all arrays have same length
    min_len = min(len(actual_test), len(garch_predictions), len(hybrid_predictions))
    actual_test = actual_test[:min_len]
    garch_predictions = garch_predictions[:min_len]
    hybrid_predictions = hybrid_predictions[:min_len]
    
    # Evaluate
    evaluator = ModelEvaluator()
    evaluator.print_evaluation_report(actual_test, garch_predictions, "GARCH(1,1)")
    evaluator.print_evaluation_report(actual_test, hybrid_predictions, "GARCH-GRU Hybrid")
    
    # Compare models
    comparison_df = evaluator.compare_models(
        actual_test,
        {'GARCH': garch_predictions, 'Hybrid': hybrid_predictions}
    )
    print("\nModel Comparison:")
    print(comparison_df)
    
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
    
    # Step 8: Generate Comprehensive Diagnostic Report
    print("\n8. Generating comprehensive diagnostic report...")
    
    end_time = time.time()
    start_time = getattr(main, '_start_time', end_time - 300)  # Fallback if not set
    execution_time = end_time - start_time
    
    # Initialize report generator
    report_generator = ReportGenerator(output_dir="reports", save_plots=True)
    
    # Prepare data for report
    model_data = {
        'model': hybrid_model,
        'returns': log_returns,
        'volatility': gkyz_volatility,
        'actual': actual_test,
        'predicted': hybrid_predictions,
        'model_name': 'GARCH-GRU Hybrid'
    }
    
    simulation_data = {
        'simulated_paths': hybrid_paths,
        'historical_returns': log_returns.values,
        'model_name': 'GARCH-GRU Hybrid'
    }
    
    performance_data = {
        'r2': comparison_df.loc['Hybrid', 'R²'] if 'Hybrid' in comparison_df.index and 'R²' in comparison_df.columns else 0,
        'mse': comparison_df.loc['Hybrid', 'MSE'] if 'Hybrid' in comparison_df.index and 'MSE' in comparison_df.columns else 0,
        'mae': comparison_df.loc['Hybrid', 'MAE'] if 'Hybrid' in comparison_df.index and 'MAE' in comparison_df.columns else 0
    }
    
    execution_info = {
        'total_time': execution_time,
        'resources': {'memory_usage': 0}  # Placeholder
    }
    
    # Generate comprehensive report
    report = report_generator.generate_comprehensive_report(
        model_name="GARCH-GRU Hybrid",
        configuration=settings._settings,
        model_data=model_data,
        simulation_data=simulation_data,
        performance_data=performance_data,
        execution_info=execution_info
    )
    
    # Export reports
    html_path = report_generator.export_html_report(report)
    json_path = report_generator.export_json_report(report)
    
    print(f"   📊 Diagnostic report generated:")
    print(f"   HTML: {html_path}")
    print(f"   JSON: {json_path}")
    
    # Print executive summary
    print(f"\n   🎯 EXECUTIVE SUMMARY:")
    print(f"   Overall Health Score: {report.overall_health_score:.1%}")
    print(f"   Validation Status: {'PASS' if report.passed_validation else 'FAIL'}")
    print(f"   Critical Issues: {len(report.critical_issues)}")
    print(f"   Warnings: {len(report.warnings)}")
    
    if report.critical_issues:
        print(f"\n   🚨 CRITICAL ISSUES:")
        for issue in report.critical_issues:
            print(f"      • {issue}")
    
    if report.recommendations:
        print(f"\n   💡 RECOMMENDATIONS:")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"      • {rec}")
    
    print("\n" + "=" * 50)
    print("🎉 Complete workflow finished with diagnostic report!")
    print("📖 Review the HTML report for comprehensive analysis")
    
    # Clean up temporary files
    report_generator.cleanup_plots()
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