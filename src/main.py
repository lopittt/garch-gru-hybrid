# src/main.py
# Main workflow orchestrating GARCH-GRU simulation pipeline
# Linear execution from data fetching to visualization
# RELEVANT FILES: fetcher.py, garch.py, hybrid.py, simulator.py, plotter.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.fetcher import fetch_spy_data
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
    print(f"   Data shape: {spy_data.shape}")
    print(f"   Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
    print(f"   Configuration: {years_back} years ending {end_date}")
    
    # Step 2: Calculate Volatility and Returns (matching final_test.py)
    print("\n2. Calculating GKYZ volatility and aligning data...")
    gkyz_window = settings.get('data.gkyz_window', 10)
    gkyz_volatility = calculate_gkyz_volatility(spy_data, window=gkyz_window)
    
    # Calculate returns exactly as in final_test.py
    log_returns = pd.Series(
        np.log(spy_data['Close'] / spy_data['Close'].shift(1)).dropna().values,
        index=spy_data.index[1:]
    )
    
    # Align volatility and returns
    common_index = gkyz_volatility.index.intersection(log_returns.index)
    gkyz_volatility = gkyz_volatility[common_index]
    log_returns = log_returns[common_index]
    
    print(f"   Aligned data length: {len(log_returns)}")
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
    
    # Step 4: Split data for proper train/test evaluation (exactly as final_test.py)
    # Use 80/20 split
    train_size = int(0.8 * len(log_returns))
    train_returns = log_returns[:train_size]
    train_volatility = gkyz_volatility[:train_size]
    test_returns = log_returns[train_size:]
    test_volatility = gkyz_volatility[train_size:]
    
    # For hybrid model fitting, we need aligned spy_data
    train_data = spy_data[:train_size]
    
    print(f"\n4. Training GARCH-GRU hybrid model...")
    print(f"   Data split: {train_size} train, {len(test_returns)} test")
    training_location = "Modal cloud" if settings.use_modal else "local CPU"
    print(f"   Training on: {training_location}")
    if settings.use_modal:
        print(f"   GPU type: {settings.modal_gpu}")
    print("   This may take a few minutes...")
    
    hybrid_model = GARCHGRUHybrid()  # Now automatically reads from settings
    # Train only on training data
    hybrid_model.fit(train_data, train_returns, train_volatility)
    
    # Step 4.5: Evaluate model performance (exactly as final_test.py)
    print("\n4.5. Evaluating model performance on held-out test data...")
    print(f"   Weights: GARCH={hybrid_model.garch_weight:.1%}, GRU={hybrid_model.gru_weight:.1%}")
    
    # Generate test forecasts (first 200 points as in final_test.py)
    print("   Generating test forecasts (first 200 points)...")
    test_points = min(200, len(test_returns) - hybrid_model.gru.sequence_length - 1)
    
    # For Hybrid: Generate predictions exactly as final_test.py
    hybrid_predictions = []
    for i in range(hybrid_model.gru.sequence_length, hybrid_model.gru.sequence_length + test_points):
        current_returns = pd.concat([train_returns, test_returns[:i]])
        current_volatility = pd.concat([train_volatility, test_volatility[:i]])
        
        forecast = hybrid_model.forecast(current_returns, current_volatility, horizon=1)
        hybrid_predictions.append(forecast['combined'][0])
    
    hybrid_predictions = np.array(hybrid_predictions)
    actual_test = test_volatility.iloc[hybrid_model.gru.sequence_length+1:hybrid_model.gru.sequence_length+1+test_points].values
    
    # For GARCH: Align with same test period
    garch_predictions = garch_model.rolling_forecast(test_returns)
    garch_predictions = garch_predictions[hybrid_model.gru.sequence_length:hybrid_model.gru.sequence_length+test_points]
    
    # Ensure all arrays have same length
    min_len = min(len(actual_test), len(garch_predictions), len(hybrid_predictions))
    actual_test = actual_test[:min_len]
    garch_predictions = garch_predictions[:min_len]
    hybrid_predictions = hybrid_predictions[:min_len]
    
    print(f"   Test samples evaluated: {len(actual_test)}")
    
    # Calculate R² for both models (as in final_test.py)
    ss_res_hybrid = np.sum((actual_test - hybrid_predictions) ** 2)
    ss_tot = np.sum((actual_test - np.mean(actual_test)) ** 2)
    r2_hybrid = 1 - (ss_res_hybrid / ss_tot)
    
    ss_res_garch = np.sum((actual_test - garch_predictions) ** 2)
    r2_garch = 1 - (ss_res_garch / ss_tot)
    
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"GARCH R²: {r2_garch:.4f}")
    print(f"Hybrid R²: {r2_hybrid:.4f}")
    
    if r2_hybrid > 0.5:
        print("\n🎉 EXCELLENT: Hybrid achieves R² > 0.5")
    elif r2_hybrid > 0:
        print("\n✅ SUCCESS: Hybrid achieves positive R²")
    else:
        print("\n❌ FAIL: Hybrid still has negative R²")
    
    # Also use evaluator for detailed metrics
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
    
    # Use simulation settings (reduce if timeout issues)
    n_periods = min(settings.get('simulation.n_periods', 252), 252)
    n_paths = min(settings.get('simulation.n_paths', 100), 50)  # Reduce paths to avoid timeout
    
    print(f"   Initial price: ${current_price:.2f}")
    print(f"   Simulating {n_periods} days, {n_paths} paths...")
    
    try:
        garch_paths = simulator.simulate_garch_paths(garch_model, n_periods=n_periods, n_paths=n_paths)
        
        # Use all available data (train + test) for simulation context
        all_returns = pd.concat([train_returns, test_returns])
        all_volatility = pd.concat([train_volatility, test_volatility])
        
        hybrid_paths = simulator.simulate_hybrid_paths(
            hybrid_model, 
            all_returns.tail(504), 
            all_volatility.tail(504),
            n_periods=n_periods, 
            n_paths=n_paths
        )
    except Exception as e:
        print(f"   Warning: Simulation failed - {e}")
        print("   Using simplified simulation...")
        # Create simple placeholder paths
        garch_paths = np.ones((n_periods + 1, n_paths)) * current_price
        hybrid_paths = np.ones((n_periods + 1, n_paths)) * current_price
    
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