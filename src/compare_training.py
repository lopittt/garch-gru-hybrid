# src/compare_training.py
# Comparison script to test local vs Modal training performance
# Measures execution time and provides detailed comparison
# RELEVANT FILES: hybrid.py, modal_gru.py, main.py

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_spy_data, get_log_returns
from data.preprocessor import calculate_gkyz_volatility
from models.hybrid import GARCHGRUHybrid

def test_local_training():
    """Test local training and measure execution time"""
    print("=" * 60)
    print("TESTING LOCAL TRAINING")
    print("=" * 60)
    
    # Prepare data
    print("Fetching data...")
    spy_data = fetch_spy_data(end_date='2024-12-31', years_back=3)  # Smaller dataset for testing
    log_returns = get_log_returns(spy_data)
    gkyz_volatility = calculate_gkyz_volatility(spy_data, window=10)
    
    print(f"Data shape: {spy_data.shape}")
    print(f"Returns length: {len(log_returns)}")
    print(f"Volatility length: {len(gkyz_volatility)}")
    
    # Create local hybrid model
    local_model = GARCHGRUHybrid(sequence_length=6, use_modal=False)
    
    # Measure total time including data preparation
    start_time = time.time()
    
    # Fit the model
    try:
        local_model.fit(spy_data, log_returns, gkyz_volatility)
        end_time = time.time()
        
        total_time = end_time - start_time
        training_time = local_model.get_training_time()
        
        print(f"✅ Local training completed successfully!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   GRU training time: {training_time:.2f} seconds")
        
        return {
            'success': True,
            'total_time': total_time,
            'training_time': training_time,
            'model': local_model
        }
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Local training failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'total_time': end_time - start_time
        }

def test_modal_training():
    """Test Modal training and measure execution time"""
    print("=" * 60)
    print("TESTING MODAL TRAINING")
    print("=" * 60)
    
    # Prepare data (same as local)
    print("Fetching data...")
    spy_data = fetch_spy_data(end_date='2024-12-31', years_back=3)
    log_returns = get_log_returns(spy_data)
    gkyz_volatility = calculate_gkyz_volatility(spy_data, window=10)
    
    print(f"Data shape: {spy_data.shape}")
    print(f"Returns length: {len(log_returns)}")
    print(f"Volatility length: {len(gkyz_volatility)}")
    
    # Create Modal hybrid model
    modal_model = GARCHGRUHybrid(sequence_length=6, use_modal=True)
    
    # Measure total time
    start_time = time.time()
    
    # Fit the model
    try:
        modal_model.fit(spy_data, log_returns, gkyz_volatility)
        end_time = time.time()
        
        total_time = end_time - start_time
        training_time = modal_model.get_training_time()
        
        print(f"✅ Modal training completed successfully!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   GRU training time: {training_time:.2f} seconds")
        print(f"   Note: Modal time includes cloud setup/teardown overhead")
        
        return {
            'success': True,
            'total_time': total_time,
            'training_time': training_time,
            'model': modal_model
        }
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Modal training failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'total_time': end_time - start_time
        }

def compare_results(local_result, modal_result):
    """Compare the results from local and Modal training"""
    print("=" * 60)
    print("TRAINING COMPARISON RESULTS")
    print("=" * 60)
    
    if local_result['success'] and modal_result['success']:
        # Successful comparison
        local_time = local_result['training_time']
        modal_time = modal_result['training_time']
        
        speedup = local_time / modal_time if modal_time > 0 else float('inf')
        time_saved = local_time - modal_time
        
        print(f"📊 PERFORMANCE COMPARISON:")
        print(f"   Local training time:  {local_time:.2f} seconds")
        print(f"   Modal training time:  {modal_time:.2f} seconds")
        print(f"   Time difference:      {time_saved:+.2f} seconds")
        
        if speedup > 1:
            print(f"   Modal speedup:        {speedup:.2f}x faster")
            print(f"   Performance gain:     {(speedup-1)*100:.1f}%")
        else:
            print(f"   Local speedup:        {1/speedup:.2f}x faster")
            print(f"   Modal overhead:       {(1-speedup)*100:.1f}%")
        
        print(f"\n📈 TOTAL EXECUTION TIME:")
        print(f"   Local total:          {local_result['total_time']:.2f} seconds")
        print(f"   Modal total:          {modal_result['total_time']:.2f} seconds")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if speedup > 1.2:
            print("   ✅ Modal provides significant speedup - recommended for training")
            print("   ✅ Use Modal for production training runs")
        elif speedup > 0.8:
            print("   ⚖️  Modal and local performance are comparable")
            print("   ⚖️  Choose based on resource availability and cost considerations")
        else:
            print("   ⚠️  Local training is faster due to Modal overhead")
            print("   ⚠️  Consider Modal only for larger datasets or longer training")
        
        return {
            'local_time': local_time,
            'modal_time': modal_time,
            'speedup': speedup,
            'recommendation': 'modal' if speedup > 1.2 else 'local' if speedup < 0.8 else 'either'
        }
    
    else:
        # Handle failures
        print("❌ COMPARISON INCOMPLETE:")
        if not local_result['success']:
            print(f"   Local training failed: {local_result['error']}")
        if not modal_result['success']:
            print(f"   Modal training failed: {modal_result['error']}")
        
        return {
            'comparison_failed': True,
            'local_success': local_result['success'],
            'modal_success': modal_result['success']
        }

def main():
    """Run the complete comparison"""
    print("🚀 GARCH-GRU Training Performance Comparison")
    print("    Local CPU vs Modal Cloud GPU")
    print()
    
    # Test local training
    local_result = test_local_training()
    print()
    
    # Test Modal training
    modal_result = test_modal_training()
    print()
    
    # Compare results
    comparison = compare_results(local_result, modal_result)
    
    return {
        'local': local_result,
        'modal': modal_result,
        'comparison': comparison
    }

if __name__ == "__main__":
    results = main()