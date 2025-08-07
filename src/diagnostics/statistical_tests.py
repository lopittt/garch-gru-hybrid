# src/diagnostics/statistical_tests.py
# Statistical validation and testing for GARCH-GRU model
# Implements residual analysis, significance testing, and model validation
# RELEVANT FILES: metrics.py, model_analyzer.py, report_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import jarque_bera, normaltest, kstest, anderson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatisticalTestResults:
    """Container for statistical test results"""
    # Residual analysis
    residual_normality: Dict[str, float]  # test_name -> p_value
    residual_autocorr: Dict[str, float]   # lag -> p_value
    residual_heteroscedasticity: Dict[str, float]  # test_name -> p_value
    residual_stationarity: Dict[str, float]  # test_name -> p_value
    
    # Model comparison
    forecast_accuracy_tests: Dict[str, Dict]  # test_name -> results
    model_significance: Dict[str, float]  # metric -> bootstrap_p_value
    
    # Volatility clustering
    volatility_clustering: Dict[str, float]  # test_name -> p_value
    arch_effects: Dict[str, float]  # lag -> p_value
    
    # Parameter stability
    parameter_stability: Dict[str, Dict]  # parameter -> stability_metrics
    structural_breaks: Dict[str, List[int]]  # test -> break_points
    
    # Overall assessment
    test_summary: Dict[str, str]  # test_category -> pass/fail/warning
    critical_failures: List[str]
    warnings: List[str]
    overall_validity: float  # 0-1 score

class StatisticalValidator:
    """Comprehensive statistical testing and validation system"""
    
    def __init__(self, alpha: float = 0.05, bootstrap_samples: int = 1000):
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
    
    def validate_model(self, actual: np.ndarray, predicted: np.ndarray,
                      residuals: Optional[np.ndarray] = None,
                      model_name: str = "Model") -> StatisticalTestResults:
        """Comprehensive statistical validation of model performance"""
        
        # Calculate residuals if not provided
        if residuals is None:
            residuals = actual - predicted
        
        # Residual analysis
        normality_tests = self._test_residual_normality(residuals)
        autocorr_tests = self._test_residual_autocorrelation(residuals)
        hetero_tests = self._test_heteroscedasticity(residuals)
        stationarity_tests = self._test_stationarity(residuals)
        
        # Volatility clustering tests
        clustering_tests = self._test_volatility_clustering(actual, residuals)
        arch_tests = self._test_arch_effects(residuals)
        
        # Model significance testing
        significance_tests = self._test_model_significance(actual, predicted)
        
        # Parameter stability (placeholder - would need time series of parameters)
        stability_tests = {}
        break_tests = {}
        
        # Generate summary and alerts
        test_summary, critical_failures, warnings_list = self._generate_test_summary(
            normality_tests, autocorr_tests, hetero_tests, clustering_tests, significance_tests
        )
        
        # Calculate overall validity score
        validity_score = self._calculate_validity_score(test_summary, critical_failures)
        
        return StatisticalTestResults(
            residual_normality=normality_tests,
            residual_autocorr=autocorr_tests,
            residual_heteroscedasticity=hetero_tests,
            residual_stationarity=stationarity_tests,
            forecast_accuracy_tests={},  # Placeholder for forecast comparison tests
            model_significance=significance_tests,
            volatility_clustering=clustering_tests,
            arch_effects=arch_tests,
            parameter_stability=stability_tests,
            structural_breaks=break_tests,
            test_summary=test_summary,
            critical_failures=critical_failures,
            warnings=warnings_list,
            overall_validity=validity_score
        )
    
    def _test_residual_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test residual normality using multiple methods"""
        tests = {}
        
        try:
            # Jarque-Bera test
            jb_stat, jb_pval = jarque_bera(residuals)
            tests['jarque_bera'] = jb_pval
        except:
            tests['jarque_bera'] = np.nan
        
        try:
            # D'Agostino's normality test
            stat, pval = normaltest(residuals)
            tests['dagostino'] = pval
        except:
            tests['dagostino'] = np.nan
        
        try:
            # Kolmogorov-Smirnov test against normal distribution
            std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            ks_stat, ks_pval = kstest(std_residuals, 'norm')
            tests['kolmogorov_smirnov'] = ks_pval
        except:
            tests['kolmogorov_smirnov'] = np.nan
        
        try:
            # Anderson-Darling test
            ad_result = anderson(residuals, dist='norm')
            # Convert to p-value approximation (rough)
            if ad_result.statistic > ad_result.critical_values[-1]:
                tests['anderson_darling'] = 0.01  # Very low p-value
            else:
                tests['anderson_darling'] = 0.1   # Higher p-value
        except:
            tests['anderson_darling'] = np.nan
        
        return tests
    
    def _test_residual_autocorrelation(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for residual autocorrelation"""
        tests = {}
        
        try:
            # Ljung-Box test for different lags
            for lag in [5, 10, 20]:
                if len(residuals) > lag * 2:
                    lb_result = acorr_ljungbox(residuals, lags=lag, return_df=True)
                    tests[f'ljung_box_lag_{lag}'] = lb_result['lb_pvalue'].iloc[-1]
        except:
            pass
        
        return tests
    
    def _test_heteroscedasticity(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for heteroscedasticity in residuals"""
        tests = {}
        
        try:
            # ARCH-LM test
            for lag in [5, 10]:
                if len(residuals) > lag * 3:
                    lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=lag)
                    tests[f'arch_lm_lag_{lag}'] = lm_pval
        except:
            pass
        
        try:
            # Breusch-Pagan test (simplified version)
            # Test if squared residuals correlate with fitted values
            squared_resid = residuals ** 2
            fitted_approx = np.arange(len(residuals))  # Proxy for fitted values
            
            correlation, p_val = stats.pearsonr(squared_resid, fitted_approx)
            tests['breusch_pagan_proxy'] = p_val
        except:
            tests['breusch_pagan_proxy'] = np.nan
        
        return tests
    
    def _test_stationarity(self, series: np.ndarray) -> Dict[str, float]:
        """Test for stationarity"""
        tests = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series, autolag='AIC')
            tests['adf'] = adf_result[1]  # p-value
        except:
            tests['adf'] = np.nan
        
        return tests
    
    def _test_volatility_clustering(self, actual: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """Test for volatility clustering in the data"""
        tests = {}
        
        try:
            # Test autocorrelation in absolute residuals
            abs_residuals = np.abs(residuals)
            lb_result = acorr_ljungbox(abs_residuals, lags=10, return_df=True)
            tests['abs_residual_clustering'] = lb_result['lb_pvalue'].iloc[-1]
        except:
            tests['abs_residual_clustering'] = np.nan
        
        try:
            # Test autocorrelation in squared residuals
            sq_residuals = residuals ** 2
            lb_result = acorr_ljungbox(sq_residuals, lags=10, return_df=True)
            tests['squared_residual_clustering'] = lb_result['lb_pvalue'].iloc[-1]
        except:
            tests['squared_residual_clustering'] = np.nan
        
        return tests
    
    def _test_arch_effects(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for ARCH effects at different lags"""
        tests = {}
        
        try:
            for lag in [1, 5, 10]:
                if len(residuals) > lag * 4:
                    lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=lag)
                    tests[f'arch_lag_{lag}'] = lm_pval
        except:
            pass
        
        return tests
    
    def _test_model_significance(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Test statistical significance of model predictions"""
        tests = {}
        
        # Bootstrap test for R²
        def calculate_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        try:
            # Original R²
            original_r2 = calculate_r2(actual, predicted)
            
            # Bootstrap null distribution (random predictions)
            bootstrap_r2s = []
            for _ in range(self.bootstrap_samples):
                random_pred = np.random.permutation(predicted)
                boot_r2 = calculate_r2(actual, random_pred)
                bootstrap_r2s.append(boot_r2)
            
            # P-value: fraction of bootstrap R²s >= original R²
            p_value = np.mean(np.array(bootstrap_r2s) >= original_r2)
            tests['r2_significance'] = p_value
        except:
            tests['r2_significance'] = np.nan
        
        # Test for prediction vs persistence baseline
        try:
            if len(actual) > 1:
                persistence_pred = np.roll(actual, 1)[1:]  # Yesterday's value
                actual_truncated = actual[1:]
                predicted_truncated = predicted[1:]
                
                persistence_mse = np.mean((actual_truncated - persistence_pred) ** 2)
                model_mse = np.mean((actual_truncated - predicted_truncated) ** 2)
                
                # Simple t-test for MSE difference
                persistence_errors = (actual_truncated - persistence_pred) ** 2
                model_errors = (actual_truncated - predicted_truncated) ** 2
                
                t_stat, p_val = stats.ttest_rel(persistence_errors, model_errors)
                tests['vs_persistence'] = p_val
        except:
            tests['vs_persistence'] = np.nan
        
        return tests
    
    def _generate_test_summary(self, normality: Dict, autocorr: Dict, hetero: Dict, 
                             clustering: Dict, significance: Dict) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Generate test summary with pass/fail/warning classifications"""
        
        summary = {}
        critical_failures = []
        warnings_list = []
        
        # Normality assessment
        normality_passes = sum(1 for p in normality.values() if not np.isnan(p) and p > self.alpha)
        normality_total = sum(1 for p in normality.values() if not np.isnan(p))
        
        if normality_total == 0:
            summary['residual_normality'] = 'unknown'
        elif normality_passes / normality_total >= 0.5:
            summary['residual_normality'] = 'pass'
        elif normality_passes / normality_total >= 0.25:
            summary['residual_normality'] = 'warning'
            warnings_list.append("Some residual normality tests failed")
        else:
            summary['residual_normality'] = 'fail'
            critical_failures.append("Residual normality severely violated")
        
        # Autocorrelation assessment
        autocorr_failures = sum(1 for p in autocorr.values() if not np.isnan(p) and p <= self.alpha)
        if autocorr_failures == 0:
            summary['autocorrelation'] = 'pass'
        elif autocorr_failures <= 1:
            summary['autocorrelation'] = 'warning'
            warnings_list.append("Some autocorrelation detected in residuals")
        else:
            summary['autocorrelation'] = 'fail'
            critical_failures.append("Significant autocorrelation in residuals")
        
        # Heteroscedasticity assessment
        hetero_failures = sum(1 for p in hetero.values() if not np.isnan(p) and p <= self.alpha)
        if hetero_failures == 0:
            summary['heteroscedasticity'] = 'pass'
        elif hetero_failures <= 1:
            summary['heteroscedasticity'] = 'warning'
            warnings_list.append("Some heteroscedasticity detected")
        else:
            summary['heteroscedasticity'] = 'fail'
            critical_failures.append("Significant heteroscedasticity detected")
        
        # Model significance
        r2_sig = significance.get('r2_significance', np.nan)
        if np.isnan(r2_sig):
            summary['model_significance'] = 'unknown'
        elif r2_sig <= self.alpha:
            summary['model_significance'] = 'pass'
        else:
            summary['model_significance'] = 'fail'
            critical_failures.append("Model predictions not significantly better than random")
        
        return summary, critical_failures, warnings_list
    
    def _calculate_validity_score(self, summary: Dict[str, str], critical_failures: List[str]) -> float:
        """Calculate overall statistical validity score"""
        score = 1.0
        
        # Heavy penalty for critical failures
        score -= 0.3 * len(critical_failures)
        
        # Penalties for test failures
        for test_result in summary.values():
            if test_result == 'fail':
                score -= 0.15
            elif test_result == 'warning':
                score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def create_diagnostic_plots(self, actual: np.ndarray, predicted: np.ndarray,
                               residuals: Optional[np.ndarray] = None,
                               test_results: Optional[StatisticalTestResults] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive statistical diagnostic plots"""
        
        if residuals is None:
            residuals = actual - predicted
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Residual time series
        ax = axes[0, 0]
        ax.plot(residuals, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Residual Time Series')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual')
        ax.grid(True, alpha=0.3)
        
        # Residual histogram with normal overlay
        ax = axes[0, 1]
        ax.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
        ax.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal')
        ax.set_title('Residual Distribution')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal)')
        ax.grid(True, alpha=0.3)
        
        # Actual vs Predicted scatter
        ax = axes[1, 0]
        ax.scatter(actual, predicted, alpha=0.6)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # Residual vs Fitted
        ax = axes[1, 1]
        ax.scatter(predicted, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')
        ax.grid(True, alpha=0.3)
        
        # Squared residuals (heteroscedasticity check)
        ax = axes[1, 2]
        squared_resid = residuals ** 2
        ax.plot(squared_resid, alpha=0.7)
        ax.set_title('Squared Residuals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Squared Residual')
        ax.grid(True, alpha=0.3)
        
        # ACF of residuals
        ax = axes[2, 0]
        try:
            from statsmodels.tsa.stattools import acf
            lags = min(20, len(residuals) // 4)
            acf_vals = acf(residuals, nlags=lags)
            ax.plot(range(lags + 1), acf_vals, 'b-o', markersize=4)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            # Add confidence bands
            n = len(residuals)
            ax.axhline(y=1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
            ax.set_title('ACF of Residuals')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'ACF calculation failed', ha='center', va='center', transform=ax.transAxes)
        
        # Test results summary
        ax = axes[2, 1]
        if test_results:
            test_text = []
            test_text.append("STATISTICAL TEST SUMMARY")
            test_text.append("=" * 25)
            
            for category, result in test_results.test_summary.items():
                status_symbol = "✅" if result == "pass" else "⚠️" if result == "warning" else "❌" if result == "fail" else "❓"
                test_text.append(f"{status_symbol} {category}: {result}")
            
            test_text.append("")
            test_text.append(f"Validity Score: {test_results.overall_validity:.3f}")
            
            if test_results.critical_failures:
                test_text.append("")
                test_text.append("CRITICAL FAILURES:")
                for failure in test_results.critical_failures:
                    test_text.append(f"  • {failure}")
            
            ax.text(0.05, 0.95, '\n'.join(test_text), transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No test results provided', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Test Results')
        
        # Detailed p-values
        ax = axes[2, 2]
        if test_results:
            p_val_text = []
            p_val_text.append("P-VALUES")
            p_val_text.append("=" * 15)
            
            # Normality tests
            p_val_text.append("\nNormality:")
            for test, pval in test_results.residual_normality.items():
                if not np.isnan(pval):
                    p_val_text.append(f"  {test}: {pval:.4f}")
            
            # Autocorrelation tests
            if test_results.residual_autocorr:
                p_val_text.append("\nAutocorrelation:")
                for test, pval in test_results.residual_autocorr.items():
                    if not np.isnan(pval):
                        p_val_text.append(f"  {test}: {pval:.4f}")
            
            # Model significance
            p_val_text.append("\nSignificance:")
            for test, pval in test_results.model_significance.items():
                if not np.isnan(pval):
                    p_val_text.append(f"  {test}: {pval:.4f}")
            
            ax.text(0.05, 0.95, '\n'.join(p_val_text), transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No test results provided', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Detailed P-values')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compare_models(self, actual: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Compare multiple models using statistical tests"""
        results = {}
        
        model_names = list(predictions_dict.keys())
        
        # Diebold-Mariano test for each pair
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pred1 = predictions_dict[model1]
                pred2 = predictions_dict[model2]
                
                # Calculate loss differences
                loss1 = (actual - pred1) ** 2
                loss2 = (actual - pred2) ** 2
                loss_diff = loss1 - loss2
                
                # Simple t-test as proxy for DM test
                if len(loss_diff) > 1 and np.std(loss_diff) > 0:
                    t_stat, p_val = stats.ttest_1samp(loss_diff, 0)
                    results[f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'interpretation': 'model1_better' if t_stat < 0 else 'model2_better'
                    }
        
        return results