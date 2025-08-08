# src/diagnostics/simulation_validator.py
# Simulation quality assessment and validation for Monte Carlo paths
# Validates distributional properties, stylized facts, and path realism
# RELEVANT FILES: simulator.py, statistical_tests.py, report_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp, jarque_bera
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class SimulationValidationResults:
    """Container for simulation validation results"""
    # Distributional tests
    return_distribution_tests: Dict[str, float]  # test_name -> p_value
    moment_matching: Dict[str, Dict[str, float]]  # moment -> {simulated, historical, difference}
    distributional_similarity: Dict[str, float]  # test -> p_value
    
    # Stylized facts validation
    fat_tails_test: Dict[str, float]
    volatility_clustering_test: Dict[str, float]
    leverage_effect_test: Dict[str, float]
    autocorr_tests: Dict[str, float]
    
    # Path realism checks
    jump_detection: Dict[str, Union[int, float]]
    trend_analysis: Dict[str, float]
    path_statistics: Dict[str, float]
    extreme_event_frequency: Dict[str, float]
    
    # Monte Carlo diagnostics
    convergence_analysis: Dict[str, float]
    sample_size_adequacy: Dict[str, bool]
    simulation_stability: Dict[str, float]
    
    # Cross-sectional analysis
    path_diversity: float
    correlation_structure: Dict[str, float]
    
    # Overall assessment
    realism_score: float
    simulation_quality_score: float
    passed_validation: bool
    
    # Alerts
    red_flags: List[str]
    warnings: List[str]
    recommendations: List[str]

class SimulationValidator:
    """Comprehensive simulation quality assessment and validation system"""
    
    def __init__(self, alpha: float = 0.05, min_sample_size: int = 100):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
    
    def validate_simulation(self, simulated_paths: np.ndarray, 
                           historical_returns: np.ndarray,
                           model_name: str = "Model") -> SimulationValidationResults:
        """Comprehensive validation of simulated paths against historical data"""
        
        # Convert paths to returns
        simulated_returns = self._paths_to_returns(simulated_paths)
        
        # Distributional tests
        dist_tests = self._test_return_distributions(simulated_returns, historical_returns)
        moment_tests = self._test_moment_matching(simulated_returns, historical_returns)
        similarity_tests = self._test_distributional_similarity(simulated_returns, historical_returns)
        
        # Stylized facts
        fat_tails = self._test_fat_tails(simulated_returns, historical_returns)
        vol_clustering = self._test_volatility_clustering(simulated_returns, historical_returns)
        leverage = self._test_leverage_effect(simulated_returns)
        autocorr = self._test_return_autocorrelation(simulated_returns, historical_returns)
        
        # Path realism
        jumps = self._detect_jumps(simulated_returns)
        trends = self._analyze_trends(simulated_paths)
        path_stats = self._calculate_path_statistics(simulated_paths)
        extremes = self._analyze_extreme_events(simulated_returns, historical_returns)
        
        # Monte Carlo diagnostics
        convergence = self._test_monte_carlo_convergence(simulated_paths)
        adequacy = self._test_sample_size_adequacy(simulated_paths)
        stability = self._test_simulation_stability(simulated_paths)
        
        # Cross-sectional analysis
        diversity = self._calculate_path_diversity(simulated_paths)
        correlation = self._analyze_correlation_structure(simulated_returns)
        
        # Generate alerts and scores
        red_flags, warnings_list, recommendations = self._generate_alerts(
            dist_tests, fat_tails, vol_clustering, jumps, convergence
        )
        
        realism_score = self._calculate_realism_score(
            moment_tests, fat_tails, vol_clustering, jumps, trends
        )
        
        quality_score = self._calculate_quality_score(
            dist_tests, convergence, stability, red_flags
        )
        
        passed = len(red_flags) == 0 and realism_score > 0.6 and quality_score > 0.6
        
        return SimulationValidationResults(
            return_distribution_tests=dist_tests,
            moment_matching=moment_tests,
            distributional_similarity=similarity_tests,
            
            fat_tails_test=fat_tails,
            volatility_clustering_test=vol_clustering,
            leverage_effect_test=leverage,
            autocorr_tests=autocorr,
            
            jump_detection=jumps,
            trend_analysis=trends,
            path_statistics=path_stats,
            extreme_event_frequency=extremes,
            
            convergence_analysis=convergence,
            sample_size_adequacy=adequacy,
            simulation_stability=stability,
            
            path_diversity=diversity,
            correlation_structure=correlation,
            
            realism_score=realism_score,
            simulation_quality_score=quality_score,
            passed_validation=passed,
            
            red_flags=red_flags,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _paths_to_returns(self, paths: np.ndarray) -> np.ndarray:
        """Convert price paths to returns"""
        if len(paths.shape) != 2:
            raise ValueError("Expected 2D array: (time_periods, n_paths)")
        
        # Calculate log returns
        returns = np.diff(np.log(paths), axis=0)
        return returns
    
    def _test_return_distributions(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Test if simulated returns match historical distribution"""
        tests = {}
        
        # Flatten simulated returns across all paths
        sim_flat = simulated.flatten()
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(historical, sim_flat)
            tests['kolmogorov_smirnov'] = ks_pval
        except:
            tests['kolmogorov_smirnov'] = np.nan
        
        try:
            # Anderson-Darling test (approximate)
            ad_result = anderson_ksamp([historical, sim_flat])
            tests['anderson_darling'] = ad_result.significance_level / 100.0
        except:
            tests['anderson_darling'] = np.nan
        
        try:
            # Mann-Whitney U test
            u_stat, u_pval = stats.mannwhitneyu(historical, sim_flat, alternative='two-sided')
            tests['mann_whitney'] = u_pval
        except:
            tests['mann_whitney'] = np.nan
        
        return tests
    
    def _test_moment_matching(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Test matching of statistical moments"""
        sim_flat = simulated.flatten()
        
        moments = {}
        
        # Mean
        hist_mean = np.mean(historical)
        sim_mean = np.mean(sim_flat)
        moments['mean'] = {
            'historical': hist_mean,
            'simulated': sim_mean,
            'difference': abs(sim_mean - hist_mean),
            'relative_error': abs(sim_mean - hist_mean) / abs(hist_mean) if hist_mean != 0 else np.inf
        }
        
        # Standard deviation
        hist_std = np.std(historical)
        sim_std = np.std(sim_flat)
        moments['std'] = {
            'historical': hist_std,
            'simulated': sim_std,
            'difference': abs(sim_std - hist_std),
            'relative_error': abs(sim_std - hist_std) / hist_std if hist_std != 0 else np.inf
        }
        
        # Skewness
        hist_skew = stats.skew(historical)
        sim_skew = stats.skew(sim_flat)
        moments['skewness'] = {
            'historical': hist_skew,
            'simulated': sim_skew,
            'difference': abs(sim_skew - hist_skew),
            'relative_error': abs(sim_skew - hist_skew) / abs(hist_skew) if hist_skew != 0 else np.inf
        }
        
        # Kurtosis
        hist_kurt = stats.kurtosis(historical)
        sim_kurt = stats.kurtosis(sim_flat)
        moments['kurtosis'] = {
            'historical': hist_kurt,
            'simulated': sim_kurt,
            'difference': abs(sim_kurt - hist_kurt),
            'relative_error': abs(sim_kurt - hist_kurt) / abs(hist_kurt) if hist_kurt != 0 else np.inf
        }
        
        return moments
    
    def _test_distributional_similarity(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Additional distributional similarity tests"""
        tests = {}
        sim_flat = simulated.flatten()
        
        try:
            # Quantile-quantile correlation
            n_quantiles = min(100, len(historical) // 10)
            hist_quantiles = np.percentile(historical, np.linspace(0, 100, n_quantiles))
            sim_quantiles = np.percentile(sim_flat, np.linspace(0, 100, n_quantiles))
            qq_corr, qq_pval = stats.pearsonr(hist_quantiles, sim_quantiles)
            tests['qq_correlation'] = qq_pval  # p-value for correlation = 1
        except:
            tests['qq_correlation'] = np.nan
        
        return tests
    
    def _test_fat_tails(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Test for fat tails (excess kurtosis)"""
        tests = {}
        
        try:
            # Historical kurtosis
            hist_kurtosis = stats.kurtosis(historical)
            
            # Simulated kurtosis - check each path and aggregate data
            sim_kurtoses = []
            paths_with_fat_tails = 0
            
            for i in range(simulated.shape[1]):
                path_returns = simulated[:, i]
                if len(path_returns) > 10:  # Minimum for kurtosis
                    path_kurt = stats.kurtosis(path_returns)
                    sim_kurtoses.append(path_kurt)
                    if path_kurt > 0:  # Excess kurtosis present
                        paths_with_fat_tails += 1
            
            if sim_kurtoses:
                # Use median kurtosis and also pool all returns
                median_sim_kurtosis = np.median(sim_kurtoses)
                pooled_returns = simulated.flatten()
                pooled_kurtosis = stats.kurtosis(pooled_returns)
                
                tests['kurtosis_difference'] = abs(pooled_kurtosis - hist_kurtosis)
                tests['historical_kurtosis'] = hist_kurtosis
                tests['simulated_kurtosis'] = pooled_kurtosis  # Use pooled data
                tests['median_path_kurtosis'] = median_sim_kurtosis
                tests['proportion_paths_with_fat_tails'] = paths_with_fat_tails / len(sim_kurtoses)
                
                # Test if both show excess kurtosis (considering pooled data)
                tests['both_fat_tails'] = float(hist_kurtosis > 0 and pooled_kurtosis > 0)
        except:
            pass
        
        return tests
    
    def _test_volatility_clustering(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Test for volatility clustering patterns"""
        tests = {}
        
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Historical volatility clustering
            hist_abs_returns = np.abs(historical)
            hist_lb = acorr_ljungbox(hist_abs_returns, lags=10, return_df=True)
            hist_clustering_pval = hist_lb['lb_pvalue'].iloc[-1]
            
            # Simulated volatility clustering - test each path individually
            sim_clustering_pvals = []
            paths_with_clustering = 0
            n_paths_tested = min(simulated.shape[1], 10)  # Test up to 10 paths for efficiency
            
            for i in range(n_paths_tested):
                path_returns = simulated[:, i]
                if len(path_returns) > 20:
                    abs_returns = np.abs(path_returns)
                    try:
                        lb_result = acorr_ljungbox(abs_returns, lags=min(10, len(abs_returns)//3), return_df=True)
                        p_val = lb_result['lb_pvalue'].iloc[-1]
                        sim_clustering_pvals.append(p_val)
                        if p_val < 0.05:  # Path shows significant clustering
                            paths_with_clustering += 1
                    except:
                        continue
            
            if sim_clustering_pvals:
                # Report the proportion of paths showing clustering (correct approach)
                proportion_with_clustering = paths_with_clustering / len(sim_clustering_pvals)
                median_pval = np.median(sim_clustering_pvals)  # More robust than mean
                
                tests['historical_clustering_pval'] = hist_clustering_pval
                tests['simulated_clustering_pval'] = median_pval  # Use median instead of mean
                tests['proportion_paths_with_clustering'] = proportion_with_clustering
                
                # Consider clustering present if >40% of paths show it (realistic threshold)
                hist_has_clustering = hist_clustering_pval < 0.05
                sim_has_clustering = proportion_with_clustering > 0.4
                tests['both_show_clustering'] = float(hist_has_clustering and sim_has_clustering)
        
        except ImportError:
            pass
        except:
            pass
        
        return tests
    
    def _test_leverage_effect(self, simulated: np.ndarray) -> Dict[str, float]:
        """Test for leverage effect (negative correlation between returns and future volatility)"""
        tests = {}
        
        try:
            leverage_corrs = []
            for i in range(simulated.shape[1]):
                path_returns = simulated[:, i]
                if len(path_returns) > 10:
                    # Calculate rolling volatility
                    abs_returns = np.abs(path_returns)
                    if len(abs_returns) > 5:
                        # Correlation between returns and next-period absolute returns
                        returns_lag = path_returns[:-1]
                        vol_lead = abs_returns[1:]
                        if len(returns_lag) > 5:
                            corr, pval = stats.pearsonr(returns_lag, vol_lead)
                            leverage_corrs.append(corr)
            
            if leverage_corrs:
                avg_leverage = np.mean(leverage_corrs)
                tests['average_leverage_correlation'] = avg_leverage
                tests['leverage_effect_present'] = float(avg_leverage < -0.05)  # Negative correlation
        
        except:
            pass
        
        return tests
    
    def _test_return_autocorrelation(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Test for return autocorrelation patterns"""
        tests = {}
        
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Historical autocorrelation
            hist_lb = acorr_ljungbox(historical, lags=10, return_df=True)
            hist_autocorr_pval = hist_lb['lb_pvalue'].iloc[-1]
            
            # Simulated autocorrelation - test each path and use median
            sim_autocorr_pvals = []
            paths_with_no_autocorr = 0
            n_paths_tested = min(simulated.shape[1], 10)
            
            for i in range(n_paths_tested):
                path_returns = simulated[:, i]
                if len(path_returns) > 20:
                    try:
                        lb_result = acorr_ljungbox(path_returns, lags=min(10, len(path_returns)//3), return_df=True)
                        p_val = lb_result['lb_pvalue'].iloc[-1]
                        sim_autocorr_pvals.append(p_val)
                        if p_val > 0.05:  # No significant autocorrelation
                            paths_with_no_autocorr += 1
                    except:
                        continue
            
            if sim_autocorr_pvals:
                median_sim_autocorr = np.median(sim_autocorr_pvals)  # More robust than mean
                proportion_no_autocorr = paths_with_no_autocorr / len(sim_autocorr_pvals)
                
                tests['historical_autocorr_pval'] = hist_autocorr_pval
                tests['simulated_autocorr_pval'] = median_sim_autocorr
                tests['proportion_paths_no_autocorr'] = proportion_no_autocorr
                
                # Returns should generally not be autocorrelated (high p-values)
                # Consider acceptable if >60% of paths show no autocorrelation
                tests['both_no_autocorr'] = float(hist_autocorr_pval > 0.05 and proportion_no_autocorr > 0.6)
        
        except ImportError:
            pass
        except:
            pass
        
        return tests
    
    def _detect_jumps(self, simulated: np.ndarray, threshold_multiplier: float = 5.0) -> Dict[str, Union[int, float]]:
        """Detect unrealistic jumps in simulated paths"""
        sim_flat = simulated.flatten()
        std_return = np.std(sim_flat)
        threshold = threshold_multiplier * std_return
        
        jumps = np.abs(sim_flat) > threshold
        jump_count = np.sum(jumps)
        jump_frequency = jump_count / len(sim_flat)
        
        # Max jump size
        max_jump = np.max(np.abs(sim_flat)) if len(sim_flat) > 0 else 0
        
        return {
            'jump_count': int(jump_count),
            'jump_frequency': jump_frequency,
            'max_jump_size': max_jump,
            'jump_threshold': threshold,
            'max_jump_in_std_units': max_jump / std_return if std_return > 0 else 0
        }
    
    def _analyze_trends(self, simulated_paths: np.ndarray) -> Dict[str, float]:
        """Analyze unrealistic trends in price paths"""
        n_periods, n_paths = simulated_paths.shape
        
        trends = []
        for i in range(n_paths):
            path = simulated_paths[:, i]
            # Linear trend coefficient
            time_points = np.arange(len(path))
            slope, _, r_value, _, _ = stats.linregress(time_points, path)
            trends.append(r_value ** 2)  # R-squared for trend strength
        
        return {
            'average_trend_strength': np.mean(trends),
            'max_trend_strength': np.max(trends),
            'paths_with_strong_trend': np.sum(np.array(trends) > 0.5),  # R² > 0.5
            'trend_frequency': np.sum(np.array(trends) > 0.5) / n_paths
        }
    
    def _calculate_path_statistics(self, simulated_paths: np.ndarray) -> Dict[str, float]:
        """Calculate basic path statistics"""
        final_prices = simulated_paths[-1, :]
        initial_price = simulated_paths[0, 0]  # Assume same starting price
        
        total_returns = (final_prices / initial_price) - 1
        
        return {
            'mean_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'min_final_price': np.min(final_prices),
            'max_final_price': np.max(final_prices),
            'price_range_ratio': np.max(final_prices) / np.min(final_prices) if np.min(final_prices) > 0 else np.inf,
            'paths_ending_positive': np.sum(total_returns > 0) / len(total_returns)
        }
    
    def _analyze_extreme_events(self, simulated: np.ndarray, historical: np.ndarray) -> Dict[str, float]:
        """Analyze extreme event frequency"""
        sim_flat = simulated.flatten()
        
        # Define extreme events as beyond 2.5 standard deviations
        hist_std = np.std(historical)
        threshold = 2.5 * hist_std
        
        # Historical extreme frequency
        hist_extremes = np.sum(np.abs(historical) > threshold) / len(historical)
        
        # Simulated extreme frequency
        sim_extremes = np.sum(np.abs(sim_flat) > threshold) / len(sim_flat)
        
        return {
            'historical_extreme_freq': hist_extremes,
            'simulated_extreme_freq': sim_extremes,
            'extreme_freq_ratio': sim_extremes / hist_extremes if hist_extremes > 0 else np.inf,
            'threshold_used': threshold
        }
    
    def _test_monte_carlo_convergence(self, simulated_paths: np.ndarray) -> Dict[str, float]:
        """Test Monte Carlo convergence"""
        n_periods, n_paths = simulated_paths.shape
        
        # Test convergence of mean final price
        final_prices = simulated_paths[-1, :]
        
        # Calculate running means
        running_means = []
        for i in range(10, n_paths, max(1, n_paths // 20)):
            running_means.append(np.mean(final_prices[:i]))
        
        # Measure stability of running mean
        if len(running_means) > 5:
            recent_variation = np.std(running_means[-5:]) / np.mean(running_means[-5:])
        else:
            recent_variation = np.nan
        
        # Standard error of mean
        sem = np.std(final_prices) / np.sqrt(n_paths)
        mean_price = np.mean(final_prices)
        relative_sem = sem / mean_price if mean_price > 0 else np.inf
        
        return {
            'running_mean_variation': recent_variation,
            'standard_error_of_mean': sem,
            'relative_standard_error': relative_sem,
            'convergence_quality': 1 / (1 + recent_variation) if not np.isnan(recent_variation) else 0.5
        }
    
    def _test_sample_size_adequacy(self, simulated_paths: np.ndarray) -> Dict[str, bool]:
        """Test if sample size is adequate"""
        n_periods, n_paths = simulated_paths.shape
        
        return {
            'adequate_paths': n_paths >= self.min_sample_size,
            'adequate_periods': n_periods >= 50,
            'overall_adequate': n_paths >= self.min_sample_size and n_periods >= 50
        }
    
    def _test_simulation_stability(self, simulated_paths: np.ndarray) -> Dict[str, float]:
        """Test simulation numerical stability"""
        stability_metrics = {}
        
        # Check for NaN or infinite values
        has_nan = np.any(np.isnan(simulated_paths))
        has_inf = np.any(np.isinf(simulated_paths))
        
        stability_metrics['has_numerical_issues'] = float(has_nan or has_inf)
        
        # Check for unrealistic price levels
        min_price = np.min(simulated_paths)
        max_price = np.max(simulated_paths)
        
        stability_metrics['min_price'] = min_price
        stability_metrics['max_price'] = max_price
        stability_metrics['extreme_prices'] = float(min_price <= 0 or max_price > 1000 * simulated_paths[0, 0])
        
        return stability_metrics
    
    def _calculate_path_diversity(self, simulated_paths: np.ndarray) -> float:
        """Calculate diversity among simulated paths"""
        final_prices = simulated_paths[-1, :]
        
        # Coefficient of variation as diversity measure
        cv = np.std(final_prices) / np.mean(final_prices) if np.mean(final_prices) > 0 else 0
        
        return cv
    
    def _analyze_correlation_structure(self, simulated: np.ndarray) -> Dict[str, float]:
        """Analyze correlation structure between paths"""
        n_periods, n_paths = simulated.shape
        
        if n_paths < 2:
            return {'insufficient_paths': True}
        
        # Calculate pairwise correlations between paths (sample subset for efficiency)
        max_pairs = min(10, n_paths)
        correlations = []
        
        for i in range(max_pairs):
            for j in range(i + 1, max_pairs):
                if i < n_paths and j < n_paths:
                    corr, _ = stats.pearsonr(simulated[:, i], simulated[:, j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
        else:
            avg_correlation = 0
            max_correlation = 0
        
        return {
            'average_path_correlation': avg_correlation,
            'max_path_correlation': max_correlation,
            'paths_too_correlated': float(max_correlation > 0.5)
        }
    
    def _generate_alerts(self, dist_tests: Dict, fat_tails: Dict, vol_clustering: Dict, 
                        jumps: Dict, convergence: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Generate alerts and recommendations"""
        red_flags = []
        warnings_list = []
        recommendations = []
        
        # Distribution test failures
        failed_dist_tests = sum(1 for p in dist_tests.values() if not np.isnan(p) and p < self.alpha)
        if failed_dist_tests >= 2:
            red_flags.append("Multiple distributional tests failed - simulated returns don't match historical")
        elif failed_dist_tests == 1:
            warnings_list.append("One distributional test failed")
        
        # Jump detection
        if jumps.get('jump_frequency', 0) > 0.01:  # More than 1% jumps
            warnings_list.append(f"High jump frequency: {jumps['jump_frequency']:.2%}")
        
        if jumps.get('max_jump_in_std_units', 0) > 10:
            red_flags.append(f"Extremely large jump detected: {jumps['max_jump_in_std_units']:.1f} standard deviations")
        
        # Convergence issues
        conv_quality = convergence.get('convergence_quality', 1)
        if conv_quality < 0.5:
            warnings_list.append("Poor Monte Carlo convergence")
        
        rel_se = convergence.get('relative_standard_error', 0)
        if rel_se > 0.1:
            warnings_list.append("High standard error - may need more simulation paths")
        
        # Fat tails
        if fat_tails.get('both_fat_tails', 0) == 0:
            warnings_list.append("Simulated returns may not exhibit appropriate fat tails")
        
        # Volatility clustering
        if vol_clustering.get('both_show_clustering', 0) == 0:
            proportion = vol_clustering.get('proportion_paths_with_clustering', 0)
            if proportion < 0.2:
                warnings_list.append(f"Only {proportion:.0%} of paths exhibit volatility clustering")
            else:
                warnings_list.append(f"Moderate clustering present ({proportion:.0%} of paths)")
        
        # Recommendations
        if len(warnings_list) > 3:
            recommendations.append("Consider reviewing model calibration or increasing sample size")
        
        if rel_se > 0.05:
            recommendations.append("Increase number of simulation paths for better precision")
        
        return red_flags, warnings_list, recommendations
    
    def _calculate_realism_score(self, moments: Dict, fat_tails: Dict, vol_clustering: Dict,
                               jumps: Dict, trends: Dict) -> float:
        """Calculate simulation realism score"""
        score = 1.0
        
        # Moment matching penalties
        for moment, data in moments.items():
            rel_error = data.get('relative_error', 0)
            if not np.isfinite(rel_error):
                continue
            
            if rel_error > 0.5:
                score -= 0.15
            elif rel_error > 0.2:
                score -= 0.05
        
        # Stylized facts
        if fat_tails.get('both_fat_tails', 0) == 0:
            score -= 0.1
        
        if vol_clustering.get('both_show_clustering', 0) == 0:
            score -= 0.1
        
        # Unrealistic features
        if jumps.get('jump_frequency', 0) > 0.05:
            score -= 0.2
        
        if trends.get('trend_frequency', 0) > 0.1:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_quality_score(self, dist_tests: Dict, convergence: Dict, 
                               stability: Dict, red_flags: List[str]) -> float:
        """Calculate overall simulation quality score"""
        score = 1.0
        
        # Red flags penalty
        score -= 0.3 * len(red_flags)
        
        # Distribution test results
        passed_tests = sum(1 for p in dist_tests.values() if not np.isnan(p) and p >= self.alpha)
        total_tests = sum(1 for p in dist_tests.values() if not np.isnan(p))
        if total_tests > 0:
            test_pass_rate = passed_tests / total_tests
            score += 0.2 * test_pass_rate
        
        # Convergence quality
        conv_quality = convergence.get('convergence_quality', 0.5)
        score += 0.1 * conv_quality
        
        # Stability issues
        if stability.get('has_numerical_issues', 0) == 1:
            score -= 0.3
        
        if stability.get('extreme_prices', 0) == 1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def create_diagnostic_plots(self, validation_results: SimulationValidationResults,
                               simulated_paths: np.ndarray, historical_returns: np.ndarray,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive simulation validation plots"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Convert paths to returns for plotting
        simulated_returns = self._paths_to_returns(simulated_paths)
        sim_flat = simulated_returns.flatten()
        
        # Return distribution comparison
        ax = axes[0, 0]
        ax.hist(historical_returns, bins=50, alpha=0.7, density=True, label='Historical', color='blue')
        ax.hist(sim_flat, bins=50, alpha=0.7, density=True, label='Simulated', color='red')
        ax.set_title('Return Distributions')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[0, 1]
        n_quantiles = min(100, len(historical_returns))
        hist_quantiles = np.percentile(historical_returns, np.linspace(0, 100, n_quantiles))
        sim_quantiles = np.percentile(sim_flat, np.linspace(0, 100, n_quantiles))
        ax.scatter(hist_quantiles, sim_quantiles, alpha=0.6)
        min_q, max_q = min(hist_quantiles.min(), sim_quantiles.min()), max(hist_quantiles.max(), sim_quantiles.max())
        ax.plot([min_q, max_q], [min_q, max_q], 'r--', alpha=0.5)
        ax.set_xlabel('Historical Quantiles')
        ax.set_ylabel('Simulated Quantiles')
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Sample paths
        ax = axes[0, 2]
        n_sample_paths = min(10, simulated_paths.shape[1])
        for i in range(n_sample_paths):
            ax.plot(simulated_paths[:, i], alpha=0.5, linewidth=0.8)
        ax.set_title(f'Sample Simulated Paths (n={n_sample_paths})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Moment comparison
        ax = axes[1, 0]
        moments = validation_results.moment_matching
        moment_names = list(moments.keys())
        hist_values = [moments[m]['historical'] for m in moment_names]
        sim_values = [moments[m]['simulated'] for m in moment_names]
        
        x = np.arange(len(moment_names))
        width = 0.35
        ax.bar(x - width/2, hist_values, width, label='Historical', alpha=0.7)
        ax.bar(x + width/2, sim_values, width, label='Simulated', alpha=0.7)
        ax.set_xlabel('Moments')
        ax.set_ylabel('Value')
        ax.set_title('Moment Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(moment_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Path statistics
        ax = axes[1, 1]
        path_stats = validation_results.path_statistics
        stats_text = []
        for key, value in path_stats.items():
            if isinstance(value, float):
                stats_text.append(f"{key}: {value:.4f}")
            else:
                stats_text.append(f"{key}: {value}")
        
        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Path Statistics')
        
        # Test results summary
        ax = axes[1, 2]
        test_summary = []
        test_summary.append("VALIDATION SUMMARY")
        test_summary.append("=" * 20)
        test_summary.append(f"Realism Score: {validation_results.realism_score:.3f}")
        test_summary.append(f"Quality Score: {validation_results.simulation_quality_score:.3f}")
        test_summary.append(f"Overall: {'PASS' if validation_results.passed_validation else 'FAIL'}")
        test_summary.append("")
        
        # Key metrics
        test_summary.append("KEY METRICS:")
        if validation_results.return_distribution_tests:
            for test, pval in validation_results.return_distribution_tests.items():
                if not np.isnan(pval):
                    status = "✅" if pval >= 0.05 else "❌"
                    test_summary.append(f"{status} {test}: {pval:.3f}")
        
        ax.text(0.05, 0.95, '\n'.join(test_summary), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Test Summary')
        
        # Jump analysis
        ax = axes[2, 0]
        jumps = validation_results.jump_detection
        ax.hist(np.abs(sim_flat), bins=50, alpha=0.7, density=True)
        if 'jump_threshold' in jumps:
            ax.axvline(x=jumps['jump_threshold'], color='r', linestyle='--', 
                      label=f"Jump Threshold ({jumps['jump_count']} jumps)")
        ax.set_xlabel('Absolute Return')
        ax.set_ylabel('Density')
        ax.set_title('Jump Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Convergence analysis
        ax = axes[2, 1]
        final_prices = simulated_paths[-1, :]
        running_means = []
        for i in range(10, len(final_prices), max(1, len(final_prices) // 50)):
            running_means.append(np.mean(final_prices[:i]))
        
        if running_means:
            ax.plot(range(10, 10 + len(running_means) * max(1, len(final_prices) // 50), 
                         max(1, len(final_prices) // 50)), running_means)
            ax.set_xlabel('Number of Paths')
            ax.set_ylabel('Running Mean Final Price')
            ax.set_title('Monte Carlo Convergence')
            ax.grid(True, alpha=0.3)
        
        # Alerts and recommendations
        ax = axes[2, 2]
        alerts_text = []
        
        if validation_results.red_flags:
            alerts_text.append("🚨 RED FLAGS:")
            for flag in validation_results.red_flags:
                alerts_text.append(f"  • {flag}")
        
        if validation_results.warnings:
            alerts_text.append("\n⚠️  WARNINGS:")
            for warning in validation_results.warnings:
                alerts_text.append(f"  • {warning}")
        
        if validation_results.recommendations:
            alerts_text.append("\n💡 RECOMMENDATIONS:")
            for rec in validation_results.recommendations:
                alerts_text.append(f"  • {rec}")
        
        if not alerts_text:
            alerts_text = ["✅ No issues detected"]
        
        ax.text(0.05, 0.95, '\n'.join(alerts_text), transform=ax.transAxes,
                fontsize=8, verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Alerts & Recommendations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig