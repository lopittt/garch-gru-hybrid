# src/diagnostics/model_analyzer.py
# Model behavior analysis for GARCH-GRU components
# Analyzes parameter validity, architecture performance, and model interpretability
# RELEVANT FILES: garch.py, gru.py, hybrid.py, report_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

@dataclass
class ModelAnalysisResults:
    """Container for model behavior analysis results"""
    # GARCH analysis
    garch_parameters: Dict[str, float]
    garch_validity: Dict[str, bool]
    garch_persistence: float
    garch_half_life: float
    garch_unconditional_vol: float
    
    # GRU analysis
    gru_architecture: Dict[str, Any]
    gru_parameter_stats: Dict[str, Dict[str, float]]  # layer -> {mean, std, etc}
    gru_activation_patterns: Optional[Dict[str, np.ndarray]]
    gru_gradient_analysis: Optional[Dict[str, float]]
    
    # Hybrid analysis
    hybrid_weights: Optional[Dict[str, float]]
    hybrid_weight_stability: Optional[float]
    hybrid_contribution_analysis: Optional[Dict[str, float]]
    
    # Feature importance
    feature_importance: Dict[str, float]
    sequence_importance: Optional[np.ndarray]  # importance by sequence position
    
    # Model diagnostics
    parameter_health: Dict[str, str]  # component -> health_status
    calibration_metrics: Dict[str, float]
    model_complexity: Dict[str, float]
    
    # Alerts
    red_flags: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Overall assessment
    model_health_score: float
    interpretability_score: float

class ModelAnalyzer:
    """Comprehensive model behavior analysis and interpretation system"""
    
    def __init__(self, sensitivity_samples: int = 100):
        self.sensitivity_samples = sensitivity_samples
    
    def analyze_garch_model(self, garch_model, returns: pd.Series) -> Dict:
        """Analyze GARCH model parameters and behavior"""
        params = garch_model.params
        
        # Parameter validity checks
        validity = {
            'omega_positive': params['omega'] > 0,
            'alpha_valid': 0 < params['alpha'] < 1,
            'beta_valid': 0 < params['beta'] < 1,
            'persistence_valid': params['alpha'] + params['beta'] < 1,
            'stationarity': params['alpha'] + params['beta'] < 0.99
        }
        
        # Calculate persistence and half-life
        persistence = params['alpha'] + params['beta']
        half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else np.inf
        
        # Unconditional volatility
        unconditional_vol = np.sqrt(params['omega'] / (1 - persistence)) if persistence < 1 else np.inf
        
        # Compare with sample volatility
        sample_vol = returns.std()
        
        return {
            'parameters': params,
            'validity': validity,
            'persistence': persistence,
            'half_life': half_life,
            'unconditional_vol': unconditional_vol,
            'sample_vol': sample_vol,
            'vol_ratio': unconditional_vol / sample_vol if sample_vol > 0 else np.inf
        }
    
    def analyze_gru_model(self, gru_model, sample_input: Optional[np.ndarray] = None) -> Dict:
        """Analyze GRU model architecture and parameters"""
        model = gru_model.model
        
        # Architecture information
        architecture = {
            'input_size': model.gru1.input_size,
            'hidden_sizes': [model.gru1.hidden_size, model.gru2.hidden_size, model.gru3.hidden_size],
            'output_size': model.fc.out_features,
            'dropout_rate': model.dropout1.p,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Parameter statistics for each layer
        param_stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weights = module.weight.data.cpu().numpy()
                param_stats[name] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'norm': float(np.linalg.norm(weights)),
                    'sparsity': float(np.mean(np.abs(weights) < 1e-6))
                }
        
        # Gradient analysis (if available)
        gradient_stats = {}
        if any(p.grad is not None for p in model.parameters()):
            total_norm = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    gradient_stats[name] = float(grad_norm)
                    total_norm += grad_norm.item() ** 2
                    param_count += 1
            
            gradient_stats['total_norm'] = np.sqrt(total_norm)
            gradient_stats['avg_norm'] = np.sqrt(total_norm) / param_count if param_count > 0 else 0
        
        # Activation patterns (if sample input provided)
        activation_patterns = None
        if sample_input is not None:
            try:
                activation_patterns = self._analyze_activations(model, sample_input)
            except Exception as e:
                warnings.warn(f"Could not analyze activations: {e}")
        
        return {
            'architecture': architecture,
            'parameter_stats': param_stats,
            'gradient_stats': gradient_stats,
            'activation_patterns': activation_patterns
        }
    
    def analyze_hybrid_model(self, hybrid_model, sample_returns: pd.Series, 
                           sample_volatility: pd.Series) -> Dict:
        """Analyze hybrid model behavior and weight patterns"""
        analysis = {}
        
        # Weight analysis
        if hasattr(hybrid_model, 'garch_weight') and hasattr(hybrid_model, 'gru_weight'):
            weights = {
                'garch': hybrid_model.garch_weight,
                'gru': hybrid_model.gru_weight
            }
            
            # Weight stability analysis
            if hasattr(hybrid_model, 'weight_optimization_history'):
                history = hybrid_model.weight_optimization_history
                if history:
                    weight_stability = 1 - np.std([w['garch'] for w in history]) if len(history) > 1 else 1.0
                else:
                    weight_stability = 1.0
            else:
                weight_stability = None
            
            analysis['weights'] = weights
            analysis['weight_stability'] = weight_stability
        
        # Contribution analysis
        if len(sample_returns) > hybrid_model.gru.sequence_length:
            try:
                contribution_analysis = self._analyze_model_contributions(
                    hybrid_model, sample_returns, sample_volatility
                )
                analysis['contribution_analysis'] = contribution_analysis
            except Exception as e:
                warnings.warn(f"Could not analyze model contributions: {e}")
                analysis['contribution_analysis'] = None
        
        return analysis
    
    def _analyze_activations(self, model: nn.Module, sample_input: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze activation patterns in the GRU model"""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu().numpy()
                else:
                    activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.GRU, nn.ReLU, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sample_input)
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def _analyze_model_contributions(self, hybrid_model, returns: pd.Series, 
                                   volatility: pd.Series) -> Dict[str, float]:
        """Analyze relative contributions of GARCH vs GRU components"""
        seq_len = hybrid_model.gru.sequence_length
        
        if len(returns) < seq_len + 10:
            return {'insufficient_data': True}
        
        # Get recent data
        recent_returns = returns.tail(seq_len + 10)
        recent_volatility = volatility.tail(seq_len + 10)
        
        contributions = {
            'garch_contribution': 0,
            'gru_contribution': 0,
            'interaction_effect': 0
        }
        
        try:
            # Generate predictions for last few points
            predictions = []
            for i in range(5):  # Last 5 predictions
                idx = len(recent_returns) - 5 + i
                forecast_data = hybrid_model.forecast(
                    recent_returns[:idx], 
                    recent_volatility[:idx]
                )
                predictions.append(forecast_data)
            
            # Extract component contributions
            if predictions and 'weights' in predictions[0]:
                avg_garch_weight = np.mean([p['weights']['garch'] for p in predictions])
                avg_gru_weight = np.mean([p['weights']['gru'] for p in predictions])
                
                contributions['garch_contribution'] = avg_garch_weight
                contributions['gru_contribution'] = avg_gru_weight
                contributions['weight_consistency'] = 1 - np.std([p['weights']['garch'] for p in predictions])
        
        except Exception as e:
            warnings.warn(f"Could not analyze contributions: {e}")
        
        return contributions
    
    def calculate_feature_importance(self, model, sample_data: np.ndarray, 
                                   baseline_prediction: float) -> Dict[str, float]:
        """Calculate feature importance using sensitivity analysis"""
        if len(sample_data.shape) != 3:  # batch_size, seq_len, features
            raise ValueError("Expected 3D input: (batch_size, seq_len, features)")
        
        importance = {}
        
        # Feature-wise importance
        for feature_idx in range(sample_data.shape[2]):
            feature_name = f"feature_{feature_idx}"
            
            # Permute feature values
            perturbed_data = sample_data.copy()
            np.random.shuffle(perturbed_data[:, :, feature_idx])
            
            # Calculate prediction with perturbed feature
            try:
                if hasattr(model, 'predict'):
                    perturbed_pred = model.predict(perturbed_data)
                else:
                    # Assume it's a PyTorch model
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(perturbed_data)
                        perturbed_pred = model(input_tensor).numpy()
                
                # Importance = change in prediction
                if isinstance(perturbed_pred, np.ndarray):
                    perturbed_pred = perturbed_pred.mean()
                
                importance[feature_name] = abs(baseline_prediction - perturbed_pred)
            
            except Exception as e:
                warnings.warn(f"Could not calculate importance for {feature_name}: {e}")
                importance[feature_name] = 0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def analyze_full_model(self, hybrid_model, returns: pd.Series, 
                          volatility: pd.Series, actual_predictions: Optional[np.ndarray] = None) -> ModelAnalysisResults:
        """Comprehensive analysis of the complete model"""
        
        # GARCH analysis
        garch_analysis = self.analyze_garch_model(hybrid_model.garch, returns)
        
        # GRU analysis
        sample_input = None
        if len(returns) >= hybrid_model.gru.sequence_length:
            # Create sample input for GRU analysis
            recent_returns = returns.tail(hybrid_model.gru.sequence_length).values
            recent_vol = volatility.tail(hybrid_model.gru.sequence_length).values
            sample_input = np.column_stack([recent_returns, recent_vol]).reshape(1, -1, 2)
        
        gru_analysis = self.analyze_gru_model(hybrid_model.gru, sample_input)
        
        # Hybrid analysis
        hybrid_analysis = self.analyze_hybrid_model(hybrid_model, returns, volatility)
        
        # Feature importance (if possible)
        feature_importance = {}
        if sample_input is not None:
            try:
                # Get baseline prediction
                baseline_pred = hybrid_model.forecast(
                    returns.tail(hybrid_model.gru.sequence_length),
                    volatility.tail(hybrid_model.gru.sequence_length)
                )['combined'][0]
                
                feature_importance = self.calculate_feature_importance(
                    hybrid_model.gru, sample_input, baseline_pred
                )
            except Exception as e:
                warnings.warn(f"Could not calculate feature importance: {e}")
        
        # Generate alerts and recommendations
        red_flags, warnings_list, recommendations = self._generate_model_alerts(
            garch_analysis, gru_analysis, hybrid_analysis
        )
        
        # Calculate health scores
        model_health = self._calculate_model_health(garch_analysis, gru_analysis, red_flags)
        interpretability = self._calculate_interpretability_score(feature_importance, hybrid_analysis)
        
        # Parameter health assessment
        param_health = {
            'garch': 'pass' if all(garch_analysis['validity'].values()) else 'fail',
            'gru': 'pass' if len(red_flags) == 0 else 'warning',
            'hybrid': 'pass' if hybrid_analysis.get('weights') else 'unknown'
        }
        
        return ModelAnalysisResults(
            garch_parameters=garch_analysis['parameters'],
            garch_validity=garch_analysis['validity'],
            garch_persistence=garch_analysis['persistence'],
            garch_half_life=garch_analysis['half_life'],
            garch_unconditional_vol=garch_analysis['unconditional_vol'],
            
            gru_architecture=gru_analysis['architecture'],
            gru_parameter_stats=gru_analysis['parameter_stats'],
            gru_activation_patterns=gru_analysis['activation_patterns'],
            gru_gradient_analysis=gru_analysis['gradient_stats'],
            
            hybrid_weights=hybrid_analysis.get('weights'),
            hybrid_weight_stability=hybrid_analysis.get('weight_stability'),
            hybrid_contribution_analysis=hybrid_analysis.get('contribution_analysis'),
            
            feature_importance=feature_importance,
            sequence_importance=None,  # Placeholder for sequence-level importance
            
            parameter_health=param_health,
            calibration_metrics={},  # Placeholder
            model_complexity={'gru_params': gru_analysis['architecture']['total_parameters']},
            
            red_flags=red_flags,
            warnings=warnings_list,
            recommendations=recommendations,
            
            model_health_score=model_health,
            interpretability_score=interpretability
        )
    
    def _generate_model_alerts(self, garch_analysis: Dict, gru_analysis: Dict, 
                              hybrid_analysis: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Generate alerts and recommendations for model behavior"""
        red_flags = []
        warnings_list = []
        recommendations = []
        
        # GARCH alerts
        validity = garch_analysis['validity']
        if not validity['omega_positive']:
            red_flags.append("GARCH omega parameter is not positive")
        if not validity['persistence_valid']:
            red_flags.append("GARCH model is non-stationary (α + β ≥ 1)")
        if not validity['stationarity']:
            warnings_list.append("GARCH persistence very high (α + β > 0.99)")
        
        if garch_analysis['half_life'] > 100:
            warnings_list.append(f"Very slow volatility decay (half-life: {garch_analysis['half_life']:.1f} days)")
        
        # GRU alerts
        arch = gru_analysis['architecture']
        if arch['total_parameters'] > 100000:
            warnings_list.append("GRU model has many parameters - may overfit")
        
        param_stats = gru_analysis['parameter_stats']
        for layer_name, stats in param_stats.items():
            if stats['norm'] < 0.01:
                warnings_list.append(f"Very small weights in {layer_name}")
            elif stats['norm'] > 10:
                warnings_list.append(f"Very large weights in {layer_name}")
            
            if stats['sparsity'] > 0.5:
                warnings_list.append(f"High sparsity in {layer_name} ({stats['sparsity']:.2%})")
        
        # Hybrid alerts
        weights = hybrid_analysis.get('weights')
        if weights:
            if weights['garch'] < 0.1 or weights['garch'] > 0.9:
                warnings_list.append(f"Hybrid weights at boundary (GARCH: {weights['garch']:.3f})")
        
        # Recommendations
        if garch_analysis['persistence'] > 0.95:
            recommendations.append("Consider using IGARCH or alternative volatility model")
        
        if len(warnings_list) > 3:
            recommendations.append("Consider model simplification or regularization")
        
        return red_flags, warnings_list, recommendations
    
    def _calculate_model_health(self, garch_analysis: Dict, gru_analysis: Dict, 
                               red_flags: List[str]) -> float:
        """Calculate overall model health score"""
        score = 1.0
        
        # Penalize red flags
        score -= 0.3 * len(red_flags)
        
        # GARCH health
        garch_validity = garch_analysis['validity']
        score += 0.2 if all(garch_validity.values()) else -0.1
        
        # GRU health (based on parameter norms)
        param_stats = gru_analysis['parameter_stats']
        healthy_norms = sum(1 for stats in param_stats.values() 
                           if 0.1 <= stats['norm'] <= 5.0)
        total_layers = len(param_stats)
        if total_layers > 0:
            norm_health = healthy_norms / total_layers
            score += 0.1 * norm_health
        
        return max(0.0, min(1.0, score))
    
    def _calculate_interpretability_score(self, feature_importance: Dict, 
                                        hybrid_analysis: Dict) -> float:
        """Calculate model interpretability score"""
        score = 0.5  # Base score
        
        # Feature importance availability
        if feature_importance:
            score += 0.2
            
            # Check if importance is well-distributed (not dominated by single feature)
            importances = list(feature_importance.values())
            if importances:
                max_importance = max(importances)
                if max_importance < 0.8:  # No single feature dominates
                    score += 0.1
        
        # Hybrid weight interpretability
        weights = hybrid_analysis.get('weights')
        if weights:
            score += 0.2
            # Balanced weights are more interpretable
            if 0.2 <= weights['garch'] <= 0.8:
                score += 0.1
        
        return min(1.0, score)
    
    def create_diagnostic_plots(self, analysis_results: ModelAnalysisResults, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive model analysis plots"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # GARCH parameters
        ax = axes[0, 0]
        params = analysis_results.garch_parameters
        param_names = ['omega', 'alpha', 'beta']
        param_values = [params[name] for name in param_names]
        colors = ['green' if analysis_results.garch_validity.get(f'{name}_valid', 
                  analysis_results.garch_validity.get('omega_positive', True)) else 'red' 
                 for name in param_names]
        
        bars = ax.bar(param_names, param_values, color=colors, alpha=0.7)
        ax.set_title('GARCH Parameters')
        ax.set_ylabel('Parameter Value')
        ax.grid(True, alpha=0.3)
        
        # Add persistence line
        persistence = analysis_results.garch_persistence
        ax.axhline(y=persistence, color='blue', linestyle='--', 
                  label=f'Persistence: {persistence:.3f}')
        ax.legend()
        
        # GRU architecture
        ax = axes[0, 1]
        arch = analysis_results.gru_architecture
        layer_info = [
            f"Input: {arch['input_size']}",
            f"Hidden: {arch['hidden_sizes']}",
            f"Output: {arch['output_size']}",
            f"Dropout: {arch['dropout_rate']}",
            f"Params: {arch['total_parameters']:,}"
        ]
        
        ax.text(0.05, 0.95, '\n'.join(layer_info), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('GRU Architecture')
        
        # Feature importance
        ax = axes[0, 2]
        if analysis_results.feature_importance:
            features = list(analysis_results.feature_importance.keys())
            importances = list(analysis_results.feature_importance.values())
            
            ax.bar(features, importances, alpha=0.7)
            ax.set_title('Feature Importance')
            ax.set_ylabel('Importance')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')
        
        # Parameter health summary
        ax = axes[1, 0]
        health_data = []
        for component, health in analysis_results.parameter_health.items():
            symbol = "✅" if health == "pass" else "⚠️" if health == "warning" else "❌" if health == "fail" else "❓"
            health_data.append(f"{symbol} {component}: {health}")
        
        health_data.extend([
            "",
            f"Model Health: {analysis_results.model_health_score:.3f}",
            f"Interpretability: {analysis_results.interpretability_score:.3f}"
        ])
        
        ax.text(0.05, 0.95, '\n'.join(health_data), transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Health Summary')
        
        # Hybrid weights (if available)
        ax = axes[1, 1]
        if analysis_results.hybrid_weights:
            weights = analysis_results.hybrid_weights
            ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', startangle=90)
            ax.set_title('Hybrid Model Weights')
        else:
            ax.text(0.5, 0.5, 'No hybrid weights available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hybrid Weights (N/A)')
        
        # GARCH validity checks
        ax = axes[1, 2]
        validity_names = list(analysis_results.garch_validity.keys())
        validity_values = [1 if v else 0 for v in analysis_results.garch_validity.values()]
        colors = ['green' if v else 'red' for v in analysis_results.garch_validity.values()]
        
        ax.barh(validity_names, validity_values, color=colors, alpha=0.7)
        ax.set_title('GARCH Validity Checks')
        ax.set_xlabel('Pass (1) / Fail (0)')
        ax.set_xlim(0, 1.2)
        
        # Parameter statistics heatmap
        ax = axes[2, 0]
        if analysis_results.gru_parameter_stats:
            stats_data = []
            stat_names = ['mean', 'std', 'norm']
            layer_names = []
            
            for layer_name, stats in analysis_results.gru_parameter_stats.items():
                layer_names.append(layer_name.replace('model.', ''))
                stats_data.append([stats.get(stat, 0) for stat in stat_names])
            
            if stats_data:
                im = ax.imshow(stats_data, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(stat_names)))
                ax.set_xticklabels(stat_names)
                ax.set_yticks(range(len(layer_names)))
                ax.set_yticklabels(layer_names, fontsize=8)
                ax.set_title('Parameter Statistics')
                plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No parameter stats', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Stats (N/A)')
        
        # Alerts and recommendations
        ax = axes[2, 1]
        alert_text = []
        
        if analysis_results.red_flags:
            alert_text.append("🚨 RED FLAGS:")
            for flag in analysis_results.red_flags:
                alert_text.append(f"  • {flag}")
        
        if analysis_results.warnings:
            alert_text.append("\n⚠️  WARNINGS:")
            for warning in analysis_results.warnings:
                alert_text.append(f"  • {warning}")
        
        if analysis_results.recommendations:
            alert_text.append("\n💡 RECOMMENDATIONS:")
            for rec in analysis_results.recommendations:
                alert_text.append(f"  • {rec}")
        
        if not alert_text:
            alert_text = ["✅ No issues detected"]
        
        ax.text(0.05, 0.95, '\n'.join(alert_text), transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Alerts & Recommendations')
        
        # Model complexity metrics
        ax = axes[2, 2]
        complexity_data = [
            f"GRU Parameters: {analysis_results.model_complexity.get('gru_params', 'N/A'):,}",
            f"GARCH Persistence: {analysis_results.garch_persistence:.4f}",
            f"GARCH Half-life: {analysis_results.garch_half_life:.1f} days",
            f"Unconditional Vol: {analysis_results.garch_unconditional_vol:.4f}"
        ]
        
        ax.text(0.05, 0.95, '\n'.join(complexity_data), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Complexity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig