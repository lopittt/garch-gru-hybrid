# src/diagnostics/training_monitor.py
# Training diagnostics and monitoring for GARCH-GRU model
# Tracks loss curves, convergence, overfitting, and weight evolution
# RELEVANT FILES: gru.py, hybrid.py, report_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class TrainingDiagnostics:
    """Container for training diagnostic results"""
    # Loss tracking
    train_losses: List[float]
    val_losses: List[float]
    epochs_completed: int
    early_stopped: bool
    best_epoch: int
    
    # Convergence analysis
    converged: bool
    convergence_epoch: Optional[int]
    final_improvement_rate: float
    stagnation_epochs: int
    
    # Overfitting detection
    overfitting_detected: bool
    overfitting_start_epoch: Optional[int]
    max_val_loss_increase: float
    train_val_gap: float
    
    # Weight evolution (for hybrid model)
    weight_history: Optional[Dict[str, List[float]]]
    weight_converged: bool
    final_weights: Optional[Dict[str, float]]
    
    # Red flags
    red_flags: List[str]
    warnings: List[str]
    
    # Overall health
    training_health_score: float  # 0-1 scale
    passed_validation: bool

class TrainingMonitor:
    """Comprehensive training diagnostics and monitoring system"""
    
    def __init__(self, convergence_patience: int = 10, min_improvement: float = 1e-6):
        self.convergence_patience = convergence_patience
        self.min_improvement = min_improvement
        self.reset()
    
    def reset(self):
        """Reset monitor for new training session"""
        self.train_losses = []
        self.val_losses = []
        self.weight_history = {'garch': [], 'gru': []}
        self.gradient_norms = []
        self.learning_rates = []
        self.epoch_times = []
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  weights: Optional[Dict[str, float]] = None,
                  gradient_norm: Optional[float] = None,
                  learning_rate: Optional[float] = None,
                  epoch_time: Optional[float] = None):
        """Log metrics for a single epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if weights:
            self.weight_history['garch'].append(weights.get('garch', 0.5))
            self.weight_history['gru'].append(weights.get('gru', 0.5))
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
            
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
    
    def analyze_training(self) -> TrainingDiagnostics:
        """Comprehensive analysis of training session"""
        if len(self.train_losses) < 3:
            raise ValueError("Need at least 3 epochs for meaningful analysis")
        
        # Basic metrics
        epochs_completed = len(self.train_losses)
        best_epoch = np.argmin(self.val_losses)
        early_stopped = best_epoch < epochs_completed - 5  # Stopped well before end
        
        # Convergence analysis
        convergence_results = self._analyze_convergence()
        
        # Overfitting detection
        overfitting_results = self._detect_overfitting()
        
        # Weight evolution analysis
        weight_results = self._analyze_weights()
        
        # Generate red flags and warnings
        red_flags, warnings = self._generate_alerts()
        
        # Calculate overall health score
        health_score = self._calculate_health_score(convergence_results, overfitting_results, red_flags)
        
        return TrainingDiagnostics(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            epochs_completed=epochs_completed,
            early_stopped=early_stopped,
            best_epoch=best_epoch,
            
            converged=convergence_results['converged'],
            convergence_epoch=convergence_results['convergence_epoch'],
            final_improvement_rate=convergence_results['final_improvement_rate'],
            stagnation_epochs=convergence_results['stagnation_epochs'],
            
            overfitting_detected=overfitting_results['detected'],
            overfitting_start_epoch=overfitting_results['start_epoch'],
            max_val_loss_increase=overfitting_results['max_increase'],
            train_val_gap=overfitting_results['final_gap'],
            
            weight_history=self.weight_history if any(self.weight_history.values()) else None,
            weight_converged=weight_results['converged'],
            final_weights=weight_results['final_weights'],
            
            red_flags=red_flags,
            warnings=warnings,
            training_health_score=health_score,
            passed_validation=len(red_flags) == 0
        )
    
    def _analyze_convergence(self) -> Dict:
        """Analyze training convergence patterns"""
        val_losses = np.array(self.val_losses)
        
        # Find convergence point (where improvement becomes minimal)
        improvements = -np.diff(val_losses)  # Negative because lower loss is better
        
        # Smooth improvements to reduce noise
        if len(improvements) > 5:
            smoothed = np.convolve(improvements, np.ones(5)/5, mode='valid')
            convergence_threshold = self.min_improvement
            
            # Find last significant improvement
            significant_improvements = np.where(smoothed > convergence_threshold)[0]
            convergence_epoch = significant_improvements[-1] + 3 if len(significant_improvements) > 0 else None
        else:
            convergence_epoch = None
        
        # Calculate final improvement rate (last 10 epochs)
        final_epochs = min(10, len(improvements))
        final_improvement_rate = np.mean(improvements[-final_epochs:]) if final_epochs > 0 else 0
        
        # Count stagnation epochs (no significant improvement)
        stagnation_count = 0
        for i in range(len(improvements)-1, -1, -1):
            if improvements[i] <= self.min_improvement:
                stagnation_count += 1
            else:
                break
        
        converged = (convergence_epoch is not None and 
                    stagnation_count >= self.convergence_patience)
        
        return {
            'converged': converged,
            'convergence_epoch': convergence_epoch,
            'final_improvement_rate': final_improvement_rate,
            'stagnation_epochs': stagnation_count
        }
    
    def _detect_overfitting(self) -> Dict:
        """Detect overfitting patterns"""
        train_losses = np.array(self.train_losses)
        val_losses = np.array(self.val_losses)
        
        if len(val_losses) < 10:
            return {
                'detected': False,
                'start_epoch': None,
                'max_increase': 0,
                'final_gap': abs(train_losses[-1] - val_losses[-1])
            }
        
        # Find where validation loss starts consistently increasing
        # while training loss continues decreasing
        overfitting_start = None
        val_increases = 0
        train_decreases = 0
        
        for i in range(10, len(val_losses)):
            # Look at trends over last 5 epochs
            val_trend = val_losses[i] > np.mean(val_losses[i-5:i])
            train_trend = train_losses[i] < np.mean(train_losses[i-5:i])
            
            if val_trend and train_trend:
                if overfitting_start is None:
                    overfitting_start = i - 5
                val_increases += 1
                train_decreases += 1
            else:
                # Reset if pattern breaks
                val_increases = 0
                train_decreases = 0
                overfitting_start = None
        
        # Overfitting detected if pattern persists for 5+ epochs
        overfitting_detected = val_increases >= 5 and train_decreases >= 5
        
        # Calculate maximum validation loss increase after best point
        best_val_idx = np.argmin(val_losses)
        max_val_increase = (np.max(val_losses[best_val_idx:]) - val_losses[best_val_idx] 
                           if best_val_idx < len(val_losses) - 1 else 0)
        
        # Final train-validation gap
        final_gap = abs(train_losses[-1] - val_losses[-1])
        
        return {
            'detected': overfitting_detected,
            'start_epoch': overfitting_start,
            'max_increase': max_val_increase,
            'final_gap': final_gap
        }
    
    def _analyze_weights(self) -> Dict:
        """Analyze weight evolution for hybrid models"""
        if not any(self.weight_history.values()):
            return {'converged': True, 'final_weights': None}
        
        garch_weights = np.array(self.weight_history['garch'])
        gru_weights = np.array(self.weight_history['gru'])
        
        if len(garch_weights) < 10:
            return {
                'converged': False,
                'final_weights': {'garch': garch_weights[-1], 'gru': gru_weights[-1]}
            }
        
        # Check for convergence (stable weights in last 10 epochs)
        recent_garch = garch_weights[-10:]
        recent_gru = gru_weights[-10:]
        
        garch_stable = np.std(recent_garch) < 0.05
        gru_stable = np.std(recent_gru) < 0.05
        
        weight_converged = garch_stable and gru_stable
        
        return {
            'converged': weight_converged,
            'final_weights': {'garch': float(garch_weights[-1]), 'gru': float(gru_weights[-1])}
        }
    
    def _generate_alerts(self) -> Tuple[List[str], List[str]]:
        """Generate red flags and warnings based on training patterns"""
        red_flags = []
        warnings = []
        
        if len(self.train_losses) < 10:
            red_flags.append("Training stopped too early (< 10 epochs)")
        
        # Check for non-decreasing losses
        if len(self.train_losses) >= 5:
            recent_train = self.train_losses[-5:]
            if all(recent_train[i] >= recent_train[i-1] for i in range(1, 5)):
                red_flags.append("Training loss not decreasing in last 5 epochs")
        
        # Check for exploding losses
        if any(loss > 100 * self.train_losses[0] for loss in self.train_losses):
            red_flags.append("Loss explosion detected")
        
        # Check for validation loss increase
        if len(self.val_losses) >= 10:
            best_val = min(self.val_losses)
            current_val = self.val_losses[-1]
            if current_val > 2 * best_val:
                red_flags.append("Validation loss increased >100% from best")
        
        # Check gradient norms if available
        if self.gradient_norms:
            recent_grads = self.gradient_norms[-10:]
            if np.mean(recent_grads) < 1e-8:
                warnings.append("Very small gradient norms - may indicate vanishing gradients")
            elif np.mean(recent_grads) > 10:
                warnings.append("Large gradient norms - may indicate exploding gradients")
        
        # Check weight boundaries for hybrid models
        if self.weight_history and any(self.weight_history.values()):
            final_garch = self.weight_history['garch'][-1]
            if final_garch <= 0.1 or final_garch >= 0.9:
                warnings.append(f"Hybrid weights at boundary (GARCH: {final_garch:.3f})")
        
        return red_flags, warnings
    
    def _calculate_health_score(self, convergence_results: Dict, 
                               overfitting_results: Dict, red_flags: List[str]) -> float:
        """Calculate overall training health score (0-1)"""
        score = 1.0
        
        # Penalize red flags heavily
        score -= 0.3 * len(red_flags)
        
        # Reward convergence
        if convergence_results['converged']:
            score += 0.1
        else:
            score -= 0.2
        
        # Penalize overfitting
        if overfitting_results['detected']:
            score -= 0.2
        
        # Penalize excessive train-val gap
        if overfitting_results['final_gap'] > 0.01:  # Arbitrary threshold
            score -= 0.1
        
        # Ensure score stays in [0, 1]
        return max(0.0, min(1.0, score))
    
    def create_diagnostic_plots(self, diagnostics: TrainingDiagnostics, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        ax = axes[0, 0]
        epochs = range(1, len(diagnostics.train_losses) + 1)
        ax.plot(epochs, diagnostics.train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax.plot(epochs, diagnostics.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax.axvline(x=diagnostics.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label='Best Epoch')
        if diagnostics.convergence_epoch:
            ax.axvline(x=diagnostics.convergence_epoch + 1, color='orange', linestyle='--', alpha=0.5, label='Convergence')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss improvement rate
        ax = axes[0, 1]
        if len(diagnostics.val_losses) > 1:
            improvements = -np.diff(diagnostics.val_losses)
            ax.plot(range(2, len(diagnostics.val_losses) + 1), improvements, 'g-', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss Improvement')
            ax.set_title('Learning Progress')
            ax.grid(True, alpha=0.3)
        
        # Weight evolution (if available)
        ax = axes[0, 2]
        if diagnostics.weight_history:
            epochs_w = range(1, len(diagnostics.weight_history['garch']) + 1)
            ax.plot(epochs_w, diagnostics.weight_history['garch'], 'b-', label='GARCH Weight', alpha=0.8)
            ax.plot(epochs_w, diagnostics.weight_history['gru'], 'r-', label='GRU Weight', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight')
            ax.set_title('Hybrid Weight Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No weight history available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Evolution (N/A)')
        
        # Training summary statistics
        ax = axes[1, 0]
        stats_data = [
            f"Epochs: {diagnostics.epochs_completed}",
            f"Best Epoch: {diagnostics.best_epoch + 1}",
            f"Early Stopped: {diagnostics.early_stopped}",
            f"Converged: {diagnostics.converged}",
            f"Overfitting: {diagnostics.overfitting_detected}",
            f"Health Score: {diagnostics.training_health_score:.3f}",
            f"Final Train Loss: {diagnostics.train_losses[-1]:.6f}",
            f"Final Val Loss: {diagnostics.val_losses[-1]:.6f}"
        ]
        ax.text(0.05, 0.95, '\n'.join(stats_data), transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Training Summary')
        
        # Red flags and warnings
        ax = axes[1, 1]
        alerts_text = []
        if diagnostics.red_flags:
            alerts_text.append("🚨 RED FLAGS:")
            for flag in diagnostics.red_flags:
                alerts_text.append(f"  • {flag}")
        if diagnostics.warnings:
            alerts_text.append("\n⚠️  WARNINGS:")
            for warning in diagnostics.warnings:
                alerts_text.append(f"  • {warning}")
        if not alerts_text:
            alerts_text = ["✅ No alerts"]
        
        ax.text(0.05, 0.95, '\n'.join(alerts_text), transform=ax.transAxes,
                fontsize=10, verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Alerts & Warnings')
        
        # Training efficiency
        ax = axes[1, 2]
        if self.epoch_times:
            ax.plot(range(1, len(self.epoch_times) + 1), self.epoch_times, 'g-', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Speed')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Speed (N/A)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig