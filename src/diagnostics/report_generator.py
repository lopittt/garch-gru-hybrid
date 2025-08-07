# src/diagnostics/report_generator.py
# Automated diagnostic report generation with HTML dashboard
# Integrates all diagnostic modules into comprehensive reports
# RELEVANT FILES: training_monitor.py, statistical_tests.py, model_analyzer.py, simulation_validator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import os
import warnings

# Import diagnostic modules
from .training_monitor import TrainingDiagnostics, TrainingMonitor
from .statistical_tests import StatisticalTestResults, StatisticalValidator  
from .model_analyzer import ModelAnalysisResults, ModelAnalyzer
from .simulation_validator import SimulationValidationResults, SimulationValidator

@dataclass
class ComprehensiveReport:
    """Container for complete diagnostic report"""
    # Metadata
    report_id: str
    timestamp: str
    model_name: str
    configuration: Dict[str, Any]
    
    # Diagnostic results
    training_diagnostics: Optional[TrainingDiagnostics]
    statistical_validation: Optional[StatisticalTestResults]
    model_analysis: Optional[ModelAnalysisResults]
    simulation_validation: Optional[SimulationValidationResults]
    
    # Overall assessment
    overall_health_score: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    passed_validation: bool
    
    # Performance benchmarks
    performance_metrics: Dict[str, float]
    benchmark_comparisons: Dict[str, str]  # metric -> comparison_result
    
    # Execution info
    execution_time: float
    resource_usage: Dict[str, float]

class ReportGenerator:
    """Comprehensive diagnostic report generation system"""
    
    def __init__(self, output_dir: str = "reports", save_plots: bool = True):
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.plot_paths = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize diagnostic modules
        self.training_monitor = TrainingMonitor()
        self.statistical_validator = StatisticalValidator()
        self.model_analyzer = ModelAnalyzer()
        self.simulation_validator = SimulationValidator()
    
    def generate_comprehensive_report(self, 
                                    model_name: str,
                                    configuration: Dict[str, Any],
                                    training_data: Optional[Dict] = None,
                                    model_data: Optional[Dict] = None,
                                    simulation_data: Optional[Dict] = None,
                                    performance_data: Optional[Dict] = None,
                                    execution_info: Optional[Dict] = None) -> ComprehensiveReport:
        """Generate comprehensive diagnostic report from all available data"""
        
        report_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # Initialize results containers
        training_results = None
        statistical_results = None
        model_results = None
        simulation_results = None
        
        # Process training diagnostics
        if training_data:
            try:
                training_results = self._process_training_data(training_data, report_id)
            except Exception as e:
                warnings.warn(f"Training diagnostics failed: {e}")
        
        # Process statistical validation
        if model_data and 'actual' in model_data and 'predicted' in model_data:
            try:
                statistical_results = self._process_statistical_data(model_data, report_id)
            except Exception as e:
                warnings.warn(f"Statistical validation failed: {e}")
        
        # Process model analysis
        if model_data and 'model' in model_data:
            try:
                model_results = self._process_model_data(model_data, report_id)
            except Exception as e:
                warnings.warn(f"Model analysis failed: {e}")
        
        # Process simulation validation
        if simulation_data:
            try:
                simulation_results = self._process_simulation_data(simulation_data, report_id)
            except Exception as e:
                warnings.warn(f"Simulation validation failed: {e}")
        
        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(
            training_results, statistical_results, model_results, simulation_results
        )
        
        # Process performance metrics
        perf_metrics = performance_data or {}
        benchmark_comparisons = self._compare_with_benchmarks(perf_metrics)
        
        # Execution info
        exec_info = execution_info or {}
        
        report = ComprehensiveReport(
            report_id=report_id,
            timestamp=timestamp,
            model_name=model_name,
            configuration=configuration,
            
            training_diagnostics=training_results,
            statistical_validation=statistical_results,
            model_analysis=model_results,
            simulation_validation=simulation_results,
            
            overall_health_score=overall_assessment['health_score'],
            critical_issues=overall_assessment['critical_issues'],
            warnings=overall_assessment['warnings'],
            recommendations=overall_assessment['recommendations'],
            passed_validation=overall_assessment['passed'],
            
            performance_metrics=perf_metrics,
            benchmark_comparisons=benchmark_comparisons,
            
            execution_time=exec_info.get('total_time', 0),
            resource_usage=exec_info.get('resources', {})
        )
        
        return report
    
    def _process_training_data(self, training_data: Dict, report_id: str) -> TrainingDiagnostics:
        """Process training data and generate diagnostics"""
        
        # Extract training history
        train_losses = training_data.get('train_losses', [])
        val_losses = training_data.get('val_losses', [])
        weight_history = training_data.get('weight_history', {})
        
        # Update training monitor with data
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            weights = None
            if weight_history:
                weights = {k: v[i] if i < len(v) else v[-1] for k, v in weight_history.items()}
            
            self.training_monitor.log_epoch(
                epoch=i,
                train_loss=train_loss,
                val_loss=val_loss,
                weights=weights
            )
        
        # Analyze training
        diagnostics = self.training_monitor.analyze_training()
        
        # Generate plots
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, f"{report_id}_training.png")
            fig = self.training_monitor.create_diagnostic_plots(diagnostics, plot_path)
            plt.close(fig)
            self.plot_paths['training'] = plot_path
        
        return diagnostics
    
    def _process_statistical_data(self, model_data: Dict, report_id: str) -> StatisticalTestResults:
        """Process model predictions for statistical validation"""
        
        actual = np.array(model_data['actual'])
        predicted = np.array(model_data['predicted'])
        residuals = model_data.get('residuals', actual - predicted)
        
        # Run statistical validation
        results = self.statistical_validator.validate_model(
            actual, predicted, residuals, model_data.get('model_name', 'Model')
        )
        
        # Generate plots
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, f"{report_id}_statistical.png")
            fig = self.statistical_validator.create_diagnostic_plots(
                actual, predicted, residuals, results, plot_path
            )
            plt.close(fig)
            self.plot_paths['statistical'] = plot_path
        
        return results
    
    def _process_model_data(self, model_data: Dict, report_id: str) -> ModelAnalysisResults:
        """Process model for behavioral analysis"""
        
        model = model_data['model']
        returns = model_data.get('returns', pd.Series([]))
        volatility = model_data.get('volatility', pd.Series([]))
        
        # Analyze model
        results = self.model_analyzer.analyze_full_model(model, returns, volatility)
        
        # Generate plots
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, f"{report_id}_model_analysis.png")
            fig = self.model_analyzer.create_diagnostic_plots(results, plot_path)
            plt.close(fig)
            self.plot_paths['model_analysis'] = plot_path
        
        return results
    
    def _process_simulation_data(self, simulation_data: Dict, report_id: str) -> SimulationValidationResults:
        """Process simulation results for validation"""
        
        simulated_paths = simulation_data['simulated_paths']
        historical_returns = simulation_data['historical_returns']
        
        # Validate simulation
        results = self.simulation_validator.validate_simulation(
            simulated_paths, historical_returns, simulation_data.get('model_name', 'Model')
        )
        
        # Generate plots
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, f"{report_id}_simulation.png")
            fig = self.simulation_validator.create_diagnostic_plots(
                results, simulated_paths, historical_returns, plot_path
            )
            plt.close(fig)
            self.plot_paths['simulation'] = plot_path
        
        return results
    
    def _calculate_overall_assessment(self, training_results, statistical_results, 
                                    model_results, simulation_results) -> Dict[str, Any]:
        """Calculate overall health assessment"""
        
        scores = []
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Training assessment
        if training_results:
            scores.append(training_results.training_health_score)
            critical_issues.extend(training_results.red_flags)
            warnings.extend(training_results.warnings)
        
        # Statistical assessment
        if statistical_results:
            scores.append(statistical_results.overall_validity)
            critical_issues.extend(statistical_results.critical_failures)
            warnings.extend(statistical_results.warnings)
        
        # Model assessment
        if model_results:
            scores.append(model_results.model_health_score)
            critical_issues.extend(model_results.red_flags)
            warnings.extend(model_results.warnings)
            recommendations.extend(model_results.recommendations)
        
        # Simulation assessment
        if simulation_results:
            scores.append(simulation_results.simulation_quality_score)
            critical_issues.extend(simulation_results.red_flags)
            warnings.extend(simulation_results.warnings)
            recommendations.extend(simulation_results.recommendations)
        
        # Overall score
        overall_score = np.mean(scores) if scores else 0.5
        passed = len(critical_issues) == 0 and overall_score > 0.6
        
        return {
            'health_score': overall_score,
            'critical_issues': list(set(critical_issues)),  # Remove duplicates
            'warnings': list(set(warnings)),
            'recommendations': list(set(recommendations)),
            'passed': passed
        }
    
    def _compare_with_benchmarks(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Compare performance metrics with benchmarks"""
        comparisons = {}
        
        # Define benchmark thresholds
        benchmarks = {
            'r2': {'excellent': 0.7, 'good': 0.5, 'acceptable': 0.3},
            'mse': {'excellent': 0.0001, 'good': 0.001, 'acceptable': 0.01},
            'mae': {'excellent': 0.005, 'good': 0.01, 'acceptable': 0.02}
        }
        
        for metric, value in metrics.items():
            if metric.lower() in benchmarks:
                thresholds = benchmarks[metric.lower()]
                
                if metric.lower() in ['r2']:  # Higher is better
                    if value >= thresholds['excellent']:
                        comparisons[metric] = 'excellent'
                    elif value >= thresholds['good']:
                        comparisons[metric] = 'good'
                    elif value >= thresholds['acceptable']:
                        comparisons[metric] = 'acceptable'
                    else:
                        comparisons[metric] = 'poor'
                else:  # Lower is better (MSE, MAE)
                    if value <= thresholds['excellent']:
                        comparisons[metric] = 'excellent'
                    elif value <= thresholds['good']:
                        comparisons[metric] = 'good'
                    elif value <= thresholds['acceptable']:
                        comparisons[metric] = 'acceptable'
                    else:
                        comparisons[metric] = 'poor'
        
        return comparisons
    
    def export_html_report(self, report: ComprehensiveReport) -> str:
        """Export comprehensive report as interactive HTML dashboard"""
        
        html_template = self._get_html_template()
        
        # Convert plots to base64 for embedding
        plot_data = {}
        for plot_type, plot_path in self.plot_paths.items():
            if os.path.exists(plot_path):
                with open(plot_path, 'rb') as f:
                    plot_data[plot_type] = base64.b64encode(f.read()).decode()
        
        # Prepare data for template
        template_data = {
            'report': report,
            'plot_data': plot_data,
            'json_data': json.dumps(asdict(report), indent=2, default=str)
        }
        
        # Generate HTML
        html_content = self._render_template(html_template, template_data)
        
        # Save HTML report
        html_path = os.path.join(self.output_dir, f"{report.report_id}_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _get_html_template(self) -> str:
        """Get HTML template for report"""
        # Use double braces for CSS to escape Python string formatting
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GARCH-GRU Diagnostic Report - {report.report_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .health-indicator {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .health-excellent { background: #4CAF50; color: white; }
        .health-good { background: #8BC34A; color: white; }
        .health-warning { background: #FF9800; color: white; }
        .health-critical { background: #F44336; color: white; }
        .nav {
            background: #f8f9fa;
            padding: 0;
            border-bottom: 1px solid #dee2e6;
        }
        .nav ul {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
        }
        .nav li {
            flex: 1;
        }
        .nav a {
            display: block;
            padding: 15px 20px;
            text-decoration: none;
            color: #495057;
            font-weight: 500;
            text-align: center;
            transition: background-color 0.3s;
        }
        .nav a:hover, .nav a.active {
            background-color: #e9ecef;
            color: #007bff;
        }
        .content {
            padding: 30px;
        }
        .section {
            display: none;
            margin-bottom: 30px;
        }
        .section.active {
            display: block;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 1.1em;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
        .config-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .config-table th, .config-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        .config-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-pass { background: #28a745; color: white; }
        .status-warn { background: #ffc107; color: black; }
        .status-fail { background: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GARCH-GRU Diagnostic Report</h1>
            <div class="subtitle">{report.model_name} - {report.timestamp}</div>
            <div class="health-indicator {health_class}">
                Overall Health: {report.overall_health_score:.1%}
                {validation_status}
            </div>
        </div>
        
        <nav class="nav">
            <ul>
                <li><a href="#overview" onclick="showSection('overview')" class="active">Overview</a></li>
                <li><a href="#training" onclick="showSection('training')">Training</a></li>
                <li><a href="#statistical" onclick="showSection('statistical')">Statistical</a></li>
                <li><a href="#model" onclick="showSection('model')">Model Analysis</a></li>
                <li><a href="#simulation" onclick="showSection('simulation')">Simulation</a></li>
                <li><a href="#config" onclick="showSection('config')">Configuration</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <!-- Overview Section -->
            <div id="overview" class="section active">
                <h2>Executive Summary</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Overall Health Score</h3>
                        <div class="metric-value">{report.overall_health_score:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Validation Status</h3>
                        <div class="metric-value">{validation_status}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Execution Time</h3>
                        <div class="metric-value">{report.execution_time:.1f}s</div>
                    </div>
                    <div class="metric-card">
                        <h3>Critical Issues</h3>
                        <div class="metric-value">{num_critical}</div>
                    </div>
                </div>
                
                {alerts_html}
                
                <h3>Performance Metrics</h3>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Benchmark</th>
                        </tr>
                    </thead>
                    <tbody>
                        {performance_table}
                    </tbody>
                </table>
            </div>
            
            <!-- Training Section -->
            <div id="training" class="section">
                <h2>Training Diagnostics</h2>
                {training_content}
            </div>
            
            <!-- Statistical Section -->
            <div id="statistical" class="section">
                <h2>Statistical Validation</h2>
                {statistical_content}
            </div>
            
            <!-- Model Analysis Section -->
            <div id="model" class="section">
                <h2>Model Behavior Analysis</h2>
                {model_content}
            </div>
            
            <!-- Simulation Section -->
            <div id="simulation" class="section">
                <h2>Simulation Quality Assessment</h2>
                {simulation_content}
            </div>
            
            <!-- Configuration Section -->
            <div id="config" class="section">
                <h2>Model Configuration</h2>
                <pre>{json_data}</pre>
            </div>
        </div>
    </div>
    
    <script>
        function showSection(sectionId) {
            // Hide all sections
            var sections = document.querySelectorAll('.section');
            sections.forEach(function(section) {
                section.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
            
            // Update navigation
            var navLinks = document.querySelectorAll('.nav a');
            navLinks.forEach(function(link) {
                link.classList.remove('active');
            });
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
        """
    
    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render HTML template with data"""
        
        report = data['report']
        plot_data = data['plot_data']
        json_data = data['json_data']
        
        # Health class
        score = report.overall_health_score
        if score >= 0.8:
            health_class = 'health-excellent'
        elif score >= 0.6:
            health_class = 'health-good'  
        elif score >= 0.4:
            health_class = 'health-warning'
        else:
            health_class = 'health-critical'
        
        # Validation status
        validation_status = "PASS" if report.passed_validation else "FAIL"
        
        # Alerts HTML
        alerts_html = self._generate_alerts_html(report)
        
        # Performance table
        performance_table = self._generate_performance_table(report)
        
        # Section content
        training_content = self._generate_training_html(report, plot_data)
        statistical_content = self._generate_statistical_html(report, plot_data)
        model_content = self._generate_model_html(report, plot_data)
        simulation_content = self._generate_simulation_html(report, plot_data)
        
        # Use simple string replacement instead of .format() to avoid CSS brace conflicts
        html = template
        html = html.replace('{report.report_id}', report.report_id)
        html = html.replace('{report.model_name}', report.model_name)
        html = html.replace('{report.timestamp}', report.timestamp)
        html = html.replace('{health_class}', health_class)
        html = html.replace('{report.overall_health_score:.1%}', f"{report.overall_health_score:.1%}")
        html = html.replace('{validation_status}', validation_status)
        html = html.replace('{num_critical}', str(len(report.critical_issues)))
        html = html.replace('{report.execution_time:.1f}', f"{report.execution_time:.1f}")
        html = html.replace('{alerts_html}', alerts_html)
        html = html.replace('{performance_table}', performance_table)
        html = html.replace('{training_content}', training_content)
        html = html.replace('{statistical_content}', statistical_content)
        html = html.replace('{model_content}', model_content)
        html = html.replace('{simulation_content}', simulation_content)
        html = html.replace('{json_data}', json_data)
        
        return html
    
    def _generate_alerts_html(self, report: ComprehensiveReport) -> str:
        """Generate alerts HTML section"""
        html = []
        
        if report.critical_issues:
            html.append('<div class="alert alert-danger">')
            html.append('<strong>🚨 Critical Issues:</strong><ul>')
            for issue in report.critical_issues:
                html.append(f'<li>{issue}</li>')
            html.append('</ul></div>')
        
        if report.warnings:
            html.append('<div class="alert alert-warning">')
            html.append('<strong>⚠️ Warnings:</strong><ul>')
            for warning in report.warnings:
                html.append(f'<li>{warning}</li>')
            html.append('</ul></div>')
        
        if report.recommendations:
            html.append('<div class="alert alert-info">')
            html.append('<strong>💡 Recommendations:</strong><ul>')
            for rec in report.recommendations:
                html.append(f'<li>{rec}</li>')
            html.append('</ul></div>')
        
        if not html:
            html.append('<div class="alert alert-success">')
            html.append('<strong>✅ No issues detected</strong>')
            html.append('</div>')
        
        return '\n'.join(html)
    
    def _generate_performance_table(self, report: ComprehensiveReport) -> str:
        """Generate performance metrics table HTML"""
        rows = []
        
        for metric, value in report.performance_metrics.items():
            benchmark = report.benchmark_comparisons.get(metric, 'unknown')
            benchmark_class = {
                'excellent': 'status-pass',
                'good': 'status-pass', 
                'acceptable': 'status-warn',
                'poor': 'status-fail',
                'unknown': ''
            }.get(benchmark, '')
            
            rows.append(f'''
                <tr>
                    <td>{metric.upper()}</td>
                    <td>{value:.6f}</td>
                    <td><span class="status-badge {benchmark_class}">{benchmark}</span></td>
                </tr>
            ''')
        
        return '\n'.join(rows)
    
    def _generate_training_html(self, report: ComprehensiveReport, plot_data: Dict) -> str:
        """Generate training diagnostics HTML"""
        if not report.training_diagnostics:
            return '<p>No training diagnostics available</p>'
        
        training = report.training_diagnostics
        
        html = f'''
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Epochs Completed</h3>
                    <div class="metric-value">{training.epochs_completed}</div>
                </div>
                <div class="metric-card">
                    <h3>Best Epoch</h3>
                    <div class="metric-value">{training.best_epoch + 1}</div>
                </div>
                <div class="metric-card">
                    <h3>Converged</h3>
                    <div class="metric-value">{"Yes" if training.converged else "No"}</div>
                </div>
                <div class="metric-card">
                    <h3>Health Score</h3>
                    <div class="metric-value">{training.training_health_score:.1%}</div>
                </div>
            </div>
        '''
        
        if 'training' in plot_data:
            html += f'''
                <div class="plot-container">
                    <h3>Training Progress</h3>
                    <img src="data:image/png;base64,{plot_data['training']}" alt="Training Diagnostics"/>
                </div>
            '''
        
        return html
    
    def _generate_statistical_html(self, report: ComprehensiveReport, plot_data: Dict) -> str:
        """Generate statistical validation HTML"""
        if not report.statistical_validation:
            return '<p>No statistical validation available</p>'
        
        stats = report.statistical_validation
        
        html = f'''
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Validity Score</h3>
                    <div class="metric-value">{stats.overall_validity:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Test Summary</h3>
                    <div class="metric-value">{len(stats.test_summary)} tests</div>
                </div>
            </div>
        '''
        
        if 'statistical' in plot_data:
            html += f'''
                <div class="plot-container">
                    <h3>Statistical Diagnostics</h3>
                    <img src="data:image/png;base64,{plot_data['statistical']}" alt="Statistical Validation"/>
                </div>
            '''
        
        return html
    
    def _generate_model_html(self, report: ComprehensiveReport, plot_data: Dict) -> str:
        """Generate model analysis HTML"""
        if not report.model_analysis:
            return '<p>No model analysis available</p>'
        
        model = report.model_analysis
        
        html = f'''
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Model Health</h3>
                    <div class="metric-value">{model.model_health_score:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Interpretability</h3>
                    <div class="metric-value">{model.interpretability_score:.1%}</div>
                </div>
            </div>
        '''
        
        if 'model_analysis' in plot_data:
            html += f'''
                <div class="plot-container">
                    <h3>Model Analysis</h3>
                    <img src="data:image/png;base64,{plot_data['model_analysis']}" alt="Model Analysis"/>
                </div>
            '''
        
        return html
    
    def _generate_simulation_html(self, report: ComprehensiveReport, plot_data: Dict) -> str:
        """Generate simulation validation HTML"""
        if not report.simulation_validation:
            return '<p>No simulation validation available</p>'
        
        sim = report.simulation_validation
        
        html = f'''
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Realism Score</h3>
                    <div class="metric-value">{sim.realism_score:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Quality Score</h3>
                    <div class="metric-value">{sim.simulation_quality_score:.1%}</div>
                </div>
            </div>
        '''
        
        if 'simulation' in plot_data:
            html += f'''
                <div class="plot-container">
                    <h3>Simulation Validation</h3>
                    <img src="data:image/png;base64,{plot_data['simulation']}" alt="Simulation Validation"/>
                </div>
            '''
        
        return html
    
    def export_json_report(self, report: ComprehensiveReport) -> str:
        """Export report as JSON for programmatic access"""
        json_path = os.path.join(self.output_dir, f"{report.report_id}_report.json")
        
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        return json_path
    
    def cleanup_plots(self):
        """Clean up temporary plot files"""
        for plot_path in self.plot_paths.values():
            if os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                except:
                    pass
        self.plot_paths.clear()