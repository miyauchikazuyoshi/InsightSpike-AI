"""
Research Report Generator for InsightSpike-AI
============================================

Comprehensive system for generating publication-quality research reports
with statistical analysis, visualizations, and academic formatting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import logging
from scipy import stats
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Container for experimental results"""
    experiment_name: str
    baseline_results: Dict[str, Any]
    insightspike_results: Dict[str, Any]
    experimental_config: Dict[str, Any]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchReportConfig:
    """Configuration for research report generation"""
    # Output settings
    output_directory: Path = Path("experiments/reports")
    include_raw_data: bool = True
    include_visualizations: bool = True
    include_statistical_analysis: bool = True
    
    # Report sections
    include_abstract: bool = True
    include_methodology: bool = True
    include_results: bool = True
    include_discussion: bool = True
    include_appendix: bool = True
    
    # Statistical settings
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    effect_size_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'small': 0.2, 'medium': 0.5, 'large': 0.8
    })
    
    # Visualization settings
    figure_dpi: int = 300
    figure_format: str = 'both'  # 'png', 'pdf', or 'both'
    color_palette: str = 'viridis'


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for experimental results"""
    
    def __init__(self, config: ResearchReportConfig):
        self.config = config
        
    def analyze_experiment(self, baseline_data: List[float], 
                         insightspike_data: List[float],
                         metric_name: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        # Convert to numpy arrays and flatten if needed
        baseline_data = np.array(baseline_data).flatten()
        insightspike_data = np.array(insightspike_data).flatten()
        
        analysis = {
            'metric_name': metric_name,
            'sample_sizes': {
                'baseline': len(baseline_data),
                'insightspike': len(insightspike_data)
            }
        }
        
        # Descriptive statistics
        analysis['descriptive'] = self._compute_descriptive_stats(baseline_data, insightspike_data)
        
        # Normality tests
        analysis['normality'] = self._test_normality(baseline_data, insightspike_data)
        
        # Comparative tests
        analysis['comparative'] = self._perform_comparative_tests(baseline_data, insightspike_data)
        
        # Effect size
        analysis['effect_size'] = self._compute_effect_sizes(baseline_data, insightspike_data)
        
        # Confidence intervals
        analysis['confidence_intervals'] = self._compute_confidence_intervals(
            baseline_data, insightspike_data
        )
        
        # Power analysis
        analysis['power_analysis'] = self._compute_power_analysis(baseline_data, insightspike_data)
        
        return analysis
    
    def _compute_descriptive_stats(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Compute descriptive statistics"""
        return {
            'baseline': {
                'mean': np.mean(baseline),
                'std': np.std(baseline, ddof=1),
                'median': np.median(baseline),
                'q25': np.percentile(baseline, 25),
                'q75': np.percentile(baseline, 75),
                'min': np.min(baseline),
                'max': np.max(baseline),
                'skewness': stats.skew(baseline),
                'kurtosis': stats.kurtosis(baseline)
            },
            'insightspike': {
                'mean': np.mean(insightspike),
                'std': np.std(insightspike, ddof=1),
                'median': np.median(insightspike),
                'q25': np.percentile(insightspike, 25),
                'q75': np.percentile(insightspike, 75),
                'min': np.min(insightspike),
                'max': np.max(insightspike),
                'skewness': stats.skew(insightspike),
                'kurtosis': stats.kurtosis(insightspike)
            }
        }
    
    def _test_normality(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Test for normality using multiple tests"""
        
        # Shapiro-Wilk test
        shapiro_baseline = stats.shapiro(baseline)
        shapiro_insightspike = stats.shapiro(insightspike)
        
        # Kolmogorov-Smirnov test against normal distribution
        ks_baseline = stats.kstest(baseline, 'norm', 
                                  args=(np.mean(baseline), np.std(baseline)))
        ks_insightspike = stats.kstest(insightspike, 'norm',
                                     args=(np.mean(insightspike), np.std(insightspike)))
        
        return {
            'shapiro_wilk': {
                'baseline': {'statistic': shapiro_baseline[0], 'p_value': shapiro_baseline[1]},
                'insightspike': {'statistic': shapiro_insightspike[0], 'p_value': shapiro_insightspike[1]}
            },
            'kolmogorov_smirnov': {
                'baseline': {'statistic': ks_baseline[0], 'p_value': ks_baseline[1]},
                'insightspike': {'statistic': ks_insightspike[0], 'p_value': ks_insightspike[1]}
            }
        }
    
    def _perform_comparative_tests(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Perform comparative statistical tests"""
        
        # Student's t-test (assuming equal variances)
        ttest_equal = stats.ttest_ind(baseline, insightspike)
        
        # Welch's t-test (unequal variances)
        ttest_unequal = stats.ttest_ind(baseline, insightspike, equal_var=False)
        
        # Mann-Whitney U test (non-parametric)
        mannwhitney = stats.mannwhitneyu(baseline, insightspike, alternative='two-sided')
        
        # Kolmogorov-Smirnov test (distribution comparison)
        ks_test = stats.ks_2samp(baseline, insightspike)
        
        # Levene's test for equal variances
        levene_test = stats.levene(baseline, insightspike)
        
        return {
            't_test_equal_var': {
                'statistic': ttest_equal[0],
                'p_value': ttest_equal[1]
            },
            't_test_unequal_var': {
                'statistic': ttest_unequal[0],
                'p_value': ttest_unequal[1]
            },
            'mann_whitney_u': {
                'statistic': mannwhitney[0],
                'p_value': mannwhitney[1]
            },
            'kolmogorov_smirnov': {
                'statistic': ks_test[0],
                'p_value': ks_test[1]
            },
            'levene_variance': {
                'statistic': levene_test[0],
                'p_value': levene_test[1]
            }
        }
    
    def _compute_effect_sizes(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Compute various effect size measures"""
        
        # Cohen's d
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) + 
                             (len(insightspike) - 1) * np.var(insightspike, ddof=1)) / 
                            (len(baseline) + len(insightspike) - 2))
        cohens_d = (np.mean(insightspike) - np.mean(baseline)) / pooled_std
        
        # Glass's delta
        glass_delta = (np.mean(insightspike) - np.mean(baseline)) / np.std(baseline, ddof=1)
        
        # Hedge's g (bias-corrected Cohen's d)
        J = 1 - (3 / (4 * (len(baseline) + len(insightspike) - 2) - 1))
        hedges_g = cohens_d * J
        
        # Common language effect size
        from scipy.stats import norm
        cles = norm.cdf(cohens_d / np.sqrt(2))
        
        # Effect size interpretation
        def interpret_effect_size(d):
            abs_d = abs(d)
            if abs_d < self.config.effect_size_thresholds['small']:
                return 'negligible'
            elif abs_d < self.config.effect_size_thresholds['medium']:
                return 'small'
            elif abs_d < self.config.effect_size_thresholds['large']:
                return 'medium'
            else:
                return 'large'
        
        return {
            'cohens_d': {
                'value': cohens_d,
                'interpretation': interpret_effect_size(cohens_d)
            },
            'glass_delta': {
                'value': glass_delta,
                'interpretation': interpret_effect_size(glass_delta)
            },
            'hedges_g': {
                'value': hedges_g,
                'interpretation': interpret_effect_size(hedges_g)
            },
            'common_language_effect_size': cles
        }
    
    def _compute_confidence_intervals(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Compute confidence intervals for means and differences"""
        alpha = 1 - self.config.confidence_level
        
        # Confidence intervals for means
        baseline_ci = stats.t.interval(
            self.config.confidence_level,
            len(baseline) - 1,
            loc=np.mean(baseline),
            scale=stats.sem(baseline)
        )
        
        insightspike_ci = stats.t.interval(
            self.config.confidence_level,
            len(insightspike) - 1,
            loc=np.mean(insightspike),
            scale=stats.sem(insightspike)
        )
        
        # Confidence interval for difference of means
        mean_diff = np.mean(insightspike) - np.mean(baseline)
        se_diff = np.sqrt(stats.sem(baseline)**2 + stats.sem(insightspike)**2)
        df_welch = ((stats.sem(baseline)**2 + stats.sem(insightspike)**2)**2 / 
                   (stats.sem(baseline)**4 / (len(baseline) - 1) + 
                    stats.sem(insightspike)**4 / (len(insightspike) - 1)))
        
        diff_ci = stats.t.interval(
            self.config.confidence_level,
            df_welch,
            loc=mean_diff,
            scale=se_diff
        )
        
        return {
            'baseline_mean': {
                'lower': baseline_ci[0],
                'upper': baseline_ci[1]
            },
            'insightspike_mean': {
                'lower': insightspike_ci[0],
                'upper': insightspike_ci[1]
            },
            'difference': {
                'value': mean_diff,
                'lower': diff_ci[0],
                'upper': diff_ci[1]
            }
        }
    
    def _compute_power_analysis(self, baseline: List[float], insightspike: List[float]) -> Dict:
        """Compute post-hoc power analysis"""
        from scipy.stats import norm
        
        # Effect size
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) + 
                             (len(insightspike) - 1) * np.var(insightspike, ddof=1)) / 
                            (len(baseline) + len(insightspike) - 2))
        effect_size = abs(np.mean(insightspike) - np.mean(baseline)) / pooled_std
        
        # Standard error
        se = pooled_std * np.sqrt(1/len(baseline) + 1/len(insightspike))
        
        # Critical value
        alpha = 1 - self.config.confidence_level
        critical_value = norm.ppf(1 - alpha/2)
        
        # Power calculation
        z_beta = (effect_size * pooled_std / se) - critical_value
        power = norm.cdf(z_beta)
        
        return {
            'effect_size': effect_size,
            'power': power,
            'sample_size_baseline': len(baseline),
            'sample_size_insightspike': len(insightspike),
            'alpha': alpha,
            'adequate_power': power >= 0.8
        }


class ResearchReportGenerator:
    """Generate comprehensive research reports"""
    
    def __init__(self, config: Optional[ResearchReportConfig] = None):
        if config is None:
            config = ResearchReportConfig()
        self.config = config
        self.analyzer = StatisticalAnalyzer(config)
        
        # Create output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, results: ExperimentResults) -> Path:
        """Generate complete research report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.config.output_directory / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Perform statistical analysis
        statistical_results = self._perform_complete_analysis(results)
        
        # Generate visualizations
        if self.config.include_visualizations:
            self._generate_all_visualizations(results, statistical_results, report_dir)
        
        # Generate report sections
        report_content = self._generate_report_content(results, statistical_results)
        
        # Save report as HTML
        html_path = self._save_html_report(report_content, report_dir)
        
        # Save report as PDF (if available)
        # pdf_path = self._save_pdf_report(report_content, report_dir)
        
        # Save raw data
        if self.config.include_raw_data:
            self._save_raw_data(results, statistical_results, report_dir)
        
        # Generate summary
        self._generate_summary(results, statistical_results, report_dir)
        
        return report_dir
    
    def _perform_complete_analysis(self, results: ExperimentResults) -> Dict[str, Any]:
        """Perform complete statistical analysis"""
        
        statistical_results = {}
        
        # Analyze each metric
        for metric in ['success_rates', 'episode_rewards', 'learning_efficiency']:
            if (metric in results.baseline_results and 
                metric in results.insightspike_results):
                
                baseline_data = results.baseline_results[metric]
                insightspike_data = results.insightspike_results[metric]
                
                # Convert to numpy arrays and handle different data structures
                if isinstance(baseline_data, list):
                    if len(baseline_data) > 0 and isinstance(baseline_data[0], (list, np.ndarray)):
                        # Handle episode-wise data - flatten or take means
                        baseline_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                       for episode in baseline_data]
                        insightspike_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                           for episode in insightspike_data]
                
                # Ensure we have 1D arrays
                baseline_data = np.array(baseline_data).flatten()
                insightspike_data = np.array(insightspike_data).flatten()
                
                analysis = self.analyzer.analyze_experiment(
                    baseline_data.tolist(), insightspike_data.tolist(), metric
                )
                statistical_results[metric] = analysis
        
        return statistical_results
    
    def _generate_all_visualizations(self, results: ExperimentResults,
                                   statistical_results: Dict[str, Any],
                                   output_dir: Path) -> None:
        """Generate all visualizations for the report"""
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'figure.dpi': self.config.figure_dpi,
            'savefig.dpi': self.config.figure_dpi,
            'savefig.bbox': 'tight'
        })
        
        # Generate specific visualizations
        self._create_comparison_plots(results, statistical_results, viz_dir)
        self._create_distribution_plots(results, statistical_results, viz_dir)
        self._create_learning_curves(results, viz_dir)
        self._create_statistical_summary(statistical_results, viz_dir)
        self._create_effect_size_plots(statistical_results, viz_dir)
    
    def _create_comparison_plots(self, results: ExperimentResults,
                               statistical_results: Dict[str, Any],
                               output_dir: Path) -> None:
        """Create comparison plots between baseline and InsightSpike"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['success_rates', 'episode_rewards']
        for i, metric in enumerate(metrics):
            if metric in results.baseline_results and metric in results.insightspike_results:
                
                baseline_data = results.baseline_results[metric]
                insightspike_data = results.insightspike_results[metric]
                
                # Handle episode-wise data and ensure proper flattening
                if isinstance(baseline_data, list) and len(baseline_data) > 0:
                    if isinstance(baseline_data[0], (list, np.ndarray)):
                        baseline_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                       for episode in baseline_data]
                        insightspike_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                           for episode in insightspike_data]
                
                # Convert to numpy arrays and flatten
                baseline_data = np.array(baseline_data).flatten()
                insightspike_data = np.array(insightspike_data).flatten()
                
                # Box plot
                ax = axes[i, 0]
                data_to_plot = [baseline_data, insightspike_data]
                box_plot = ax.boxplot(data_to_plot, labels=['Baseline', 'InsightSpike'], patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightcoral')
                box_plot['boxes'][1].set_facecolor('lightblue')
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_ylabel('Value')
                
                # Add statistical annotation
                if metric in statistical_results:
                    p_value = statistical_results[metric]['comparative']['t_test_unequal_var']['p_value']
                    ax.text(0.5, 0.95, f'p = {p_value:.4f}', 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Violin plot
                ax = axes[i, 1]
                parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Baseline', 'InsightSpike'])
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax.set_ylabel('Value')
        
        plt.tight_layout()
        self._save_figure(fig, output_dir / 'comparison_plots')
        plt.close()
    
    def _create_distribution_plots(self, results: ExperimentResults,
                                 statistical_results: Dict[str, Any],
                                 output_dir: Path) -> None:
        """Create distribution plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, metric in enumerate(['success_rates', 'episode_rewards']):
            if metric in results.baseline_results and metric in results.insightspike_results:
                
                baseline_data = results.baseline_results[metric]
                insightspike_data = results.insightspike_results[metric]
                
                # Handle episode-wise data and ensure proper flattening
                if isinstance(baseline_data, list) and len(baseline_data) > 0:
                    if isinstance(baseline_data[0], (list, np.ndarray)):
                        baseline_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                       for episode in baseline_data]
                        insightspike_data = [np.mean(episode) if hasattr(episode, '__len__') else episode 
                                           for episode in insightspike_data]
                
                # Convert to numpy arrays and flatten
                baseline_data = np.array(baseline_data).flatten()
                insightspike_data = np.array(insightspike_data).flatten()
                
                # Histogram
                ax = axes[i, 0]
                ax.hist(baseline_data, alpha=0.7, label='Baseline', color='red', bins=15)
                ax.hist(insightspike_data, alpha=0.7, label='InsightSpike', color='blue', bins=15)
                ax.set_title(f'{metric.replace("_", " ").title()} Histogram')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                
                # Q-Q plots
                ax = axes[i, 1]
                stats.probplot(baseline_data, dist="norm", plot=ax)
                ax.set_title(f'Baseline {metric} Q-Q Plot')
                
                ax = axes[i, 2]
                stats.probplot(insightspike_data, dist="norm", plot=ax)
                ax.set_title(f'InsightSpike {metric} Q-Q Plot')
        
        plt.tight_layout()
        self._save_figure(fig, output_dir / 'distribution_plots')
        plt.close()
    
    def _create_learning_curves(self, results: ExperimentResults, output_dir: Path) -> None:
        """Create learning curves"""
        
        if 'episode_rewards' in results.baseline_results and 'episode_rewards' in results.insightspike_results:
            
            baseline_rewards = results.baseline_results['episode_rewards']
            insightspike_rewards = results.insightspike_results['episode_rewards']
            
            # Calculate means and confidence intervals
            if isinstance(baseline_rewards, list) and len(baseline_rewards) > 0:
                if isinstance(baseline_rewards[0], (list, np.ndarray)):
                    # Multi-trial episode data
                    episodes = range(len(baseline_rewards[0]))
                    baseline_means = np.mean(baseline_rewards, axis=0)
                    baseline_stds = np.std(baseline_rewards, axis=0)
                    baseline_cis = 1.96 * baseline_stds / np.sqrt(len(baseline_rewards))
                    
                    insightspike_means = np.mean(insightspike_rewards, axis=0)
                    insightspike_stds = np.std(insightspike_rewards, axis=0)
                    insightspike_cis = 1.96 * insightspike_stds / np.sqrt(len(insightspike_rewards))
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    ax.plot(episodes, baseline_means, label='Baseline', color='red', linewidth=2)
                    ax.fill_between(episodes, 
                                   baseline_means - baseline_cis, 
                                   baseline_means + baseline_cis,
                                   color='red', alpha=0.2)
                    
                    ax.plot(episodes, insightspike_means, label='InsightSpike', color='blue', linewidth=2)
                    ax.fill_between(episodes,
                                   insightspike_means - insightspike_cis,
                                   insightspike_means + insightspike_cis,
                                   color='blue', alpha=0.2)
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Average Reward')
                    ax.set_title('Learning Curves with 95% Confidence Intervals')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    self._save_figure(fig, output_dir / 'learning_curves')
                    plt.close()
                else:
                    # Single trial data - create simple plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    episodes = range(len(baseline_rewards))
                    ax.plot(episodes, baseline_rewards, label='Baseline', color='red', linewidth=2)
                    ax.plot(episodes, insightspike_rewards, label='InsightSpike', color='blue', linewidth=2)
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Reward')
                    ax.set_title('Learning Curves')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    self._save_figure(fig, output_dir / 'learning_curves')
                    plt.close()
    
    def _create_statistical_summary(self, statistical_results: Dict[str, Any], output_dir: Path) -> None:
        """Create statistical summary visualization"""
        
        metrics = list(statistical_results.keys())
        if not metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # P-values plot
        ax = axes[0, 0]
        p_values = []
        metric_names = []
        for metric in metrics:
            if 'comparative' in statistical_results[metric]:
                p_val = statistical_results[metric]['comparative']['t_test_unequal_var']['p_value']
                p_values.append(p_val)
                metric_names.append(metric.replace('_', ' ').title())
        
        if p_values:
            bars = ax.bar(metric_names, [-np.log10(p) for p in p_values])
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')
            ax.set_ylabel('-log₁₀(p-value)')
            ax.set_title('Statistical Significance Tests')
            ax.legend()
            
            # Color bars based on significance
            for bar, p_val in zip(bars, p_values):
                if p_val < 0.05:
                    bar.set_color('green')
                else:
                    bar.set_color('gray')
        
        # Effect sizes plot
        ax = axes[0, 1]
        effect_sizes = []
        for metric in metrics:
            if 'effect_size' in statistical_results[metric]:
                effect_size = statistical_results[metric]['effect_size']['cohens_d']['value']
                effect_sizes.append(effect_size)
        
        if effect_sizes:
            bars = ax.bar(metric_names, effect_sizes)
            ax.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.7, label='Small effect')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large effect')
            ax.set_ylabel("Cohen's d")
            ax.set_title('Effect Sizes')
            ax.legend()
        
        # Power analysis
        ax = axes[1, 0]
        powers = []
        for metric in metrics:
            if 'power_analysis' in statistical_results[metric]:
                power = statistical_results[metric]['power_analysis']['power']
                powers.append(power)
        
        if powers:
            bars = ax.bar(metric_names, powers)
            ax.axhline(y=0.8, color='red', linestyle='--', label='Adequate power')
            ax.set_ylabel('Statistical Power')
            ax.set_title('Post-hoc Power Analysis')
            ax.legend()
            
            # Color bars based on adequate power
            for bar, power in zip(bars, powers):
                if power >= 0.8:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        # Sample sizes
        ax = axes[1, 1]
        sample_sizes_baseline = []
        sample_sizes_insightspike = []
        for metric in metrics:
            if 'sample_sizes' in statistical_results[metric]:
                sample_sizes_baseline.append(statistical_results[metric]['sample_sizes']['baseline'])
                sample_sizes_insightspike.append(statistical_results[metric]['sample_sizes']['insightspike'])
        
        if sample_sizes_baseline:
            x = np.arange(len(metric_names))
            width = 0.35
            ax.bar(x - width/2, sample_sizes_baseline, width, label='Baseline', alpha=0.8)
            ax.bar(x + width/2, sample_sizes_insightspike, width, label='InsightSpike', alpha=0.8)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Sample Size')
            ax.set_title('Sample Sizes by Metric')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.legend()
        
        plt.tight_layout()
        self._save_figure(fig, output_dir / 'statistical_summary')
        plt.close()
    
    def _create_effect_size_plots(self, statistical_results: Dict[str, Any], output_dir: Path) -> None:
        """Create effect size visualization"""
        
        metrics = list(statistical_results.keys())
        if not metrics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Effect size comparison
        cohens_d = []
        hedges_g = []
        metric_names = []
        
        for metric in metrics:
            if 'effect_size' in statistical_results[metric]:
                cohens_d.append(statistical_results[metric]['effect_size']['cohens_d']['value'])
                hedges_g.append(statistical_results[metric]['effect_size']['hedges_g']['value'])
                metric_names.append(metric.replace('_', ' ').title())
        
        if cohens_d:
            x = np.arange(len(metric_names))
            width = 0.35
            
            ax1.bar(x - width/2, cohens_d, width, label="Cohen's d", alpha=0.8)
            ax1.bar(x + width/2, hedges_g, width, label="Hedge's g", alpha=0.8)
            
            # Add effect size thresholds
            ax1.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.7, label='Small')
            ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium') 
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Effect Size')
            ax1.set_title('Effect Size Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metric_names)
            ax1.legend()
        
        # Common Language Effect Size
        cles_values = []
        for metric in metrics:
            if 'effect_size' in statistical_results[metric]:
                cles = statistical_results[metric]['effect_size']['common_language_effect_size']
                cles_values.append(cles)
        
        if cles_values:
            bars = ax2.bar(metric_names, cles_values, alpha=0.8, color='purple')
            ax2.axhline(y=0.5, color='red', linestyle='--', label='No effect')
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Common Language Effect Size')
            ax2.set_title('Probability of Superiority')
            ax2.legend()
            
            # Add percentage labels on bars
            for bar, cles in zip(bars, cles_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{cles:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, output_dir / 'effect_sizes')
        plt.close()
    
    def _save_figure(self, fig, filename: Path) -> None:
        """Save figure in specified format(s)"""
        if self.config.figure_format in ['png', 'both']:
            fig.savefig(f"{filename}.png", dpi=self.config.figure_dpi, bbox_inches='tight')
        if self.config.figure_format in ['pdf', 'both']:
            fig.savefig(f"{filename}.pdf", dpi=self.config.figure_dpi, bbox_inches='tight')
    
    def _generate_report_content(self, results: ExperimentResults,
                               statistical_results: Dict[str, Any]) -> str:
        """Generate complete report content"""
        
        # Use Jinja2 template for report generation
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>InsightSpike-AI Research Report</title>
    <style>
        body { font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; }
        .abstract { background-color: #f8f9fa; padding: 20px; border-left: 4px solid #3498db; }
        .results-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .results-table th { background-color: #f2f2f2; }
        .significant { color: #27ae60; font-weight: bold; }
        .not-significant { color: #e74c3c; }
        .effect-size { font-style: italic; }
        .figure-caption { font-style: italic; color: #7f8c8d; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>InsightSpike-AI: Enhanced Learning Through Insight Detection</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    
    {% if include_abstract %}
    <h2>Abstract</h2>
    <div class="abstract">
        <p>This report presents a comprehensive evaluation of InsightSpike-AI, an enhanced reinforcement learning 
        system that incorporates insight detection mechanisms to improve learning efficiency. We compare 
        InsightSpike-AI against baseline Q-learning across multiple metrics including success rate, learning 
        speed, and exploration efficiency. Our results demonstrate {{ summary.main_finding }}.</p>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            {% for finding in summary.key_findings %}
            <li>{{ finding }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    {% if include_methodology %}
    <h2>Methodology</h2>
    <h3>Experimental Design</h3>
    <p>We conducted a controlled comparison between baseline Q-learning and InsightSpike-AI using the following configuration:</p>
    <ul>
        <li><strong>Number of trials:</strong> {{ config.num_trials }}</li>
        <li><strong>Episodes per trial:</strong> {{ config.num_episodes }}</li>
        <li><strong>Environment:</strong> {{ config.maze_size }}x{{ config.maze_size }} maze with {{ config.wall_density }} wall density</li>
        <li><strong>Statistical significance threshold:</strong> α = {{ config.significance_threshold }}</li>
        <li><strong>Confidence level:</strong> {{ config.confidence_level * 100 }}%</li>
    </ul>
    
    <h3>Metrics</h3>
    <p>We evaluated both systems using the following metrics:</p>
    <ul>
        <li><strong>Success Rate:</strong> Proportion of episodes reaching the goal</li>
        <li><strong>Learning Efficiency:</strong> Rate of improvement in performance</li>
        <li><strong>Exploration Efficiency:</strong> Quality of state space exploration</li>
    </ul>
    {% endif %}
    
    {% if include_results %}
    <h2>Results</h2>
    
    <h3>Statistical Analysis Summary</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Baseline Mean (SD)</th>
                <th>InsightSpike Mean (SD)</th>
                <th>p-value</th>
                <th>Effect Size (Cohen's d)</th>
                <th>Interpretation</th>
            </tr>
        </thead>
        <tbody>
            {% for metric, analysis in statistical_results.items() %}
            <tr>
                <td>{{ metric.replace('_', ' ').title() }}</td>
                <td>{{ "%.3f (%.3f)"|format(analysis.descriptive.baseline.mean, analysis.descriptive.baseline.std) }}</td>
                <td>{{ "%.3f (%.3f)"|format(analysis.descriptive.insightspike.mean, analysis.descriptive.insightspike.std) }}</td>
                <td class="{{ 'significant' if analysis.comparative.t_test_unequal_var.p_value < 0.05 else 'not-significant' }}">
                    {{ "%.4f"|format(analysis.comparative.t_test_unequal_var.p_value) }}
                </td>
                <td class="effect-size">{{ "%.3f"|format(analysis.effect_size.cohens_d.value) }}</td>
                <td>{{ analysis.effect_size.cohens_d.interpretation.title() }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <h3>Detailed Findings</h3>
    {% for metric, analysis in statistical_results.items() %}
    <h4>{{ metric.replace('_', ' ').title() }}</h4>
    <p><strong>Descriptive Statistics:</strong></p>
    <ul>
        <li>Baseline: M = {{ "%.3f"|format(analysis.descriptive.baseline.mean) }}, 
            SD = {{ "%.3f"|format(analysis.descriptive.baseline.std) }}</li>
        <li>InsightSpike: M = {{ "%.3f"|format(analysis.descriptive.insightspike.mean) }}, 
            SD = {{ "%.3f"|format(analysis.descriptive.insightspike.std) }}</li>
    </ul>
    
    <p><strong>Statistical Test:</strong> 
    {% if analysis.comparative.t_test_unequal_var.p_value < 0.05 %}
    <span class="significant">Significant difference found</span>
    {% else %}
    <span class="not-significant">No significant difference</span>
    {% endif %}
    (t = {{ "%.3f"|format(analysis.comparative.t_test_unequal_var.statistic) }}, 
    p = {{ "%.4f"|format(analysis.comparative.t_test_unequal_var.p_value) }}).</p>
    
    <p><strong>Effect Size:</strong> Cohen's d = {{ "%.3f"|format(analysis.effect_size.cohens_d.value) }} 
    ({{ analysis.effect_size.cohens_d.interpretation }}).</p>
    
    <p><strong>Confidence Interval for Difference:</strong> 
    [{{ "%.3f"|format(analysis.confidence_intervals.difference.lower) }}, 
    {{ "%.3f"|format(analysis.confidence_intervals.difference.upper) }}]</p>
    {% endfor %}
    {% endif %}
    
    {% if include_discussion %}
    <h2>Discussion</h2>
    <h3>Interpretation of Results</h3>
    <p>{{ discussion.interpretation }}</p>
    
    <h3>Implications</h3>
    <p>{{ discussion.implications }}</p>
    
    <h3>Limitations</h3>
    <p>{{ discussion.limitations }}</p>
    
    <h3>Future Work</h3>
    <p>{{ discussion.future_work }}</p>
    {% endif %}
    
    <h2>Conclusion</h2>
    <p>{{ conclusion }}</p>
    
    {% if include_appendix %}
    <h2>Appendix</h2>
    <h3>Statistical Test Details</h3>
    {% for metric, analysis in statistical_results.items() %}
    <h4>{{ metric.replace('_', ' ').title() }}</h4>
    <p><strong>Normality Tests:</strong></p>
    <ul>
        <li>Baseline Shapiro-Wilk: W = {{ "%.4f"|format(analysis.normality.shapiro_wilk.baseline.statistic) }}, 
            p = {{ "%.4f"|format(analysis.normality.shapiro_wilk.baseline.p_value) }}</li>
        <li>InsightSpike Shapiro-Wilk: W = {{ "%.4f"|format(analysis.normality.shapiro_wilk.insightspike.statistic) }}, 
            p = {{ "%.4f"|format(analysis.normality.shapiro_wilk.insightspike.p_value) }}</li>
    </ul>
    
    <p><strong>Additional Tests:</strong></p>
    <ul>
        <li>Mann-Whitney U: U = {{ "%.2f"|format(analysis.comparative.mann_whitney_u.statistic) }}, 
            p = {{ "%.4f"|format(analysis.comparative.mann_whitney_u.p_value) }}</li>
        <li>Power Analysis: Power = {{ "%.3f"|format(analysis.power_analysis.power) }} 
            ({{ 'Adequate' if analysis.power_analysis.adequate_power else 'Inadequate' }})</li>
    </ul>
    {% endfor %}
    {% endif %}
    
    <footer>
        <p><em>Report generated automatically by InsightSpike-AI Research Report Generator</em></p>
    </footer>
</body>
</html>
        """
        
        template = Template(template_str)
        
        # Prepare template variables
        summary = self._generate_summary_stats(statistical_results)
        discussion = self._generate_discussion(statistical_results)
        conclusion = self._generate_conclusion(statistical_results)
        
        template_vars = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': results.experimental_config,
            'statistical_results': statistical_results,
            'summary': summary,
            'discussion': discussion,
            'conclusion': conclusion,
            'include_abstract': self.config.include_abstract,
            'include_methodology': self.config.include_methodology,
            'include_results': self.config.include_results,
            'include_discussion': self.config.include_discussion,
            'include_appendix': self.config.include_appendix
        }
        
        return template.render(**template_vars)
    
    def _generate_summary_stats(self, statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the report"""
        
        significant_results = []
        key_findings = []
        
        for metric, analysis in statistical_results.items():
            p_value = analysis['comparative']['t_test_unequal_var']['p_value']
            effect_size = analysis['effect_size']['cohens_d']['value']
            
            if p_value < 0.05:
                significant_results.append(metric)
                direction = "improvement" if effect_size > 0 else "decrease"
                key_findings.append(
                    f"{metric.replace('_', ' ').title()}: {direction} with {analysis['effect_size']['cohens_d']['interpretation']} effect size"
                )
        
        if significant_results:
            main_finding = f"significant improvements in {len(significant_results)} out of {len(statistical_results)} metrics"
        else:
            main_finding = "no statistically significant differences between baseline and InsightSpike-AI"
        
        return {
            'main_finding': main_finding,
            'key_findings': key_findings,
            'significant_metrics': significant_results
        }
    
    def _generate_discussion(self, statistical_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate discussion section"""
        
        return {
            'interpretation': "The results provide evidence for the effectiveness of insight detection mechanisms in reinforcement learning. The observed improvements suggest that meta-cognitive awareness can enhance learning efficiency.",
            'implications': "These findings have important implications for the development of more sophisticated AI systems that can learn more efficiently and adaptively in complex environments.",
            'limitations': "This study was conducted in a controlled maze environment. Further research is needed to evaluate performance in more complex, real-world scenarios.",
            'future_work': "Future research should explore the scalability of insight detection mechanisms and their applicability to other domains such as robotics and natural language processing."
        }
    
    def _generate_conclusion(self, statistical_results: Dict[str, Any]) -> str:
        """Generate conclusion"""
        
        significant_count = sum(1 for analysis in statistical_results.values() 
                              if analysis['comparative']['t_test_unequal_var']['p_value'] < 0.05)
        
        if significant_count > 0:
            return f"Our comprehensive evaluation demonstrates that InsightSpike-AI significantly outperforms baseline Q-learning in {significant_count} out of {len(statistical_results)} evaluated metrics. These results support the hypothesis that incorporating insight detection mechanisms can enhance reinforcement learning performance."
        else:
            return "While InsightSpike-AI shows promising trends, the statistical analysis did not reveal significant differences from baseline performance in this evaluation. Further investigation with larger sample sizes or different experimental conditions may be warranted."
    
    def _save_html_report(self, content: str, output_dir: Path) -> Path:
        """Save HTML report"""
        html_path = output_dir / "research_report.html"
        html_path.write_text(content, encoding='utf-8')
        return html_path
    
    def _save_raw_data(self, results: ExperimentResults,
                      statistical_results: Dict[str, Any],
                      output_dir: Path) -> None:
        """Save raw data and statistical results"""
        
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save experimental results
        with open(data_dir / "experimental_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.baseline_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[f"baseline_{key}"] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_results[f"baseline_{key}"] = [arr.tolist() for arr in value]
                else:
                    serializable_results[f"baseline_{key}"] = value
            
            for key, value in results.insightspike_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[f"insightspike_{key}"] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_results[f"insightspike_{key}"] = [arr.tolist() for arr in value]
                else:
                    serializable_results[f"insightspike_{key}"] = value
            
            json.dump(serializable_results, f, indent=2)
        
        # Save statistical results
        with open(data_dir / "statistical_analysis.json", 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        # Save configuration
        with open(data_dir / "experiment_config.json", 'w') as f:
            json.dump(results.experimental_config, f, indent=2)
    
    def _generate_summary(self, results: ExperimentResults,
                         statistical_results: Dict[str, Any],
                         output_dir: Path) -> None:
        """Generate executive summary"""
        
        summary_path = output_dir / "executive_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("InsightSpike-AI Research Report - Executive Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic info
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {results.experiment_name}\n\n")
            
            # Key metrics
            f.write("Key Results:\n")
            f.write("-" * 20 + "\n")
            
            for metric, analysis in statistical_results.items():
                baseline_mean = analysis['descriptive']['baseline']['mean']
                insightspike_mean = analysis['descriptive']['insightspike']['mean']
                p_value = analysis['comparative']['t_test_unequal_var']['p_value']
                effect_size = analysis['effect_size']['cohens_d']['value']
                
                improvement = ((insightspike_mean - baseline_mean) / baseline_mean) * 100
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Baseline: {baseline_mean:.3f}\n")
                f.write(f"  InsightSpike: {insightspike_mean:.3f}\n")
                f.write(f"  Improvement: {improvement:+.1f}%\n")
                f.write(f"  Significance: {significance} (p = {p_value:.4f})\n")
                f.write(f"  Effect Size: {effect_size:.3f} ({analysis['effect_size']['cohens_d']['interpretation']})\n\n")
            
            # Overall conclusion
            significant_count = sum(1 for analysis in statistical_results.values() 
                                  if analysis['comparative']['t_test_unequal_var']['p_value'] < 0.05)
            
            f.write("Conclusion:\n")
            f.write("-" * 20 + "\n")
            f.write(f"InsightSpike-AI showed significant improvements in {significant_count}/{len(statistical_results)} metrics.\n")
            
            if significant_count > len(statistical_results) / 2:
                f.write("Overall assessment: POSITIVE - InsightSpike-AI demonstrates clear advantages.\n")
            elif significant_count > 0:
                f.write("Overall assessment: MIXED - Some benefits observed, further investigation recommended.\n")
            else:
                f.write("Overall assessment: INCONCLUSIVE - No significant differences detected.\n")


# Example usage
def create_sample_research_report():
    """Create a sample research report with synthetic data"""
    
    # Generate synthetic experimental results
    np.random.seed(42)
    
    baseline_results = {
        'success_rates': np.random.beta(2, 3, 30),
        'episode_rewards': [np.random.normal(0, 1, 100).cumsum() for _ in range(30)],
        'learning_efficiency': np.random.gamma(2, 2, 30)
    }
    
    insightspike_results = {
        'success_rates': np.random.beta(3, 2, 30),
        'episode_rewards': [np.random.normal(0.5, 1, 100).cumsum() for _ in range(30)],
        'learning_efficiency': np.random.gamma(3, 2, 30)
    }
    
    experimental_config = {
        'num_trials': 30,
        'num_episodes': 100,
        'maze_size': 10,
        'wall_density': 0.25,
        'significance_threshold': 0.05,
        'confidence_level': 0.95
    }
    
    # Create experiment results object
    results = ExperimentResults(
        experiment_name="InsightSpike-AI Evaluation",
        baseline_results=baseline_results,
        insightspike_results=insightspike_results,
        experimental_config=experimental_config
    )
    
    # Generate report
    config = ResearchReportConfig(
        output_directory=Path("experiments/outputs/sample_reports"),
        include_visualizations=True,
        include_statistical_analysis=True
    )
    
    generator = ResearchReportGenerator(config)
    report_dir = generator.generate_report(results)
    
    print(f"Sample research report generated in: {report_dir}")
    return report_dir


if __name__ == "__main__":
    # Create sample research report
    report_dir = create_sample_research_report()
    print("Research report generation completed successfully!")
