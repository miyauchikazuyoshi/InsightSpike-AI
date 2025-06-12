"""
Improved Experimental Comparison Analysis
========================================

This module provides unbiased comparison analysis with proper controls:
- Multiple baseline implementations
- Statistical rigor with correction for multiple testing
- Effect size reporting with confidence intervals
- Cross-validation and robustness checks
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ComparisonMetrics:
    """Metrics for algorithm comparison"""
    name: str
    mean_performance: float
    std_performance: float
    median_performance: float
    q25: float
    q75: float
    confidence_interval_95: Tuple[float, float]
    sample_size: int

class ImprovedComparisonAnalysis:
    """Enhanced comparison analysis with bias reduction"""
    
    def __init__(self, output_dir: str = "experiments/outputs/improved_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache = {}
    
    def analyze_results(self, results_file: str) -> Dict[str, Any]:
        """Analyze experimental results with statistical rigor"""
        
        # Load results
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        analysis = {
            'summary_statistics': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'robustness_checks': {},
            'visualizations_created': []
        }
        
        # Process each environment
        for env_name, env_results in data['results'].items():
            env_analysis = self._analyze_environment(env_name, env_results)
            analysis['summary_statistics'][env_name] = env_analysis['summary']
            analysis['statistical_tests'][env_name] = env_analysis['tests']
            analysis['effect_sizes'][env_name] = env_analysis['effects']
            analysis['robustness_checks'][env_name] = env_analysis['robustness']
            
            # Create visualizations
            viz_files = self._create_visualizations(env_name, env_results)
            analysis['visualizations_created'].extend(viz_files)
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
    
    def _analyze_environment(self, env_name: str, env_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze results for a specific environment"""
        
        algorithms = list(env_results.keys())
        n_algorithms = len(algorithms)
        
        # Summary statistics
        summary = {}
        for alg_name, results in env_results.items():
            data = np.array(results)
            
            # Calculate confidence interval
            sem = stats.sem(data)
            ci_95 = stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=sem)
            
            summary[alg_name] = ComparisonMetrics(
                name=alg_name,
                mean_performance=np.mean(data),
                std_performance=np.std(data, ddof=1),
                median_performance=np.median(data),
                q25=np.percentile(data, 25),
                q75=np.percentile(data, 75),
                confidence_interval_95=ci_95,
                sample_size=len(data)
            ).__dict__
        
        # Statistical tests with multiple comparison correction
        tests = self._perform_statistical_tests(env_results)
        
        # Effect sizes
        effects = self._calculate_effect_sizes(env_results)
        
        # Robustness checks
        robustness = self._robustness_checks(env_results)
        
        return {
            'summary': summary,
            'tests': tests,
            'effects': effects,
            'robustness': robustness
        }
    
    def _perform_statistical_tests(self, env_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical tests with proper corrections"""
        
        algorithms = list(env_results.keys())
        n_comparisons = len(algorithms) * (len(algorithms) - 1) // 2
        
        # Bonferroni correction
        alpha = 0.05
        corrected_alpha = alpha / n_comparisons
        
        tests = {
            'alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n_comparisons,
            'pairwise_tests': {},
            'omnibus_test': None
        }
        
        # Omnibus test (Kruskal-Wallis)
        all_data = []
        all_groups = []
        for alg_name, results in env_results.items():
            all_data.extend(results)
            all_groups.extend([alg_name] * len(results))
        
        if len(algorithms) > 2:
            # Kruskal-Wallis test
            groups = [env_results[alg] for alg in algorithms]
            h_stat, p_value = stats.kruskal(*groups)
            tests['omnibus_test'] = {
                'test': 'Kruskal-Wallis',
                'statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        
        # Pairwise tests
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    data1 = np.array(env_results[alg1])
                    data2 = np.array(env_results[alg2])
                    
                    # Normality tests
                    normal1 = stats.shapiro(data1)[1] > 0.05 if len(data1) >= 3 else False
                    normal2 = stats.shapiro(data2)[1] > 0.05 if len(data2) >= 3 else False
                    
                    # Choose appropriate test
                    if normal1 and normal2 and len(data1) > 5 and len(data2) > 5:
                        # Welch's t-test
                        stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                        test_name = "Welch's t-test"
                    else:
                        # Mann-Whitney U test
                        stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test_name = "Mann-Whitney U"
                    
                    # Bootstrap confidence interval for difference
                    diff_ci = self._bootstrap_difference_ci(data1, data2)
                    
                    comparison_key = f"{alg1} vs {alg2}"
                    tests['pairwise_tests'][comparison_key] = {
                        'test': test_name,
                        'statistic': stat,
                        'p_value': p_val,
                        'significant_uncorrected': p_val < alpha,
                        'significant_corrected': p_val < corrected_alpha,
                        'difference_ci_95': diff_ci,
                        'normality': {'alg1': normal1, 'alg2': normal2}
                    }
        
        return tests
    
    def _calculate_effect_sizes(self, env_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate effect sizes with confidence intervals"""
        
        algorithms = list(env_results.keys())
        effects = {}
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    data1 = np.array(env_results[alg1])
                    data2 = np.array(env_results[alg2])
                    
                    # Cohen's d
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                       (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                    
                    # Glass's delta (using control group std)
                    glass_delta = (np.mean(data1) - np.mean(data2)) / np.std(data2, ddof=1) if np.std(data2, ddof=1) > 0 else 0
                    
                    # Hedges' g (bias-corrected)
                    j_correction = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
                    hedges_g = cohens_d * j_correction
                    
                    # Common Language Effect Size
                    cles = stats.mannwhitneyu(data1, data2)[0] / (len(data1) * len(data2))
                    
                    # Bootstrap confidence interval for Cohen's d
                    d_ci = self._bootstrap_cohens_d_ci(data1, data2)
                    
                    comparison_key = f"{alg1} vs {alg2}"
                    effects[comparison_key] = {
                        'cohens_d': cohens_d,
                        'cohens_d_ci_95': d_ci,
                        'hedges_g': hedges_g,
                        'glass_delta': glass_delta,
                        'cles': cles,
                        'interpretation': self._interpret_effect_size(abs(cohens_d)),
                        'mean_difference': np.mean(data1) - np.mean(data2),
                        'percent_improvement': ((np.mean(data1) - np.mean(data2)) / np.mean(data2) * 100) if np.mean(data2) != 0 else 0
                    }
        
        return effects
    
    def _robustness_checks(self, env_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform robustness checks"""
        
        robustness = {
            'outlier_analysis': {},
            'sensitivity_analysis': {},
            'assumption_checks': {}
        }
        
        for alg_name, results in env_results.items():
            data = np.array(results)
            
            # Outlier detection using IQR method
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            robustness['outlier_analysis'][alg_name] = {
                'n_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100,
                'outlier_values': outliers.tolist(),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
            # Sensitivity analysis (remove outliers)
            if len(outliers) > 0:
                clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
                sensitivity_impact = abs(np.mean(clean_data) - np.mean(data)) / np.mean(data) * 100
            else:
                sensitivity_impact = 0
            
            robustness['sensitivity_analysis'][alg_name] = {
                'outlier_impact_percent': sensitivity_impact,
                'robust_mean': np.mean(data[(data >= lower_bound) & (data <= upper_bound)]) if len(outliers) > 0 else np.mean(data)
            }
            
            # Assumption checks
            if len(data) >= 3:
                normality_p = stats.shapiro(data)[1]
            else:
                normality_p = None
            
            robustness['assumption_checks'][alg_name] = {
                'normality_p_value': normality_p,
                'is_normal': normality_p > 0.05 if normality_p else None,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        
        return robustness
    
    def _bootstrap_difference_ci(self, data1: np.ndarray, data2: np.ndarray, 
                                n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for difference in means"""
        
        differences = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            differences.append(np.mean(sample1) - np.mean(sample2))
        
        alpha = 1 - confidence
        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def _bootstrap_cohens_d_ci(self, data1: np.ndarray, data2: np.ndarray,
                              n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for Cohen's d"""
        
        cohens_ds = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            
            pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) + 
                                (len(sample2) - 1) * np.var(sample2, ddof=1)) / 
                               (len(sample1) + len(sample2) - 2))
            
            if pooled_std > 0:
                d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
                cohens_ds.append(d)
        
        if len(cohens_ds) == 0:
            return (0, 0)
        
        alpha = 1 - confidence
        lower = np.percentile(cohens_ds, 100 * alpha / 2)
        upper = np.percentile(cohens_ds, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_visualizations(self, env_name: str, env_results: Dict[str, List[float]]) -> List[str]:
        """Create improved visualizations"""
        
        viz_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Box plot with individual points
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        all_data = []
        all_labels = []
        for alg_name, results in env_results.items():
            all_data.extend(results)
            all_labels.extend([alg_name] * len(results))
        
        df = pd.DataFrame({'Algorithm': all_labels, 'Performance': all_data})
        
        # Create box plot
        box_plot = sns.boxplot(data=df, x='Algorithm', y='Performance', ax=ax)
        
        # Add individual points
        sns.stripplot(data=df, x='Algorithm', y='Performance', 
                     size=4, alpha=0.6, ax=ax)
        
        # Add mean markers
        for i, (alg_name, results) in enumerate(env_results.items()):
            mean_val = np.mean(results)
            ax.scatter(i, mean_val, marker='D', s=100, color='red', 
                      label='Mean' if i == 0 else '', zorder=10)
        
        ax.set_title(f'Algorithm Performance Comparison - {env_name.title()} Environment', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        box_plot_file = self.output_dir / f'{env_name}_boxplot_comparison.png'
        plt.savefig(box_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(box_plot_file))
        
        # 2. Confidence interval plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        algorithms = list(env_results.keys())
        means = []
        cis = []
        
        for alg_name in algorithms:
            data = np.array(env_results[alg_name])
            mean_val = np.mean(data)
            sem = stats.sem(data)
            ci = stats.t.interval(0.95, len(data)-1, loc=mean_val, scale=sem)
            
            means.append(mean_val)
            cis.append((ci[1] - mean_val, mean_val - ci[0]))  # Upper and lower errors
        
        # Create error bar plot
        x_pos = np.arange(len(algorithms))
        bars = ax.bar(x_pos, means, capsize=5, alpha=0.7)
        ax.errorbar(x_pos, means, yerr=np.array(cis).T, 
                   fmt='none', capsize=5, capthick=2, color='black')
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Mean Performance with 95% Confidence Intervals - {env_name.title()}',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        plt.tight_layout()
        
        ci_plot_file = self.output_dir / f'{env_name}_confidence_intervals.png'
        plt.savefig(ci_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(ci_plot_file))
        
        return viz_files
    
    def _save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis results"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        analysis_file = self.output_dir / f'improved_analysis_{timestamp}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Analysis saved to {analysis_file}")

def demonstrate_improved_analysis():
    """Demonstrate the improved analysis framework"""
    
    print("Running Improved Experimental Analysis...")
    print("="*60)
    
    # First run the bias-corrected evaluation
    from bias_corrected_evaluation_framework import run_bias_corrected_evaluation
    
    print("Step 1: Running bias-corrected evaluation...")
    eval_results = run_bias_corrected_evaluation()
    
    print("\nStep 2: Performing improved statistical analysis...")
    
    # Find the most recent results file
    import glob
    output_dir = Path("experiments/outputs/bias_corrected")
    result_files = glob.glob(str(output_dir / "evaluation_results_*.json"))
    
    if result_files:
        latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
        
        # Run improved analysis
        analyzer = ImprovedComparisonAnalysis()
        analysis_results = analyzer.analyze_results(latest_file)
        
        print(f"\nAnalysis completed. Results saved to: {analyzer.output_dir}")
        
        # Print summary
        print("\nImproved Analysis Summary:")
        print("-" * 40)
        
        for env_name in analysis_results['summary_statistics']:
            print(f"\nEnvironment: {env_name}")
            stats_data = analysis_results['summary_statistics'][env_name]
            
            for alg_name, metrics in stats_data.items():
                ci_lower, ci_upper = metrics['confidence_interval_95']
                print(f"  {alg_name:20}: {metrics['mean_performance']:.3f} "
                      f"[{ci_lower:.3f}, {ci_upper:.3f}] (n={metrics['sample_size']})")
        
        return analysis_results
    else:
        print("No evaluation results found. Please run the evaluation first.")
        return None

if __name__ == "__main__":
    results = demonstrate_improved_analysis()
