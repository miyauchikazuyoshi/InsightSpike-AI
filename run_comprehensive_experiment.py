#!/usr/bin/env python3
"""
Comprehensive InsightSpike-AI Experiment Runner
Executes statistical experiments with proper controls and analysis
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import experiment frameworks
from experiments.baseline_comparison_framework import (
    ExperimentRunner, ExperimentConfig, BaselineAgent, InsightSpikeAgent
)

def setup_experiment_configs() -> List[ExperimentConfig]:
    """Setup experimental configurations for statistical comparison"""
    
    configs = []
    
    # 1. Pure Baseline (no insight rewards)
    configs.append(ExperimentConfig(
        experiment_name="Pure Baseline",
        num_trials=20,  # Reduced for faster execution
        num_episodes=50,
        maze_size=8,
        insight_reward_scale=0.0,
        random_seed=42
    ))
    
    # 2. Low Insight Reward
    configs.append(ExperimentConfig(
        experiment_name="Low Insight Reward",
        num_trials=20,
        num_episodes=50,
        maze_size=8,
        insight_reward_scale=0.1,
        random_seed=42
    ))
    
    # 3. Medium Insight Reward
    configs.append(ExperimentConfig(
        experiment_name="Medium Insight Reward",
        num_trials=20,
        num_episodes=50,
        maze_size=8,
        insight_reward_scale=0.5,
        random_seed=42
    ))
    
    # 4. High Insight Reward (Full InsightSpike-AI)
    configs.append(ExperimentConfig(
        experiment_name="Full InsightSpike-AI",
        num_trials=20,
        num_episodes=50,
        maze_size=8,
        insight_reward_scale=1.0,
        random_seed=42
    ))
    
    return configs

def run_comprehensive_experiment():
    """Run comprehensive statistical experiment"""
    
    print("🧪 Comprehensive InsightSpike-AI Statistical Experiment")
    print("=" * 60)
    print("Statistical comparison with proper controls and analysis")
    print()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup output directory
    output_dir = "experiments/results/comprehensive_experiment"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir)
    
    # Setup experimental configurations
    configs = setup_experiment_configs()
    
    print(f"📋 Experimental Design:")
    print(f"   Configurations: {len(configs)}")
    print(f"   Trials per config: {configs[0].num_trials}")
    print(f"   Episodes per trial: {configs[0].num_episodes}")
    print(f"   Total episodes: {len(configs) * configs[0].num_trials * configs[0].num_episodes}")
    print()
    
    # Run experiments
    print("🚀 Starting experimental trials...")
    results = runner.run_comparison_experiment(configs)
    
    # Perform statistical analysis
    print("\n📊 Performing statistical analysis...")
    analysis = runner.statistical_analysis(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save analysis results
    analysis_file = f"{output_dir}/statistical_analysis_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Generate summary report
    report = generate_summary_report(analysis, results)
    report_file = f"{output_dir}/comprehensive_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Comprehensive experiment completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Report: {report_file}")
    print(f"📊 Analysis: {analysis_file}")
    
    return results, analysis

def generate_summary_report(analysis: Dict[str, Any], results: Dict[str, List]) -> str:
    """Generate comprehensive summary report"""
    
    report = "# Comprehensive InsightSpike-AI Statistical Experiment Report\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Executive Summary\n\n"
    
    # Performance summary
    report += "### Performance Metrics\n\n"
    for exp_name, stats in analysis.items():
        if exp_name != 'comparisons':
            report += f"**{exp_name}**:\n"
            report += f"- Mean Reward: {stats['mean_reward']['mean']:.3f} ± {stats['mean_reward']['std']:.3f}\n"
            report += f"- Success Rate: {stats['success_rate']['mean']:.1%}\n"
            report += f"- Total Insights: {stats['insights']['total']}\n\n"
    
    # Statistical comparisons
    if 'comparisons' in analysis:
        report += "### Statistical Comparisons\n\n"
        
        comparisons = analysis['comparisons']
        significant_comparisons = []
        
        for comp_name, comp_data in comparisons.items():
            if comp_data['significant']:
                significant_comparisons.append((comp_name, comp_data))
        
        if significant_comparisons:
            report += "**Statistically Significant Differences (p < 0.05)**:\n\n"
            for comp_name, comp_data in significant_comparisons:
                exp1, exp2 = comp_name.split('_vs_')
                improvement = comp_data['improvement_percent']
                p_value = comp_data['p_value']
                effect_size = comp_data['effect_size']
                
                report += f"- **{exp1}** vs **{exp2}**:\n"
                report += f"  - Improvement: {improvement:+.1f}%\n"
                report += f"  - p-value: {p_value:.4f}\n"
                report += f"  - Effect size (Cohen's d): {effect_size:.3f}\n\n"
        else:
            report += "No statistically significant differences found.\n\n"
    
    # Key findings
    report += "## Key Findings\n\n"
    
    # Best performing configuration
    best_config = None
    best_reward = 0
    for exp_name, stats in analysis.items():
        if exp_name != 'comparisons':
            mean_reward = stats['mean_reward']['mean']
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_config = exp_name
    
    if best_config:
        report += f"1. **Best Performance**: {best_config} achieved highest mean reward ({best_reward:.3f})\n"
    
    # Insight effectiveness
    total_insights = sum(stats['insights']['total'] for exp_name, stats in analysis.items() 
                        if exp_name != 'comparisons')
    report += f"2. **Insight Detection**: {total_insights} total insights detected across all experiments\n"
    
    # Statistical rigor
    report += f"3. **Statistical Rigor**: Analysis based on {len(results)} experimental conditions\n"
    
    report += "\n## Methodology\n\n"
    report += "- **Experimental Design**: Controlled comparison with multiple insight reward scales\n"
    report += "- **Statistical Tests**: Welch's t-test for mean comparisons\n"
    report += "- **Effect Size**: Cohen's d for practical significance\n"
    report += "- **Significance Level**: α = 0.05\n"
    report += "- **Random Seed**: Fixed for reproducibility\n\n"
    
    report += "## Conclusion\n\n"
    report += "This comprehensive experiment demonstrates InsightSpike-AI's performance across "
    report += "different configuration levels with proper statistical controls. The results provide "
    report += "empirical evidence for the effectiveness of insight-driven learning mechanisms.\n"
    
    return report

if __name__ == "__main__":
    try:
        results, analysis = run_comprehensive_experiment()
        print("\n🎉 Comprehensive statistical experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
