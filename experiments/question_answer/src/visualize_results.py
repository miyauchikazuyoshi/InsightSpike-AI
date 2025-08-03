"""
Visualization module for Question-Answer Experiment Results
========================================================

Generates comprehensive visualizations for experiment analysis.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ExperimentVisualizer:
    """Visualizes experiment results with publication-quality figures"""
    
    def __init__(self, results_dir: Path = Path("../results")):
        self.results_dir = results_dir
        self.output_dir = results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for compatibility"""
        # Check format and add missing columns
        if 'ged_1hop' not in df.columns:
            if 'ged_value' in df.columns:
                df['ged_1hop'] = df['ged_value']
                df['ig_1hop'] = df.get('ig_value', 0)
                df['ged_2hop'] = 0
                df['ig_2hop'] = 0
            else:
                # No GED/IG data at all
                df['ged_1hop'] = 0
                df['ig_1hop'] = 0
                df['ged_2hop'] = 0
                df['ig_2hop'] = 0
        return df
        
    def load_results(self, experiment_name: str = "latest") -> Tuple[Dict, pd.DataFrame]:
        """Load experiment results from JSON and CSV files"""
        
        # Find latest experiment if requested
        if experiment_name == "latest":
            metrics_files = list((self.results_dir / "metrics").glob("experiment_*.json"))
            if not metrics_files:
                raise FileNotFoundError("No experiment results found")
            latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
            experiment_name = latest_file.stem
        
        # Load JSON metrics
        json_path = self.results_dir / "metrics" / f"{experiment_name}.json"
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            
        # Load CSV data - try different naming patterns
        csv_path = self.results_dir / "metrics" / f"{experiment_name}.csv"
        if not csv_path.exists():
            # Try alternative naming
            timestamp = experiment_name.replace("experiment_results_", "")
            csv_path = self.results_dir / "metrics" / f"question_results_{timestamp}.csv"
        
        csv_data = pd.read_csv(csv_path)
        
        # Normalize columns for compatibility
        csv_data = self._normalize_columns(csv_data)
        
        return json_data, csv_data
    
    def plot_ged_ig_distribution(self, df: pd.DataFrame, save_name: str = "ged_ig_distribution.png"):
        """Plot distribution of GED and IG values"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1-hop distributions
        axes[0, 0].hist(df['ged_1hop'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('1-hop GED Distribution')
        axes[0, 0].set_xlabel('GED Value')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(df['ig_1hop'].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('1-hop IG Distribution')
        axes[0, 1].set_xlabel('IG Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # 2-hop distributions - only show if we have actual 2-hop data
        if df['ged_2hop'].sum() > 0:
            axes[1, 0].hist(df['ged_2hop'].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('2-hop GED Distribution')
            axes[1, 0].set_xlabel('GED Value')
            axes[1, 0].set_ylabel('Frequency')
            
            axes[1, 1].hist(df['ig_2hop'].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('2-hop IG Distribution')
            axes[1, 1].set_xlabel('IG Value')
            axes[1, 1].set_ylabel('Frequency')
        else:
            axes[1, 0].text(0.5, 0.5, '2-hop data not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, '2-hop data not available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ged_ig_correlation(self, df: pd.DataFrame, save_name: str = "ged_ig_correlation.png"):
        """Plot correlation between GED and IG values"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1-hop correlation
        axes[0].scatter(df['ged_1hop'], df['ig_1hop'], alpha=0.6)
        axes[0].set_xlabel('1-hop GED')
        axes[0].set_ylabel('1-hop IG')
        axes[0].set_title('1-hop GED vs IG Correlation')
        
        # Add correlation coefficient
        corr_1hop = df[['ged_1hop', 'ig_1hop']].corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr_1hop:.3f}', 
                    transform=axes[0].transAxes, verticalalignment='top')
        
        # 2-hop correlation
        axes[1].scatter(df['ged_2hop'], df['ig_2hop'], alpha=0.6, color='orange')
        axes[1].set_xlabel('2-hop GED')
        axes[1].set_ylabel('2-hop IG')
        axes[1].set_title('2-hop GED vs IG Correlation')
        
        # Add correlation coefficient
        corr_2hop = df[['ged_2hop', 'ig_2hop']].corr().iloc[0, 1]
        axes[1].text(0.05, 0.95, f'r = {corr_2hop:.3f}', 
                    transform=axes[1].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spike_analysis(self, json_data: Dict, df: pd.DataFrame, 
                          save_name: str = "spike_analysis.png"):
        """Analyze spike detection patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Spike detection rate by difficulty
        difficulty_stats = json_data.get('detailed_stats', {}).get('by_difficulty', {})
        if difficulty_stats:
            difficulties = list(difficulty_stats.keys())
            spike_rates = [stats.get('spike_rate', 0) for stats in difficulty_stats.values()]
            
            axes[0, 0].bar(difficulties, spike_rates)
            axes[0, 0].set_title('Spike Detection Rate by Difficulty')
            axes[0, 0].set_xlabel('Difficulty Level')
            axes[0, 0].set_ylabel('Spike Rate (%)')
            axes[0, 0].set_ylim(0, 100)
        
        # GED/IG values for spike vs non-spike
        spike_column = 'has_spike' if 'has_spike' in df.columns else 'has_insight'
        spike_mask = df[spike_column] == True
        
        # 1-hop comparison
        data_1hop = [
            df.loc[spike_mask, 'ged_1hop'].values,
            df.loc[~spike_mask, 'ged_1hop'].values,
            df.loc[spike_mask, 'ig_1hop'].values,
            df.loc[~spike_mask, 'ig_1hop'].values
        ]
        
        positions = [1, 2, 4, 5]
        labels = ['GED\n(spike)', 'GED\n(no spike)', 'IG\n(spike)', 'IG\n(no spike)']
        
        bp1 = axes[0, 1].boxplot(data_1hop, positions=positions, labels=labels)
        axes[0, 1].set_title('1-hop GED/IG: Spike vs Non-spike')
        axes[0, 1].set_ylabel('Value')
        
        # 2-hop comparison
        data_2hop = [
            df.loc[spike_mask, 'ged_2hop'].values,
            df.loc[~spike_mask, 'ged_2hop'].values,
            df.loc[spike_mask, 'ig_2hop'].values,
            df.loc[~spike_mask, 'ig_2hop'].values
        ]
        
        bp2 = axes[1, 0].boxplot(data_2hop, positions=positions, labels=labels)
        axes[1, 0].set_title('2-hop GED/IG: Spike vs Non-spike')
        axes[1, 0].set_ylabel('Value')
        
        # Combined metric analysis
        df['combined_1hop'] = df['ig_1hop'] - df['ged_1hop']
        df['combined_2hop'] = df['ig_2hop'] - df['ged_2hop']
        
        axes[1, 1].hist([df.loc[spike_mask, 'combined_2hop'], 
                        df.loc[~spike_mask, 'combined_2hop']], 
                       label=['Spike', 'No Spike'], 
                       bins=20, alpha=0.7)
        axes[1, 1].set_title('Combined Metric (IG - GED) Distribution')
        axes[1, 1].set_xlabel('IG - GED (2-hop)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_analysis(self, df: pd.DataFrame, save_name: str = "temporal_analysis.png"):
        """Analyze how metrics evolve over questions"""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rolling average of GED/IG
        window = 10
        df['ged_1hop_ma'] = df['ged_1hop'].rolling(window).mean()
        df['ig_1hop_ma'] = df['ig_1hop'].rolling(window).mean()
        df['ged_2hop_ma'] = df['ged_2hop'].rolling(window).mean()
        df['ig_2hop_ma'] = df['ig_2hop'].rolling(window).mean()
        
        # 1-hop temporal evolution
        axes[0].plot(df.index, df['ged_1hop_ma'], label='GED (1-hop)', alpha=0.8)
        axes[0].plot(df.index, df['ig_1hop_ma'], label='IG (1-hop)', alpha=0.8)
        axes[0].set_title(f'1-hop Metrics Evolution (Rolling Average, window={window})')
        axes[0].set_xlabel('Question Number')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2-hop temporal evolution
        axes[1].plot(df.index, df['ged_2hop_ma'], label='GED (2-hop)', alpha=0.8)
        axes[1].plot(df.index, df['ig_2hop_ma'], label='IG (2-hop)', alpha=0.8)
        axes[1].set_title(f'2-hop Metrics Evolution (Rolling Average, window={window})')
        axes[1].set_xlabel('Question Number')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hop_comparison(self, df: pd.DataFrame, save_name: str = "hop_comparison.png"):
        """Compare 1-hop vs 2-hop metrics"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # GED comparison
        axes[0].scatter(df['ged_1hop'], df['ged_2hop'], alpha=0.6)
        axes[0].plot([0, df['ged_1hop'].max()], [0, df['ged_1hop'].max()], 
                    'r--', label='y=x', alpha=0.5)
        axes[0].set_xlabel('1-hop GED')
        axes[0].set_ylabel('2-hop GED')
        axes[0].set_title('GED: 1-hop vs 2-hop')
        axes[0].legend()
        
        # IG comparison
        axes[1].scatter(df['ig_1hop'], df['ig_2hop'], alpha=0.6, color='orange')
        axes[1].plot([0, df['ig_1hop'].max()], [0, df['ig_1hop'].max()], 
                    'r--', label='y=x', alpha=0.5)
        axes[1].set_xlabel('1-hop IG')
        axes[1].set_ylabel('2-hop IG')
        axes[1].set_title('IG: 1-hop vs 2-hop')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, json_data: Dict, df: pd.DataFrame, 
                              save_name: str = "summary_report.txt"):
        """Generate a text summary of key findings"""
        
        report = []
        report.append("="*60)
        report.append("EXPERIMENT SUMMARY REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"Total Questions: {json_data.get('total_questions', len(df))}")
        report.append(f"Total Knowledge Entries: {json_data.get('total_knowledge', 'N/A')}")
        report.append(f"Overall Spike Rate: {json_data.get('spike_rate', 0):.2f}%")
        
        # Calculate avg response time from dataframe if not in json
        if 'avg_response_time' in json_data:
            avg_time = json_data['avg_response_time']
        else:
            avg_time = df['processing_time'].mean() if 'processing_time' in df.columns else 0
        report.append(f"Average Response Time: {avg_time:.3f}s")
        report.append("")
        
        # GED/IG statistics
        report.append("GED/IG METRICS:")
        report.append(f"1-hop GED: mean={df['ged_1hop'].mean():.3f}, std={df['ged_1hop'].std():.3f}")
        report.append(f"1-hop IG:  mean={df['ig_1hop'].mean():.3f}, std={df['ig_1hop'].std():.3f}")
        report.append(f"2-hop GED: mean={df['ged_2hop'].mean():.3f}, std={df['ged_2hop'].std():.3f}")
        report.append(f"2-hop IG:  mean={df['ig_2hop'].mean():.3f}, std={df['ig_2hop'].std():.3f}")
        report.append("")
        
        # Correlation analysis
        report.append("CORRELATION ANALYSIS:")
        corr_1hop = df[['ged_1hop', 'ig_1hop']].corr().iloc[0, 1]
        corr_2hop = df[['ged_2hop', 'ig_2hop']].corr().iloc[0, 1]
        report.append(f"1-hop GED-IG correlation: {corr_1hop:.3f}")
        report.append(f"2-hop GED-IG correlation: {corr_2hop:.3f}")
        report.append("")
        
        # Spike analysis
        spike_column = 'has_spike' if 'has_spike' in df.columns else 'has_insight'
        if df[spike_column].any():
            spike_ged_1hop = df.loc[df[spike_column], 'ged_1hop'].mean()
            no_spike_ged_1hop = df.loc[~df[spike_column], 'ged_1hop'].mean()
            spike_ig_1hop = df.loc[df[spike_column], 'ig_1hop'].mean()
            no_spike_ig_1hop = df.loc[~df[spike_column], 'ig_1hop'].mean()
            
            report.append("SPIKE DETECTION ANALYSIS:")
            report.append(f"Spike cases - 1hop GED: {spike_ged_1hop:.3f}, IG: {spike_ig_1hop:.3f}")
            report.append(f"Non-spike   - 1hop GED: {no_spike_ged_1hop:.3f}, IG: {no_spike_ig_1hop:.3f}")
        else:
            report.append("SPIKE DETECTION ANALYSIS:")
            report.append("No spikes detected in this experiment")
        
        report.append("")
        report.append("="*60)
        
        # Save report
        with open(self.output_dir / save_name, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    def run_all_visualizations(self, experiment_name: str = "latest"):
        """Run all visualization methods and generate complete report"""
        
        print(f"Loading results for experiment: {experiment_name}")
        json_data, df = self.load_results(experiment_name)
        
        print("Generating visualizations...")
        
        # Generate all plots
        self.plot_ged_ig_distribution(df)
        print("  ✓ GED/IG distribution plot")
        
        self.plot_ged_ig_correlation(df)
        print("  ✓ GED/IG correlation plot")
        
        self.plot_spike_analysis(json_data, df)
        print("  ✓ Spike analysis plot")
        
        self.plot_temporal_analysis(df)
        print("  ✓ Temporal analysis plot")
        
        self.plot_hop_comparison(df)
        print("  ✓ Hop comparison plot")
        
        # Generate summary report
        summary = self.generate_summary_report(json_data, df)
        print("  ✓ Summary report")
        
        print(f"\nAll visualizations saved to: {self.output_dir}")
        print("\nSummary:")
        print(summary)


if __name__ == "__main__":
    visualizer = ExperimentVisualizer()
    visualizer.run_all_visualizations()