"""
実験結果の可視化スクリプト
========================

人間が評価できる形で結果を可視化し、レポートを生成します。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_theme(style="whitegrid", palette="muted")


class ResultVisualizer:
    """実験結果の可視化クラス"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.figures_dir = results_dir.parent / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # 結果の読み込み
        self.results = self._load_results()
        self.summary = self._load_summary()
        self.detailed_logs = self._load_detailed_logs()
    
    def _load_results(self) -> List[Dict]:
        """最終結果を読み込む"""
        results_path = self.results_dir / "final_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _load_summary(self) -> Dict:
        """サマリーを読み込む"""
        summary_path = self.results_dir / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_detailed_logs(self) -> List[Dict]:
        """詳細ログを読み込む"""
        logs = []
        for log_file in self.results_dir.glob("episode_*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs.append(json.load(f))
        return logs
    
    def create_all_visualizations(self):
        """全ての可視化を作成"""
        print("Creating visualizations...")
        
        # 1. 性能比較バーチャート
        self.create_performance_comparison()
        
        # 2. 洞察検出率の比較
        self.create_detection_rate_comparison()
        
        # 3. ΔIG-ΔGED散布図
        self.create_physics_scatter()
        
        # 4. カテゴリ別ヒートマップ
        self.create_category_heatmap()
        
        # 5. 処理時間の比較
        self.create_processing_time_comparison()
        
        # 6. 詳細レポートの生成
        self.create_detailed_report()
        
        print(f"✅ Visualizations saved to {self.figures_dir}")
    
    def create_performance_comparison(self):
        """手法別の性能比較バーチャート"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # データの準備
        methods = ['Direct LLM', 'Standard RAG', 'InsightSpike']
        success_rates = []
        
        for method_key in ['direct', 'rag', 'insight']:
            total = self.summary['total_questions']
            success = self.summary['methods'][method_key]['success']
            success_rates.append(success / total * 100 if total > 0 else 0)
        
        # バーチャート作成
        bars = ax.bar(methods, success_rates, color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # 値をバーの上に表示
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Performance Comparison by Method', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_comparison.png', dpi=300)
        plt.close()
    
    def create_detection_rate_comparison(self):
        """洞察検出率の比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 全体の検出率（円グラフ）
        spike_count = self.summary['methods']['insight']['spikes']
        no_spike_count = self.summary['total_questions'] - spike_count
        
        ax1.pie([spike_count, no_spike_count], 
                labels=['Insight Detected', 'No Insight'],
                autopct='%1.1f%%',
                colors=['#e74c3c', '#95a5a6'],
                startangle=90)
        ax1.set_title('Overall Insight Detection Rate', fontsize=14, fontweight='bold')
        
        # カテゴリ別検出率（棒グラフ）
        categories = []
        detection_rates = []
        
        for cat_name, cat_data in self.summary['categories'].items():
            categories.append(cat_name.replace('_', ' ').title())
            rate = cat_data['spikes'] / cat_data['total'] * 100 if cat_data['total'] > 0 else 0
            detection_rates.append(rate)
        
        bars = ax2.bar(categories, detection_rates, color='#e74c3c')
        
        # 値を表示
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylabel('Detection Rate (%)', fontsize=12)
        ax2.set_title('Insight Detection by Question Category', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'detection_rate_comparison.png', dpi=300)
        plt.close()
    
    def create_physics_scatter(self):
        """ΔIG vs ΔGED散布図"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # データ収集
        delta_igs = []
        delta_geds = []
        categories = []
        spike_detected = []
        
        for result in self.results:
            if 'insight' in result.get('results', {}):
                insight_result = result['results']['insight']
                physics = insight_result.get('physics_metrics', {})
                
                if 'delta_ig' in physics and 'delta_ged' in physics:
                    delta_igs.append(physics['delta_ig'].get('value', 0))
                    delta_geds.append(physics['delta_ged'].get('value', 0))
                    categories.append(result.get('category', 'unknown'))
                    spike_detected.append(insight_result.get('spike_detected', False))
        
        if delta_igs and delta_geds:
            # カテゴリごとに色分け
            category_colors = {
                'baseline': '#3498db',
                'cross_domain': '#2ecc71',
                'abstract': '#e74c3c',
                'edge_cases': '#f39c12'
            }
            
            colors = [category_colors.get(cat, '#95a5a6') for cat in categories]
            markers = ['o' if spike else 'x' for spike in spike_detected]
            
            # 散布図
            for i, (x, y, c, m) in enumerate(zip(delta_igs, delta_geds, colors, markers)):
                ax.scatter(x, y, c=c, marker=m, s=100, alpha=0.7)
            
            # 理論的な閾値線
            ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='ΔIG threshold (0.2 bits)')
            ax.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='ΔGED threshold (-0.5 nodes)')
            
            # 洞察領域を塗りつぶし
            ax.fill([0.2, 10, 10, 0.2], [-10, -10, -0.5, -0.5], 
                    alpha=0.1, color='green', label='Insight Region')
            
            ax.set_xlabel('ΔIG (bits)', fontsize=12)
            ax.set_ylabel('ΔGED (nodes)', fontsize=12)
            ax.set_title('Information Gain vs Graph Edit Distance Changes', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # カスタム凡例を追加
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', label='Baseline'),
                Patch(facecolor='#2ecc71', label='Cross Domain'),
                Patch(facecolor='#e74c3c', label='Abstract'),
                Patch(facecolor='#f39c12', label='Edge Cases')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'physics_scatter.png', dpi=300)
        plt.close()
    
    def create_category_heatmap(self):
        """カテゴリ×評価指標のヒートマップ"""
        # データ準備
        categories = list(self.summary['categories'].keys())
        metrics = ['Success Rate', 'Detection Rate', 'Avg Response Length', 'Avg Processing Time']
        
        data = []
        for cat in categories:
            cat_results = [r for r in self.results if r.get('category') == cat]
            
            # 成功率
            success_count = sum(1 for r in cat_results 
                              if r.get('results', {}).get('insight', {}).get('success', False))
            success_rate = success_count / len(cat_results) * 100 if cat_results else 0
            
            # 検出率
            spike_count = sum(1 for r in cat_results 
                            if r.get('results', {}).get('insight', {}).get('spike_detected', False))
            detection_rate = spike_count / len(cat_results) * 100 if cat_results else 0
            
            # 平均応答長
            response_lengths = [len(r.get('results', {}).get('insight', {}).get('response', '').split())
                              for r in cat_results]
            avg_length = np.mean(response_lengths) if response_lengths else 0
            
            # 平均処理時間
            processing_times = [r.get('results', {}).get('insight', {}).get('time', 0)
                              for r in cat_results]
            avg_time = np.mean(processing_times) if processing_times else 0
            
            data.append([success_rate, detection_rate, avg_length, avg_time])
        
        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # データを正規化（0-1スケール）
        data_array = np.array(data)
        data_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0) + 1e-8)
        
        im = ax.imshow(data_normalized.T, cmap='YlOrRd', aspect='auto')
        
        # ラベル設定
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_yticklabels(metrics)
        
        # 値を表示
        for i in range(len(categories)):
            for j in range(len(metrics)):
                text = ax.text(i, j, f'{data[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Performance Metrics by Question Category', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Normalized Score')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'category_heatmap.png', dpi=300)
        plt.close()
    
    def create_processing_time_comparison(self):
        """処理時間の比較"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # データ収集
        methods = ['Direct LLM', 'Standard RAG', 'InsightSpike']
        method_keys = ['direct', 'rag', 'insight']
        
        times_data = {method: [] for method in method_keys}
        
        for result in self.results:
            for method in method_keys:
                if method in result.get('results', {}):
                    time_val = result['results'][method].get('time', 0)
                    times_data[method].append(time_val)
        
        # ボックスプロット
        data_to_plot = [times_data[method] for method in method_keys]
        
        bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True)
        
        # 色付け
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax.set_title('Processing Time Distribution by Method', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 平均値を表示
        for i, (method, times) in enumerate(times_data.items()):
            if times:
                mean_time = np.mean(times)
                ax.text(i + 1, mean_time, f'μ={mean_time:.2f}s', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'processing_time_comparison.png', dpi=300)
        plt.close()
    
    def create_detailed_report(self):
        """詳細な人間評価用レポートの生成"""
        report_path = self.results_dir.parent / "human_evaluation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# geDIG理論検証実験 v2.0 - 人間評価用レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # サマリー
            f.write("## 実験サマリー\n\n")
            f.write(f"- 総質問数: {self.summary['total_questions']}\n")
            f.write(f"- InsightSpike洞察検出率: {self.summary['methods']['insight']['spikes']}/{self.summary['total_questions']} ")
            f.write(f"({self.summary['methods']['insight']['spikes']/self.summary['total_questions']*100:.1f}%)\n")
            f.write(f"- 平均ΔIG: {self.summary['physics_summary']['average_delta_ig']:.3f} bits\n")
            f.write(f"- 平均ΔGED: {self.summary['physics_summary']['average_delta_ged']:.3f} nodes\n\n")
            
            # カテゴリ別結果
            f.write("## カテゴリ別結果\n\n")
            for cat_name, cat_data in self.summary['categories'].items():
                f.write(f"### {cat_name.replace('_', ' ').title()}\n")
                f.write(f"- 質問数: {cat_data['total']}\n")
                f.write(f"- 洞察検出: {cat_data['spikes']} ({cat_data['spikes']/cat_data['total']*100:.1f}%)\n\n")
            
            # 代表的な回答例
            f.write("## 代表的な回答例\n\n")
            
            # 洞察が検出された例を3つ抽出
            insight_examples = [r for r in self.results 
                              if r.get('results', {}).get('insight', {}).get('spike_detected', False)][:3]
            
            for i, example in enumerate(insight_examples, 1):
                f.write(f"### 例{i}: {example['question']['text']}\n\n")
                f.write(f"**カテゴリ**: {example['category'].replace('_', ' ').title()}\n\n")
                
                # 各手法の回答
                for method_name, method_key in [('Direct LLM', 'direct'), 
                                               ('Standard RAG', 'rag'), 
                                               ('InsightSpike', 'insight')]:
                    if method_key in example.get('results', {}):
                        response = example['results'][method_key].get('response', 'N/A')
                        f.write(f"**{method_name}の回答**:\n")
                        f.write(f"```\n{response[:500]}{'...' if len(response) > 500 else ''}\n```\n\n")
                
                # InsightSpikeの物理量
                if 'insight' in example.get('results', {}):
                    physics = example['results']['insight'].get('physics_metrics', {})
                    if physics:
                        f.write("**物理量**:\n")
                        f.write(f"- ΔIG: {physics.get('delta_ig', {}).get('value', 'N/A')} bits\n")
                        f.write(f"- ΔGED: {physics.get('delta_ged', {}).get('value', 'N/A')} nodes\n\n")
                
                f.write("---\n\n")
            
            # 評価ガイドライン
            f.write("## 人間評価ガイドライン\n\n")
            f.write("各回答を以下の5つの観点から5段階で評価してください：\n\n")
            f.write("1. **新規性 (Novelty)**: 質問に対して新しい視点や洞察を提供しているか\n")
            f.write("2. **有用性 (Usefulness)**: 回答が実用的で価値があるか\n")
            f.write("3. **一貫性 (Coherence)**: 論理的で矛盾のない回答か\n")
            f.write("4. **深さ (Depth)**: 表面的でなく深い理解を示しているか\n")
            f.write("5. **統合度 (Integration)**: 複数の概念を効果的に結びつけているか\n\n")
            
            f.write("### 評価スケール\n")
            f.write("- 5: 非常に優れている\n")
            f.write("- 4: 優れている\n")
            f.write("- 3: 普通\n")
            f.write("- 2: やや劣る\n")
            f.write("- 1: 劣る\n\n")
            
            # 詳細データへのリンク
            f.write("## 詳細データ\n\n")
            f.write(f"- 完全な実験結果: `{self.results_dir / 'final_results.json'}`\n")
            f.write(f"- 詳細ログ: `{self.results_dir}/episode_*.json`\n")
            f.write(f"- 可視化図表: `{self.figures_dir}/`\n")
        
        print(f"✅ Human evaluation report saved to {report_path}")


def main():
    """メイン実行関数"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print("❌ Results directory not found. Please run the experiment first.")
        return
    
    print("=== Creating Visualizations ===")
    
    visualizer = ResultVisualizer(results_dir)
    visualizer.create_all_visualizations()
    
    print("\n✅ Visualization complete!")
    print("Check the 'figures' directory for visualizations")
    print("Check 'human_evaluation_report.md' for detailed evaluation guide")


if __name__ == "__main__":
    main()