"""
Google Colab用の評価実験インターフェース
学術的研究基準に基づく包括的評価システムのColab統合版

このモジュールはColabノートブックから簡単に実行できるように設計されています。
"""

import asyncio
import nest_asyncio
import warnings
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, HTML, Markdown
import json
from pathlib import Path

# Enable nested asyncio for Colab
nest_asyncio.apply()

# Import the main evaluation framework
from objective_evaluation_framework import (
    ObjectiveEvaluationFramework, 
    ExperimentConfig, 
    ExperimentResult,
    run_quick_evaluation,
    create_evaluation_framework
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ColabEvaluationInterface:
    """Google Colab用の評価実験インターフェース"""
    
    def __init__(self):
        self.framework = create_evaluation_framework()
        self.current_result: Optional[ExperimentResult] = None
        
    def display_welcome_message(self):
        """ウェルカムメッセージの表示"""
        welcome_html = """
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f9f9f9; margin: 10px 0;">
            <h2 style="color: #2E7D32; margin-top: 0;">🧠 InsightSpike-AI 評価実験</h2>
            <p style="font-size: 16px; color: #333;">
                <strong>学術研究基準に基づく包括的評価フレームワーク</strong>
            </p>
            <div style="background-color: #E8F5E8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #2E7D32; margin-top: 0;">🎯 実験の特徴</h3>
                <ul style="color: #333; margin: 0;">
                    <li><strong>標準ベンチマーク評価</strong>: SQuAD, ARC Challenge, 論理パズル等</li>
                    <li><strong>厳密なベースライン比較</strong>: GPT, Retrieval+LLM, ルールベース手法</li>
                    <li><strong>統計的検証</strong>: クロスバリデーション、有意性検定</li>
                    <li><strong>アブレーション実験</strong>: 各コンポーネントの寄与度分析</li>
                    <li><strong>閾値感度分析</strong>: パラメータのロバスト性検証</li>
                </ul>
            </div>
            <div style="background-color: #FFF3E0; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #F57C00; margin-top: 0;">⚡ Colab最適化</h3>
                <p style="color: #333; margin: 0;">
                    このインターフェースはGoogle Colabでの実行に最適化されており、
                    資源制約を考慮したサンプルサイズと高速実行を実現します。
                </p>
            </div>
        </div>
        """
        display(HTML(welcome_html))
    
    def run_quick_evaluation_demo(self, sample_size: int = 20):
        """クイック評価デモの実行"""
        print("🚀 クイック評価実験を開始します...")
        print(f"📊 サンプルサイズ: {sample_size}")
        print("⏱️  推定実行時間: 2-3分")
        print("-" * 50)
        
        # Run the evaluation
        loop = asyncio.get_event_loop()
        self.current_result = loop.run_until_complete(run_quick_evaluation(sample_size))
        
        # Display results
        self.display_results_summary()
        
        return self.current_result
    
    def run_comprehensive_evaluation(self, 
                                   datasets: List[str] = ["squad_v2", "logic_puzzles"], 
                                   sample_size: int = 30):
        """包括的評価実験の実行"""
        print("🔬 包括的評価実験を開始します...")
        print(f"📚 データセット: {', '.join(datasets)}")
        print(f"📊 サンプルサイズ: {sample_size}")
        print("⏱️  推定実行時間: 5-8分")
        print("-" * 50)
        
        config = ExperimentConfig(
            name="comprehensive_colab_evaluation",
            description="Colab包括的客観評価実験",
            datasets=datasets,
            baselines=["simple_llm", "retrieval_llm", "rule_based", "insightspike"],
            metrics=["accuracy", "confidence", "response_time", "insight_detection"],
            sample_size=sample_size,
            cross_validation_folds=3,
            random_seed=42
        )
        
        loop = asyncio.get_event_loop()
        self.current_result = loop.run_until_complete(
            self.framework.run_comprehensive_evaluation(config)
        )
        
        # Display comprehensive results
        self.display_comprehensive_results()
        
        return self.current_result
    
    def display_results_summary(self):
        """結果サマリーの表示"""
        if not self.current_result:
            print("❌ 実行結果がありません。先に評価実験を実行してください。")
            return
        
        result = self.current_result
        
        # Basic info
        summary_html = f"""
        <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 20px; background-color: #f8f9fa; margin: 20px 0;">
            <h2 style="color: #1976D2; margin-top: 0;">📈 実験結果サマリー</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                <div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; flex: 1; min-width: 200px;">
                    <h4 style="color: #1976D2; margin: 0 0 10px 0;">基本情報</h4>
                    <p style="margin: 5px 0;"><strong>実験名:</strong> {result.config.name}</p>
                    <p style="margin: 5px 0;"><strong>実行時間:</strong> {result.execution_time:.2f}秒</p>
                    <p style="margin: 5px 0;"><strong>処理サンプル数:</strong> {result.metadata.get('total_samples_processed', 'N/A')}</p>
                </div>
                <div style="background-color: #E8F5E8; padding: 15px; border-radius: 8px; flex: 1; min-width: 200px;">
                    <h4 style="color: #2E7D32; margin: 0 0 10px 0;">テスト設定</h4>
                    <p style="margin: 5px 0;"><strong>データセット数:</strong> {len(result.config.datasets)}</p>
                    <p style="margin: 5px 0;"><strong>ベースライン数:</strong> {len(result.config.baselines)}</p>
                    <p style="margin: 5px 0;"><strong>CV分割数:</strong> {result.config.cross_validation_folds}</p>
                </div>
            </div>
        </div>
        """
        display(HTML(summary_html))
        
        # Performance comparison
        self._display_performance_comparison()
        
        # Statistical significance
        self._display_statistical_results()
        
    def display_comprehensive_results(self):
        """包括的結果の表示"""
        if not self.current_result:
            print("❌ 実行結果がありません。")
            return
        
        # Show summary first
        self.display_results_summary()
        
        # Show additional comprehensive results
        self._display_ablation_results()
        self._display_threshold_analysis()
        self._display_cross_validation_results()
    
    def _display_performance_comparison(self):
        """パフォーマンス比較の表示"""
        if not self.current_result or 'datasets' not in self.current_result.results:
            return
        
        print("\n" + "="*60)
        print("📊 ベースライン手法パフォーマンス比較")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        
        for dataset_name, dataset_data in self.current_result.results['datasets'].items():
            baseline_results = dataset_data['baseline_results']
            
            for baseline_name, baseline_data in baseline_results.items():
                metrics = baseline_data['summary_metrics']
                comparison_data.append({
                    'データセット': dataset_name,
                    'ベースライン': baseline_name,
                    '平均精度': f"{metrics['mean_accuracy']:.3f}",
                    '応答時間(秒)': f"{metrics['mean_response_time']:.3f}",
                    '洞察検出率': f"{metrics.get('insight_detection_rate', 0.0):.3f}",
                    'エラー率': f"{metrics['error_rate']:.3f}"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Style the dataframe for better display
            styled_df = df.style.format({
                '平均精度': '{:.3f}',
                '応答時間(秒)': '{:.3f}',
                '洞察検出率': '{:.3f}',
                'エラー率': '{:.3f}'
            }).background_gradient(subset=['平均精度'], cmap='RdYlGn')
            
            display(styled_df)
            
            # Highlight best performance
            best_baseline = df.loc[df['平均精度'].astype(float).idxmax(), 'ベースライン']
            print(f"\n🏆 最高パフォーマンス: {best_baseline}")
    
    def _display_statistical_results(self):
        """統計的有意性結果の表示"""
        if not self.current_result or 'statistical_analysis' not in self.current_result.results:
            return
        
        print("\n" + "="*60)
        print("📈 統計的有意性分析")
        print("="*60)
        
        stats_data = self.current_result.results['statistical_analysis']
        
        for dataset_name, comparisons in stats_data.items():
            print(f"\n📚 データセット: {dataset_name}")
            print("-" * 40)
            
            for baseline_name, significance in comparisons.items():
                if 'error' in significance:
                    print(f"   {baseline_name}: 分析エラー")
                    continue
                
                p_value = significance.get('p_value', 1.0)
                is_significant = significance.get('significant', False)
                effect_size = significance.get('effect_size', 'unknown')
                
                status = "✅ 有意" if is_significant else "❌ 非有意"
                print(f"   vs {baseline_name}: {status} (p={p_value:.4f}, 効果量: {effect_size})")
    
    def _display_ablation_results(self):
        """アブレーション実験結果の表示"""
        if not self.current_result or 'ablation_study' not in self.current_result.results:
            return
        
        print("\n" + "="*60)
        print("🔧 アブレーション実験結果")
        print("="*60)
        
        ablation_data = self.current_result.results['ablation_study']
        
        # Create ablation comparison table
        ablation_rows = []
        for variant_name, variant_data in ablation_data.items():
            ablation_rows.append({
                'バリアント': variant_name,
                '説明': variant_data['description'],
                '平均精度': f"{variant_data['mean_accuracy']:.3f}",
                '応答時間(秒)': f"{variant_data['mean_response_time']:.3f}",
                '洞察検出率': f"{variant_data['insight_detection_rate']:.3f}"
            })
        
        if ablation_rows:
            ablation_df = pd.DataFrame(ablation_rows)
            styled_ablation = ablation_df.style.background_gradient(
                subset=['平均精度'], cmap='RdYlGn'
            )
            display(styled_ablation)
            
            # Show component contribution
            full_performance = next(
                (float(row['平均精度']) for row in ablation_rows if 'full' in row['バリアント']), 
                0.0
            )
            
            print(f"\n📊 コンポーネント寄与度分析:")
            for row in ablation_rows:
                if 'full' not in row['バリアント']:
                    performance_drop = full_performance - float(row['平均精度'])
                    print(f"   {row['バリアント']}: -{performance_drop:.3f} ({performance_drop/full_performance*100:.1f}%低下)")
    
    def _display_threshold_analysis(self):
        """閾値感度分析の表示"""
        if not self.current_result or 'threshold_analysis' not in self.current_result.results:
            return
        
        print("\n" + "="*60)
        print("⚙️ 閾値感度分析")
        print("="*60)
        
        threshold_data = self.current_result.results['threshold_analysis']
        
        # Find optimal threshold
        best_threshold = None
        best_f1 = 0.0
        
        for threshold_str, metrics in threshold_data.items():
            f1_score = metrics['f1_score']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = float(threshold_str)
        
        print(f"🎯 最適閾値: {best_threshold:.2f} (F1スコア: {best_f1:.3f})")
        
        # Show threshold range performance
        threshold_rows = []
        for threshold_str, metrics in list(threshold_data.items())[::2]:  # Show every 2nd for brevity
            threshold_rows.append({
                '閾値': float(threshold_str),
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1スコア': f"{metrics['f1_score']:.3f}"
            })
        
        if threshold_rows:
            threshold_df = pd.DataFrame(threshold_rows)
            styled_threshold = threshold_df.style.background_gradient(
                subset=['F1スコア'], cmap='RdYlGn'
            )
            display(styled_threshold)
    
    def _display_cross_validation_results(self):
        """クロスバリデーション結果の表示"""
        if not self.current_result or 'cross_validation' not in self.current_result.results:
            return
        
        print("\n" + "="*60)
        print("🔄 クロスバリデーション結果")
        print("="*60)
        
        cv_data = self.current_result.results['cross_validation']
        
        for baseline_name, cv_results in cv_data.items():
            print(f"\n📈 {baseline_name}:")
            print(f"   平均精度: {cv_results['cv_accuracy_mean']:.3f} ± {cv_results['cv_accuracy_std']:.3f}")
            print(f"   平均応答時間: {cv_results['cv_time_mean']:.3f} ± {cv_results['cv_time_std']:.3f}秒")
    
    def create_visual_report(self, save_path: Optional[str] = None):
        """ビジュアルレポートの作成"""
        if not self.current_result:
            print("❌ 実行結果がありません。")
            return
        
        print("📊 ビジュアルレポートを生成中...")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('InsightSpike-AI 客観的評価結果', fontsize=16, fontweight='bold')
        
        # Plot 1: Baseline Accuracy Comparison
        self._plot_baseline_comparison(axes[0, 0])
        
        # Plot 2: Response Time Comparison  
        self._plot_response_time_comparison(axes[0, 1])
        
        # Plot 3: Insight Detection Rate
        self._plot_insight_detection_rate(axes[0, 2])
        
        # Plot 4: Ablation Study
        self._plot_ablation_study(axes[1, 0])
        
        # Plot 5: Threshold Analysis
        self._plot_threshold_analysis(axes[1, 1])
        
        # Plot 6: Cross-Validation Stability
        self._plot_cv_stability(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 レポートを保存しました: {save_path}")
        
        plt.show()
    
    def _plot_baseline_comparison(self, ax):
        """ベースライン比較プロット"""
        if 'datasets' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ベースライン精度比較')
            return
        
        # Extract accuracy data
        baselines = []
        accuracies = []
        
        first_dataset = list(self.current_result.results['datasets'].keys())[0]
        baseline_results = self.current_result.results['datasets'][first_dataset]['baseline_results']
        
        for baseline_name, baseline_data in baseline_results.items():
            baselines.append(baseline_name)
            accuracies.append(baseline_data['summary_metrics']['mean_accuracy'])
        
        # Create bar plot
        colors = ['skyblue', 'lightgreen', 'orange', 'gold'] * (len(baselines) // 4 + 1)
        bars = ax.bar(baselines, accuracies, color=colors[:len(baselines)])
        
        # Highlight InsightSpike
        for i, baseline in enumerate(baselines):
            if 'insightspike' in baseline.lower():
                bars[i].set_color('red')
                bars[i].set_edgecolor('darkred')
                bars[i].set_linewidth(2)
        
        ax.set_title('ベースライン精度比較')
        ax.set_ylabel('平均精度')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_response_time_comparison(self, ax):
        """応答時間比較プロット"""
        if 'datasets' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('応答時間比較')
            return
        
        # Extract response time data
        baselines = []
        response_times = []
        
        first_dataset = list(self.current_result.results['datasets'].keys())[0]
        baseline_results = self.current_result.results['datasets'][first_dataset]['baseline_results']
        
        for baseline_name, baseline_data in baseline_results.items():
            baselines.append(baseline_name)
            response_times.append(baseline_data['summary_metrics']['mean_response_time'])
        
        ax.bar(baselines, response_times, color='lightcoral')
        ax.set_title('応答時間比較')
        ax.set_ylabel('平均応答時間 (秒)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_insight_detection_rate(self, ax):
        """洞察検出率プロット"""
        if 'datasets' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('洞察検出率')
            return
        
        # Extract insight detection data
        baselines = []
        detection_rates = []
        
        first_dataset = list(self.current_result.results['datasets'].keys())[0]
        baseline_results = self.current_result.results['datasets'][first_dataset]['baseline_results']
        
        for baseline_name, baseline_data in baseline_results.items():
            baselines.append(baseline_name)
            detection_rates.append(baseline_data['summary_metrics'].get('insight_detection_rate', 0.0))
        
        ax.bar(baselines, detection_rates, color='lightblue')
        ax.set_title('洞察検出率')
        ax.set_ylabel('検出率')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_ablation_study(self, ax):
        """アブレーション実験プロット"""
        if 'ablation_study' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('アブレーション実験')
            return
        
        ablation_data = self.current_result.results['ablation_study']
        
        variants = list(ablation_data.keys())
        accuracies = [ablation_data[v]['mean_accuracy'] for v in variants]
        
        ax.barh(variants, accuracies, color='lightyellow')
        ax.set_title('アブレーション実験結果')
        ax.set_xlabel('平均精度')
        ax.grid(True, alpha=0.3)
    
    def _plot_threshold_analysis(self, ax):
        """閾値分析プロット"""
        if 'threshold_analysis' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('閾値感度分析')
            return
        
        threshold_data = self.current_result.results['threshold_analysis']
        
        thresholds = [float(k) for k in threshold_data.keys()]
        f1_scores = [threshold_data[k]['f1_score'] for k in threshold_data.keys()]
        precisions = [threshold_data[k]['precision'] for k in threshold_data.keys()]
        recalls = [threshold_data[k]['recall'] for k in threshold_data.keys()]
        
        ax.plot(thresholds, f1_scores, 'o-', label='F1スコア', linewidth=2)
        ax.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
        
        ax.set_title('閾値感度分析')
        ax.set_xlabel('洞察検出閾値')
        ax.set_ylabel('スコア')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cv_stability(self, ax):
        """クロスバリデーション安定性プロット"""
        if 'cross_validation' not in self.current_result.results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('CV安定性')
            return
        
        cv_data = self.current_result.results['cross_validation']
        
        baselines = list(cv_data.keys())
        means = [cv_data[b]['cv_accuracy_mean'] for b in baselines]
        stds = [cv_data[b]['cv_accuracy_std'] for b in baselines]
        
        ax.errorbar(range(len(baselines)), means, yerr=stds, 
                   fmt='o', capsize=5, capthick=2, linewidth=2)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels(baselines, rotation=45)
        ax.set_title('クロスバリデーション安定性')
        ax.set_ylabel('精度 (平均 ± 標準偏差)')
        ax.grid(True, alpha=0.3)
    
    def export_results_json(self, filename: str = "evaluation_results.json"):
        """結果をJSONでエクスポート"""
        if not self.current_result:
            print("❌ エクスポートする結果がありません。")
            return
        
        # Convert result to serializable format
        export_data = {
            'config': {
                'name': self.current_result.config.name,
                'description': self.current_result.config.description,
                'datasets': self.current_result.config.datasets,
                'baselines': self.current_result.config.baselines,
                'metrics': self.current_result.config.metrics,
                'sample_size': self.current_result.config.sample_size
            },
            'results': self.current_result.results,
            'execution_time': self.current_result.execution_time,
            'timestamp': self.current_result.timestamp,
            'metadata': self.current_result.metadata
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 結果をエクスポートしました: {filename}")
    
    def get_summary_markdown(self) -> str:
        """結果のMarkdownサマリーを生成"""
        if not self.current_result:
            return "# エラー: 実行結果がありません"
        
        result = self.current_result
        
        markdown = f"""# InsightSpike-AI 客観的評価結果

## 実験概要
- **実験名**: {result.config.name}
- **実行時間**: {result.execution_time:.2f}秒
- **処理サンプル数**: {result.metadata.get('total_samples_processed', 'N/A')}
- **タイムスタンプ**: {result.timestamp}

## テスト設定
- **データセット**: {', '.join(result.config.datasets)}
- **ベースライン手法**: {', '.join(result.config.baselines)}
- **サンプルサイズ**: {result.config.sample_size}
- **クロスバリデーション分割数**: {result.config.cross_validation_folds}

## 主要結果

### ベースライン比較
"""
        
        # Add baseline comparison if available
        if 'datasets' in result.results and result.results['datasets']:
            first_dataset = list(result.results['datasets'].keys())[0]
            baseline_results = result.results['datasets'][first_dataset]['baseline_results']
            
            markdown += "| ベースライン | 平均精度 | 応答時間(秒) | 洞察検出率 |\n"
            markdown += "|-------------|----------|-------------|----------|\n"
            
            for baseline_name, baseline_data in baseline_results.items():
                metrics = baseline_data['summary_metrics']
                markdown += f"| {baseline_name} | {metrics['mean_accuracy']:.3f} | {metrics['mean_response_time']:.3f} | {metrics.get('insight_detection_rate', 0.0):.3f} |\n"
        
        markdown += "\n### 統計的有意性\n"
        
        # Add statistical results if available
        if 'statistical_analysis' in result.results:
            for dataset_name, comparisons in result.results['statistical_analysis'].items():
                markdown += f"\n**{dataset_name}データセット**:\n"
                for baseline_name, significance in comparisons.items():
                    if 'error' not in significance:
                        p_value = significance.get('p_value', 1.0)
                        is_significant = "✅ 有意" if significance.get('significant', False) else "❌ 非有意"
                        markdown += f"- vs {baseline_name}: {is_significant} (p={p_value:.4f})\n"
        
        markdown += "\n---\n*この結果は学術的研究基準に基づく客観的評価フレームワークによって生成されました。*"
        
        return markdown

# Factory function for Colab interface
def create_colab_interface() -> ColabEvaluationInterface:
    """Colab インターフェースの作成"""
    return ColabEvaluationInterface()

# Convenience functions for direct use in Colab cells
def quick_demo(sample_size: int = 20):
    """クイックデモ実行"""
    interface = create_colab_interface()
    interface.display_welcome_message()
    return interface.run_quick_evaluation_demo(sample_size)

def comprehensive_demo(datasets: List[str] = ["logic_puzzles"], sample_size: int = 30):
    """包括的デモ実行"""
    interface = create_colab_interface()
    interface.display_welcome_message()
    return interface.run_comprehensive_evaluation(datasets, sample_size)

def create_visual_report(result: ExperimentResult, save_path: str = "evaluation_report.png"):
    """ビジュアルレポート作成"""
    interface = create_colab_interface()
    interface.current_result = result
    interface.create_visual_report(save_path)

if __name__ == "__main__":
    # Test the interface
    print("Testing Colab Interface...")
    result = quick_demo(10)
    print(f"Demo completed: {result.config.name}")
