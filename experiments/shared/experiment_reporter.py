"""
実験レポート生成・可視化ユーティリティ

実験結果のレポート生成、グラフ作成、論文用図表生成機能
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentReporter:
    """実験レポート生成クラス"""
    
    def __init__(self, output_dir: str = "./experiment_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_template = self._load_report_template()
    
    def _load_report_template(self) -> str:
        """レポートテンプレートの読み込み"""
        template = """
# {title}

## 実験概要
- **実験日時**: {timestamp}
- **実験タイプ**: {experiment_type}
- **実行者**: {executor}
- **目的**: {objective}

## 実験設定
{experimental_setup}

## 結果サマリー
{results_summary}

## 詳細結果
{detailed_results}

## パフォーマンス比較
{performance_comparison}

## 考察
{discussion}

## 結論
{conclusion}

## 次のステップ
{next_steps}

---
*このレポートは InsightSpike-AI 実験フレームワークにより自動生成されました*
"""
        return template
    
    def generate_comprehensive_report(self, experiment_data: Dict[str, Any],
                                    include_visualizations: bool = True) -> str:
        """包括的実験レポート生成"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # レポート内容構築
        report_content = {
            "title": experiment_data.get("title", "実験レポート"),
            "timestamp": timestamp,
            "experiment_type": experiment_data.get("type", "Unknown"),
            "executor": experiment_data.get("executor", "InsightSpike-AI"),
            "objective": experiment_data.get("objective", "性能評価"),
            "experimental_setup": self._format_experimental_setup(experiment_data.get("setup", {})),
            "results_summary": self._format_results_summary(experiment_data.get("results", {})),
            "detailed_results": self._format_detailed_results(experiment_data.get("results", {})),
            "performance_comparison": self._format_performance_comparison(experiment_data.get("comparisons", {})),
            "discussion": experiment_data.get("discussion", "詳細な分析が必要です。"),
            "conclusion": experiment_data.get("conclusion", "実験結果は期待された性能改善を示しています。"),
            "next_steps": experiment_data.get("next_steps", "追加実験とさらなる最適化が推奨されます。")
        }
        
        # レポート生成
        report = self.report_template.format(**report_content)
        
        # ファイル保存
        report_filename = f"experiment_report_{experiment_data.get('type', 'unknown')}_{timestamp.replace(':', '-').replace(' ', '_')}.md"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 可視化追加
        if include_visualizations and "results" in experiment_data:
            viz_dir = self.output_dir / f"visualizations_{experiment_data.get('type', 'unknown')}"
            viz_dir.mkdir(exist_ok=True)
            visualizer = ResultVisualizer(str(viz_dir))
            visualizer.create_experiment_visualizations(experiment_data["results"])
        
        logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)
    
    def _format_experimental_setup(self, setup: Dict[str, Any]) -> str:
        """実験設定のフォーマット"""
        if not setup:
            return "実験設定情報が提供されていません。"
        
        formatted = "```\n"
        for key, value in setup.items():
            formatted += f"{key}: {value}\n"
        formatted += "```\n"
        return formatted
    
    def _format_results_summary(self, results: Dict[str, Any]) -> str:
        """結果サマリーのフォーマット"""
        if not results:
            return "結果データが提供されていません。"
        
        summary = "### 主要指標\n\n"
        
        # 主要指標の抽出と表示
        if "metrics" in results:
            metrics = results["metrics"]
            summary += "| 指標 | 値 |\n|------|----|\n"
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    summary += f"| {metric} | {value:.4f} |\n"
                else:
                    summary += f"| {metric} | {value} |\n"
        
        # 改善率情報
        if "improvements" in results:
            improvements = results["improvements"]
            summary += "\n### 改善率\n\n"
            summary += "| 指標 | 改善率 |\n|------|--------|\n"
            
            for metric, improvement in improvements.items():
                summary += f"| {metric} | {improvement:.2%} |\n"
        
        return summary
    
    def _format_detailed_results(self, results: Dict[str, Any]) -> str:
        """詳細結果のフォーマット"""
        if not results:
            return "詳細結果データが提供されていません。"
        
        detailed = "### 実験データ\n\n"
        
        # JSON形式での結果表示
        detailed += "```json\n"
        detailed += json.dumps(results, indent=2, ensure_ascii=False, default=str)
        detailed += "\n```\n"
        
        return detailed
    
    def _format_performance_comparison(self, comparisons: Dict[str, Any]) -> str:
        """パフォーマンス比較のフォーマット"""
        if not comparisons:
            return "比較データが提供されていません。"
        
        comparison = "### 手法別性能比較\n\n"
        
        if "method_rankings" in comparisons:
            rankings = comparisons["method_rankings"]
            comparison += "| 手法 | 総合ランク |\n|------|------------|\n"
            
            sorted_methods = sorted(rankings.items(), key=lambda x: x[1])
            for method, rank in sorted_methods:
                comparison += f"| {method} | {rank:.2f} |\n"
        
        if "best_method_per_metric" in comparisons:
            best_methods = comparisons["best_method_per_metric"]
            comparison += "\n### 指標別最優秀手法\n\n"
            comparison += "| 指標 | 最優秀手法 | 値 |\n|------|------------|----|\n"
            
            for metric, info in best_methods.items():
                method = info.get("method", "Unknown")
                value = info.get("value", "N/A")
                if isinstance(value, float):
                    comparison += f"| {metric} | {method} | {value:.4f} |\n"
                else:
                    comparison += f"| {metric} | {method} | {value} |\n"
        
        return comparison
    
    def generate_phase_summary_report(self, phase_name: str, 
                                    experiments: List[Dict[str, Any]]) -> str:
        """フェーズサマリーレポート生成"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary_content = f"""
# {phase_name} - フェーズサマリーレポート

## 実験概要
- **フェーズ**: {phase_name}
- **生成日時**: {timestamp}
- **実験数**: {len(experiments)}

## 実行実験一覧
"""
        
        for i, exp in enumerate(experiments, 1):
            summary_content += f"### 実験 {i}: {exp.get('title', 'Untitled')}\n"
            summary_content += f"- **タイプ**: {exp.get('type', 'Unknown')}\n"
            summary_content += f"- **目的**: {exp.get('objective', 'N/A')}\n"
            summary_content += f"- **状態**: {exp.get('status', 'Completed')}\n\n"
        
        summary_content += """
## フェーズ全体の成果
- 各実験の詳細結果は個別レポートを参照
- 統合的な分析と考察は次フェーズで実施予定

## 次フェーズへの推奨事項
- 改善が必要な領域の特定
- 追加実験の必要性評価
- 手法の最適化検討
"""
        
        # レポート保存
        report_filename = f"phase_summary_{phase_name.lower().replace(' ', '_')}_{timestamp.replace(':', '-').replace(' ', '_')}.md"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"Phase summary report generated: {report_path}")
        return str(report_path)


class ResultVisualizer:
    """実験結果可視化クラス"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # カラーパレット設定
        self.colors = px.colors.qualitative.Set3
        self.color_map = {}
    
    def create_experiment_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """実験結果の包括的可視化"""
        visualization_files = []
        
        # 性能比較バーチャート
        if "method_results" in results:
            bar_chart_file = self.create_performance_bar_chart(results["method_results"])
            if bar_chart_file:
                visualization_files.append(bar_chart_file)
        
        # 改善率レーダーチャート
        if "improvements" in results:
            radar_chart_file = self.create_improvement_radar_chart(results["improvements"])
            if radar_chart_file:
                visualization_files.append(radar_chart_file)
        
        # 時系列パフォーマンス（あれば）
        if "timeline_data" in results:
            timeline_file = self.create_timeline_chart(results["timeline_data"])
            if timeline_file:
                visualization_files.append(timeline_file)
        
        # 散布図行列（複数指標間の関係）
        if "detailed_metrics" in results:
            scatter_matrix_file = self.create_scatter_matrix(results["detailed_metrics"])
            if scatter_matrix_file:
                visualization_files.append(scatter_matrix_file)
        
        return visualization_files
    
    def create_performance_bar_chart(self, method_results: Dict[str, Dict[str, float]]) -> Optional[str]:
        """性能比較バーチャート作成"""
        try:
            methods = list(method_results.keys())
            metrics = list(method_results[methods[0]].keys())
            
            # サブプロット作成
            fig = make_subplots(
                rows=len(metrics), cols=1,
                subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in metrics],
                vertical_spacing=0.05
            )
            
            for i, metric in enumerate(metrics, 1):
                method_names = []
                values = []
                
                for method in methods:
                    if metric in method_results[method]:
                        method_names.append(method)
                        values.append(method_results[method][metric])
                
                fig.add_trace(
                    go.Bar(
                        x=method_names,
                        y=values,
                        name=metric,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                title="実験手法別性能比較",
                height=200 * len(metrics),
                font=dict(size=12)
            )
            
            # ファイル保存
            filename = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Performance bar chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create performance bar chart: {e}")
            return None
    
    def create_improvement_radar_chart(self, improvements: Dict[str, float]) -> Optional[str]:
        """改善率レーダーチャート作成"""
        try:
            metrics = list(improvements.keys())
            values = [improvements[metric] * 100 for metric in metrics]  # パーセント表示
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='改善率 (%)',
                line_color='rgb(32, 201, 151)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-50, 50]  # -50%から+50%の範囲
                    )),
                title="各指標の改善率",
                font=dict(size=12)
            )
            
            # ファイル保存
            filename = f"improvement_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Improvement radar chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create improvement radar chart: {e}")
            return None
    
    def create_timeline_chart(self, timeline_data: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
        """時系列パフォーマンスチャート作成"""
        try:
            fig = go.Figure()
            
            for method_name, time_series in timeline_data.items():
                times, values = zip(*time_series)
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    name=method_name,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="時系列パフォーマンス推移",
                xaxis_title="時間 (秒)",
                yaxis_title="パフォーマンス指標",
                font=dict(size=12)
            )
            
            # ファイル保存
            filename = f"timeline_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Timeline chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create timeline chart: {e}")
            return None
    
    def create_scatter_matrix(self, detailed_metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """散布図行列作成"""
        try:
            # データフレーム作成
            df_data = []
            for method, metrics in detailed_metrics.items():
                metrics_copy = metrics.copy()
                metrics_copy['method'] = method
                df_data.append(metrics_copy)
            
            df = pd.DataFrame(df_data)
            
            # 数値列のみ選択
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for scatter matrix")
                return None
            
            fig = px.scatter_matrix(
                df, 
                dimensions=numeric_cols,
                color='method',
                title="指標間の相関関係"
            )
            
            fig.update_layout(
                width=800,
                height=800,
                font=dict(size=10)
            )
            
            # ファイル保存
            filename = f"scatter_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Scatter matrix saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create scatter matrix: {e}")
            return None
    
    def create_pathfinding_visualization(self, maze_data: Dict[str, Any],
                                       solutions: Dict[str, List[Tuple[int, int]]]) -> Optional[str]:
        """迷路経路探索結果可視化"""
        try:
            maze_grid = np.array(maze_data["grid"])
            start = maze_data["start"]
            goal = maze_data["goal"]
            
            fig = make_subplots(
                rows=1, cols=len(solutions),
                subplot_titles=list(solutions.keys()),
                specs=[[{"type": "heatmap"}] * len(solutions)]
            )
            
            for i, (method, path) in enumerate(solutions.items(), 1):
                # 迷路可視化用グリッド作成
                viz_grid = maze_grid.copy().astype(float)
                viz_grid[viz_grid == 1] = -1  # 壁
                viz_grid[viz_grid == 0] = 0   # 通路
                
                # 経路をマーク
                for x, y in path:
                    if 0 <= x < maze_grid.shape[0] and 0 <= y < maze_grid.shape[1]:
                        viz_grid[x, y] = 0.5
                
                # スタート・ゴールをマーク
                viz_grid[start[0], start[1]] = 1.0
                viz_grid[goal[0], goal[1]] = 0.8
                
                fig.add_trace(
                    go.Heatmap(
                        z=viz_grid,
                        colorscale=[[0, 'black'], [0.25, 'white'], [0.5, 'blue'], [0.8, 'red'], [1, 'green']],
                        showscale=i == 1
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(
                title=f"迷路経路探索結果比較 ({maze_data.get('size', 'N/A')}x{maze_data.get('size', 'N/A')})",
                height=400
            )
            
            # ファイル保存
            filename = f"pathfinding_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Pathfinding visualization saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create pathfinding visualization: {e}")
            return None
    
    def create_memory_efficiency_chart(self, memory_data: Dict[str, Dict[str, float]]) -> Optional[str]:
        """メモリ効率性可視化"""
        try:
            methods = list(memory_data.keys())
            
            # 3D散布図での可視化（メモリ使用量 vs 構築時間 vs 検索精度）
            fig = go.Figure()
            
            for method in methods:
                data = memory_data[method]
                
                fig.add_trace(go.Scatter3d(
                    x=[data.get("memory_usage", 0)],
                    y=[data.get("construction_time", 0)],
                    z=[data.get("retrieval_accuracy", 0)],
                    mode='markers+text',
                    marker=dict(size=15),
                    text=[method],
                    name=method
                ))
            
            fig.update_layout(
                title="メモリ効率性分析（3D）",
                scene=dict(
                    xaxis_title="メモリ使用量 (MB)",
                    yaxis_title="構築時間 (秒)",
                    zaxis_title="検索精度"
                )
            )
            
            # ファイル保存
            filename = f"memory_efficiency_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            logger.info(f"Memory efficiency chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create memory efficiency chart: {e}")
            return None
