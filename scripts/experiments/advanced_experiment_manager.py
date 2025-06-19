#!/usr/bin/env python3
"""
高度な実験管理システム
====================

パラメータスイープ、比較実験、レポート自動生成を統合した
包括的な実験管理ツール
"""

import sys
import argparse
import json
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import subprocess
import warnings
warnings.filterwarnings('ignore')

# グラフ設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedExperimentManager:
    """高度な実験管理クラス"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.outputs_dir = Path("experiments/outputs")
        self.scripts_dir = Path("scripts/experiments")
        
        # 設定ディレクトリ
        self.config_dir = self.scripts_dir / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # レポートディレクトリ
        self.reports_dir = self.outputs_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験テンプレート
        self.experiment_templates = {
            "quick_test": {
                "episodes": 100,
                "memory_dim": 384,
                "topk": 5,
                "ged_threshold": 0.15,
                "ig_threshold": 0.10,
                "similarity_threshold": 0.3
            },
            "standard": {
                "episodes": 500,
                "memory_dim": 384,
                "topk": 10,
                "ged_threshold": 0.15,
                "ig_threshold": 0.10,
                "similarity_threshold": 0.3
            },
            "comprehensive": {
                "episodes": 1000,
                "memory_dim": 384,
                "topk": 15,
                "ged_threshold": 0.15,
                "ig_threshold": 0.10,
                "similarity_threshold": 0.3
            },
            "high_sensitivity": {
                "episodes": 500,
                "memory_dim": 384,
                "topk": 10,
                "ged_threshold": 0.10,
                "ig_threshold": 0.05,
                "similarity_threshold": 0.25
            },
            "low_sensitivity": {
                "episodes": 500,
                "memory_dim": 384,
                "topk": 10,
                "ged_threshold": 0.25,
                "ig_threshold": 0.15,
                "similarity_threshold": 0.35
            }
        }
    
    def create_experiment_config(self, template_name: str, custom_params: Dict = None) -> Dict:
        """実験設定作成"""
        if template_name not in self.experiment_templates:
            raise ValueError(f"未知のテンプレート: {template_name}")
        
        config = self.experiment_templates[template_name].copy()
        if custom_params:
            config.update(custom_params)
        
        return config
    
    def save_experiment_config(self, config_name: str, config: Dict) -> Path:
        """実験設定保存"""
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 実験設定保存: {config_path}")
        return config_path
    
    def load_experiment_config(self, config_name: str) -> Dict:
        """実験設定読み込み"""
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def generate_parameter_sweep_configs(self, base_config: Dict, sweep_params: Dict[str, List]) -> List[Dict]:
        """パラメータスイープ設定生成"""
        import itertools
        
        # パラメータの組み合わせを生成
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        
        configs = []
        for combination in itertools.product(*param_values):
            config = base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            # 設定名を生成
            config_name = "_".join([f"{name}{value}" for name, value in zip(param_names, combination)])
            config['config_name'] = config_name
            configs.append(config)
        
        return configs
    
    def run_single_experiment(self, session_id: str, experiment_name: str, config: Dict, seed: int = 42) -> Dict:
        """単一実験実行"""
        print(f"🚀 実験実行開始: {experiment_name}")
        
        # 標準実験スクリプトを呼び出し
        cmd = [
            "python", str(self.scripts_dir / "run_standardized_experiment.py"),
            session_id, experiment_name,
            "--episodes", str(config["episodes"]),
            "--seed", str(seed),
            "--memory-dim", str(config["memory_dim"]),
            "--topk", str(config["topk"]),
            "--ged-threshold", str(config["ged_threshold"]),
            "--ig-threshold", str(config["ig_threshold"]),
            "--similarity-threshold", str(config["similarity_threshold"])
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ 実験完了: {experiment_name}")
                return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
            else:
                print(f"❌ 実験失敗: {experiment_name}")
                print(f"エラー: {result.stderr}")
                return {"status": "failed", "stdout": result.stdout, "stderr": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": "実験がタイムアウトしました"}
        except Exception as e:
            return {"status": "error", "stdout": "", "stderr": str(e)}
    
    def run_parameter_sweep(self, session_id: str, base_name: str, configs: List[Dict], 
                           parallel: bool = False, max_workers: int = 2) -> List[Dict]:
        """パラメータスイープ実行"""
        print(f"🔄 パラメータスイープ開始: {len(configs)}個の設定")
        
        results = []
        
        if parallel and len(configs) > 1:
            print(f"⚡ 並列実行 (最大{max_workers}並列)")
            # Note: 並列実行は慎重に実装する必要がある（リソース競合のため）
            # 今回は順次実行を基本とする
            parallel = False
        
        for i, config in enumerate(configs):
            experiment_name = f"{base_name}_{config.get('config_name', f'exp{i:03d}')}"
            
            print(f"\n--- {i+1}/{len(configs)}: {experiment_name} ---")
            print(f"設定: {config}")
            
            result = self.run_single_experiment(session_id, experiment_name, config)
            result['experiment_name'] = experiment_name
            result['config'] = config
            result['index'] = i
            
            results.append(result)
            
            # 失敗した場合の処理
            if result['status'] != 'success':
                print(f"⚠️ {experiment_name} が失敗しました。続行します...")
        
        print(f"\n✅ パラメータスイープ完了: {len(results)}実験")
        return results
    
    def collect_experiment_results(self, session_id: str, experiment_names: List[str]) -> pd.DataFrame:
        """実験結果収集"""
        session_dir = self.outputs_dir / session_id
        
        all_results = []
        
        for exp_name in experiment_names:
            exp_dir = session_dir / exp_name
            results_file = exp_dir / "06_experiment_results.json"
            
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                result['experiment_name'] = exp_name
                all_results.append(result)
            else:
                print(f"⚠️ 結果ファイルが見つかりません: {results_file}")
        
        if all_results:
            return pd.DataFrame(all_results)
        else:
            return pd.DataFrame()
    
    def generate_comparison_report(self, session_id: str, experiment_names: List[str], 
                                 report_name: str = "comparison_report") -> Path:
        """比較レポート生成"""
        print(f"📊 比較レポート生成中: {report_name}")
        
        # 実験結果収集
        results_df = self.collect_experiment_results(session_id, experiment_names)
        
        if results_df.empty:
            print("❌ 比較可能な実験結果がありません")
            return None
        
        # レポートディレクトリ作成
        report_dir = self.reports_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(exist_ok=True)
        
        # 1. 基本統計
        summary_stats = {
            'total_experiments': len(results_df),
            'total_episodes': results_df['total_episodes'].sum(),
            'total_insights': results_df['total_insights'].sum(),
            'avg_insight_rate': results_df['insight_rate'].mean(),
            'avg_processing_speed': results_df['avg_episodes_per_second'].mean(),
            'session_id': session_id,
            'report_generated': datetime.now().isoformat()
        }
        
        # 2. 詳細結果CSV
        results_df.to_csv(report_dir / "experiment_results.csv", index=False)
        
        # 3. 統計サマリーJSON
        with open(report_dir / "summary_stats.json", 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        # 4. 可視化グラフ
        self._generate_comparison_plots(results_df, report_dir)
        
        # 5. HTMLレポート生成
        html_report = self._generate_html_report(results_df, summary_stats, report_dir)
        
        print(f"✅ 比較レポート生成完了:")
        print(f"   📁 レポートディレクトリ: {report_dir}")
        print(f"   📄 HTMLレポート: {html_report}")
        
        return report_dir
    
    def _generate_comparison_plots(self, results_df: pd.DataFrame, report_dir: Path):
        """比較グラフ生成"""
        # 図のサイズとスタイル設定
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. 洞察検出率比較
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results_df)), results_df['insight_rate'], 
                color=plt.cm.viridis(np.linspace(0, 1, len(results_df))))
        plt.xlabel('実験')
        plt.ylabel('洞察検出率')
        plt.title('実験別洞察検出率比較')
        plt.xticks(range(len(results_df)), results_df['experiment_name'], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(report_dir / "insight_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 処理速度比較
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results_df)), results_df['avg_episodes_per_second'],
                color=plt.cm.plasma(np.linspace(0, 1, len(results_df))))
        plt.xlabel('実験')
        plt.ylabel('処理速度 (episodes/sec)')
        plt.title('実験別処理速度比較')
        plt.xticks(range(len(results_df)), results_df['experiment_name'], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(report_dir / "processing_speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 散布図: 洞察率 vs 処理速度
        plt.figure(figsize=(10, 8))
        plt.scatter(results_df['insight_rate'], results_df['avg_episodes_per_second'], 
                   s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
        
        for i, row in results_df.iterrows():
            plt.annotate(row['experiment_name'], 
                        (row['insight_rate'], row['avg_episodes_per_second']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('洞察検出率')
        plt.ylabel('処理速度 (episodes/sec)')
        plt.title('洞察検出率 vs 処理速度')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "insight_vs_speed_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 設定パラメータの影響（可能な場合）
        config_columns = []
        for col in results_df.columns:
            if col.startswith('config.'):
                config_columns.append(col)
        
        if config_columns:
            n_params = len(config_columns)
            fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 10))
            axes = axes.flatten() if n_params > 1 else [axes]
            
            for i, param in enumerate(config_columns):
                if i < len(axes):
                    axes[i].scatter(results_df[param], results_df['insight_rate'], alpha=0.7)
                    axes[i].set_xlabel(param.replace('config.', ''))
                    axes[i].set_ylabel('洞察検出率')
                    axes[i].set_title(f'{param.replace("config.", "")} の影響')
                    axes[i].grid(True, alpha=0.3)
            
            # 余ったサブプロットを非表示
            for i in range(len(config_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(report_dir / "parameter_effects.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self, results_df: pd.DataFrame, summary_stats: Dict, 
                            report_dir: Path) -> Path:
        """HTMLレポート生成"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>InsightSpike-AI 実験比較レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .experiment {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .graph {{ text-align: center; margin: 20px 0; }}
        .graph img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 InsightSpike-AI 実験比較レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p>セッションID: {summary_stats['session_id']}</p>
    </div>
    
    <div class="summary">
        <h2>📊 実験サマリー</h2>
        <div class="metric">
            <div class="metric-value">{summary_stats['total_experiments']}</div>
            <div class="metric-label">実験数</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary_stats['total_episodes']:,}</div>
            <div class="metric-label">総エピソード数</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary_stats['total_insights']:,}</div>
            <div class="metric-label">総洞察数</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary_stats['avg_insight_rate']:.3f}</div>
            <div class="metric-label">平均洞察率</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary_stats['avg_processing_speed']:.1f}</div>
            <div class="metric-label">平均処理速度 (eps/sec)</div>
        </div>
    </div>
    
    <h2>📈 比較グラフ</h2>
    <div class="graph">
        <h3>洞察検出率比較</h3>
        <img src="insight_rate_comparison.png" alt="洞察検出率比較">
    </div>
    
    <div class="graph">
        <h3>処理速度比較</h3>
        <img src="processing_speed_comparison.png" alt="処理速度比較">
    </div>
    
    <div class="graph">
        <h3>洞察検出率 vs 処理速度</h3>
        <img src="insight_vs_speed_scatter.png" alt="洞察検出率 vs 処理速度">
    </div>
    
    <h2>📋 詳細結果</h2>
    <table>
        <tr>
            <th>実験名</th>
            <th>エピソード数</th>
            <th>洞察数</th>
            <th>洞察率</th>
            <th>処理速度</th>
            <th>実行時間</th>
        </tr>"""
        
        for _, row in results_df.iterrows():
            html_content += f"""
        <tr>
            <td>{row['experiment_name']}</td>
            <td>{row['total_episodes']:,}</td>
            <td>{row['total_insights']:,}</td>
            <td>{row['insight_rate']:.4f}</td>
            <td>{row['avg_episodes_per_second']:.2f}</td>
            <td>{row['total_time_seconds']:.1f}s</td>
        </tr>"""
        
        html_content += """
    </table>
    
    <div class="summary">
        <h2>🎯 結論と推奨事項</h2>
        <ul>"""
        
        # 自動推奨事項生成
        best_insight_exp = results_df.loc[results_df['insight_rate'].idxmax()]
        best_speed_exp = results_df.loc[results_df['avg_episodes_per_second'].idxmax()]
        
        html_content += f"""
            <li><strong>最高洞察率:</strong> {best_insight_exp['experiment_name']} ({best_insight_exp['insight_rate']:.4f})</li>
            <li><strong>最高処理速度:</strong> {best_speed_exp['experiment_name']} ({best_speed_exp['avg_episodes_per_second']:.2f} eps/sec)</li>"""
        
        if results_df['insight_rate'].std() > 0.01:
            html_content += "<li>洞察率に大きな変動があります。パラメータチューニングの余地があります。</li>"
        
        if results_df['avg_episodes_per_second'].std() > 1.0:
            html_content += "<li>処理速度に大きな変動があります。実行環境の最適化を検討してください。</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>"""
        
        html_path = report_dir / "report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="高度な実験管理システム")
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # テンプレート一覧
    subparsers.add_parser("list-templates", help="実験テンプレート一覧")
    
    # 実験設定作成
    config_parser = subparsers.add_parser("create-config", help="実験設定作成")
    config_parser.add_argument("config_name", help="設定名")
    config_parser.add_argument("template", help="テンプレート名")
    config_parser.add_argument("--custom", help="カスタムパラメータ (JSON形式)")
    
    # パラメータスイープ実行
    sweep_parser = subparsers.add_parser("run-sweep", help="パラメータスイープ実行")
    sweep_parser.add_argument("session_id", help="セッションID")
    sweep_parser.add_argument("base_name", help="実験ベース名")
    sweep_parser.add_argument("config_name", help="ベース設定名")
    sweep_parser.add_argument("--sweep-params", help="スイープパラメータ (JSON形式)")
    
    # 比較レポート生成
    report_parser = subparsers.add_parser("generate-report", help="比較レポート生成")
    report_parser.add_argument("session_id", help="セッションID")
    report_parser.add_argument("experiments", nargs="+", help="実験名リスト")
    report_parser.add_argument("--report-name", default="comparison_report", help="レポート名")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = AdvancedExperimentManager()
    
    try:
        if args.command == "list-templates":
            print("📋 利用可能な実験テンプレート:")
            for name, config in manager.experiment_templates.items():
                print(f"   🔸 {name}:")
                for key, value in config.items():
                    print(f"      {key}: {value}")
                print()
        
        elif args.command == "create-config":
            custom_params = {}
            if args.custom:
                custom_params = json.loads(args.custom)
            
            config = manager.create_experiment_config(args.template, custom_params)
            manager.save_experiment_config(args.config_name, config)
        
        elif args.command == "run-sweep":
            base_config = manager.load_experiment_config(args.config_name)
            
            if args.sweep_params:
                sweep_params = json.loads(args.sweep_params)
            else:
                # デフォルトのパラメータスイープ
                sweep_params = {
                    "ged_threshold": [0.10, 0.15, 0.20],
                    "ig_threshold": [0.05, 0.10, 0.15]
                }
            
            configs = manager.generate_parameter_sweep_configs(base_config, sweep_params)
            results = manager.run_parameter_sweep(args.session_id, args.base_name, configs)
            
            print(f"\n📊 スイープ結果:")
            success_count = sum(1 for r in results if r['status'] == 'success')
            print(f"   成功: {success_count}/{len(results)}")
        
        elif args.command == "generate-report":
            report_dir = manager.generate_comparison_report(
                args.session_id, args.experiments, args.report_name
            )
            
            if report_dir:
                html_report = report_dir / "report.html"
                print(f"\n🌐 HTMLレポートを開くには:")
                print(f"   open {html_report}")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
