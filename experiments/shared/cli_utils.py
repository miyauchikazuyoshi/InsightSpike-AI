"""
共通実験CLI機能
==============================

各Phase実験で共通利用可能なCLI機能とユーティリティ
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List


def create_base_cli_parser(phase_name: str, description: str) -> argparse.ArgumentParser:
    """基本CLI引数パーサーの作成"""
    parser = argparse.ArgumentParser(
        description=f"{phase_name}: {description}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用例:
  python {phase_name.lower()}_experiment.py                    # 標準実行
  python {phase_name.lower()}_experiment.py --debug           # デバッグモード  
  python {phase_name.lower()}_experiment.py --quick           # クイックテスト
  python {phase_name.lower()}_experiment.py --config config.json # 設定ファイル使用
  python {phase_name.lower()}_experiment.py --no-backup       # バックアップなし（高速）
  python {phase_name.lower()}_experiment.py --export json     # JSON形式でエクスポート
        """
    )
    
    # 基本オプション
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモードで実行（詳細ログ出力）')
    
    parser.add_argument('--output', type=str, 
                       default=f"experiments/{phase_name.lower()}/results",
                       help=f'結果出力ディレクトリ（デフォルト: experiments/{phase_name.lower()}/results）')
    
    # 高度なオプション
    parser.add_argument('--config', type=str,
                       help='JSON設定ファイルのパス')
    
    parser.add_argument('--export', choices=['csv', 'json', 'excel'], 
                       default='csv',
                       help='結果エクスポート形式（デフォルト: csv）')
    
    parser.add_argument('--no-backup', action='store_true',
                       help='データバックアップをスキップ（高速実行、非推奨）')
    
    parser.add_argument('--selective-copy', nargs='+',
                       default=["processed", "embedding", "models"],
                       help='実験用にコピーするデータフォルダ（デフォルト: processed embedding models）')
    
    # 実験制御オプション
    parser.add_argument('--quick', action='store_true',
                       help='クイックテスト（小さなデータサイズのみ）')
    
    # レポート生成オプション
    parser.add_argument('--no-report', action='store_true',
                       help='詳細レポート生成をスキップ')
    
    parser.add_argument('--plot', action='store_true',
                       help='性能グラフを生成（matplotlib必要）')
    
    # 実験監視オプション
    parser.add_argument('--monitor', action='store_true',
                       help='実験実行中のリソース監視を有効化')
    
    parser.add_argument('--save-intermediate', action='store_true',
                       help='中間結果の保存を有効化')
    
    return parser


def add_phase_specific_args(parser: argparse.ArgumentParser, phase: str) -> argparse.ArgumentParser:
    """フェーズ固有の引数を追加"""
    
    if phase == "phase1":
        parser.add_argument('--sizes', type=int, nargs='+', 
                           default=[50, 100, 200, 500],
                           help='テストする文書サイズ（デフォルト: 50 100 200 500）')
        
        parser.add_argument('--runs', type=int, default=1,
                           help='実験実行回数（平均値計算用、デフォルト: 1）')
        
        parser.add_argument('--baseline-only', action='store_true',
                           help='ベースラインRAGのみテスト')
        
        parser.add_argument('--insightspike-only', action='store_true',
                           help='InsightSpikeのみテスト')
    
    elif phase == "phase2":
        parser.add_argument('--benchmarks', nargs='+',
                           default=['ms_marco', 'natural_questions', 'hotpot_qa'],
                           help='実行するベンチマーク（デフォルト: ms_marco natural_questions hotpot_qa）')
        
        parser.add_argument('--rag-systems', nargs='+',
                           default=['langchain', 'llamaindex', 'haystack'],
                           help='比較するRAGシステム（デフォルト: langchain llamaindex haystack）')
        
        parser.add_argument('--sample-size', type=int, default=100,
                           help='ベンチマークサンプルサイズ（デフォルト: 100）')
    
    elif phase == "phase3":
        parser.add_argument('--maze-sizes', nargs='+', type=int,
                           default=[10, 20, 50, 100],
                           help='迷路サイズ（デフォルト: 10 20 50 100）')
        
        parser.add_argument('--algorithms', nargs='+',
                           default=['astar', 'dijkstra', 'genetic', 'reinforcement'],
                           help='比較アルゴリズム（デフォルト: astar dijkstra genetic reinforcement）')
        
        parser.add_argument('--maze-count', type=int, default=10,
                           help='各サイズの迷路生成数（デフォルト: 10）')
        
        # GIF アニメーション出力オプション
        parser.add_argument('--animate', action='store_true',
                           help='A* と GEDIG の経路比較GIFを生成')
    
    elif phase == "phase4":
        parser.add_argument('--previous-results', nargs='+',
                           help='統合する前フェーズの結果ディレクトリ')
        
        parser.add_argument('--meta-analysis', action='store_true',
                           help='メタ分析を実行')
        
        parser.add_argument('--paper-format', action='store_true',
                           help='論文用フォーマットで出力')
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """設定ファイルの読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 設定ファイル読み込み完了: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return {}


def merge_cli_config(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    """CLI引数と設定ファイルのマージ"""
    config = {}
    
    # 設定ファイルがある場合は読み込み
    if hasattr(args, 'config') and args.config:
        config = load_config_file(args.config)
    
    # 共通設定
    config.update({
        'debug': args.debug,
        'export_format': args.export,
        'output_dir': args.output,
        'no_backup': args.no_backup,
        'selective_copy': args.selective_copy,
        'quick_mode': args.quick,
        'generate_report': not args.no_report,
        'generate_plots': args.plot,
        'monitor_resources': getattr(args, 'monitor', False),
        'save_intermediate': getattr(args, 'save_intermediate', False)
    })
    
    # フェーズ固有設定
    if phase == "phase1":
        config.update({
            'document_sizes': getattr(args, 'sizes', [50, 100, 200, 500]),
            'num_runs': getattr(args, 'runs', 1),
            'baseline_only': getattr(args, 'baseline_only', False),
            'insightspike_only': getattr(args, 'insightspike_only', False)
        })
        
        # クイックモードの場合は小さなサイズに制限
        if args.quick:
            config['document_sizes'] = [50, 100]
            config['num_runs'] = 1
    
    elif phase == "phase2":
        config.update({
            'benchmarks': getattr(args, 'benchmarks', ['ms_marco', 'natural_questions', 'hotpot_qa']),
            'rag_systems': getattr(args, 'rag_systems', ['langchain', 'llamaindex', 'haystack']),
            'sample_size': getattr(args, 'sample_size', 100)
        })
        
        if args.quick:
            config['benchmarks'] = ['ms_marco']
            config['sample_size'] = 20
    
    elif phase == "phase3":
        config.update({
            'maze_sizes': getattr(args, 'maze_sizes', [10, 20, 50, 100]),
            'algorithms': getattr(args, 'algorithms', ['astar', 'dijkstra', 'genetic', 'reinforcement']),
            'maze_count': getattr(args, 'maze_count', 10),
            'animate': getattr(args, 'animate', False)
        })
        
        if args.quick:
            config['maze_sizes'] = [10, 20]
            config['maze_count'] = 3
    
    elif phase == "phase4":
        config.update({
            'previous_results': getattr(args, 'previous_results', None),
            'meta_analysis': getattr(args, 'meta_analysis', False),
            'paper_format': getattr(args, 'paper_format', False)
        })
    
    return config


def print_experiment_header(phase_name: str, config: Dict[str, Any]):
    """実験開始ヘッダーの表示"""
    print(f"🔬 {phase_name}")
    print("=" * 50)
    print(f"💾 出力形式: {config['export_format']}")
    print(f"🛡️  データバックアップ: {'無効' if config['no_backup'] else '有効'}")
    print(f"🐛 デバッグモード: {'有効' if config['debug'] else '無効'}")
    print(f"⚡ クイックモード: {'有効' if config['quick_mode'] else '無効'}")
    print(f"📊 レポート生成: {'無効' if not config['generate_report'] else '有効'}")
    print(f"📈 グラフ生成: {'有効' if config['generate_plots'] else '無効'}")


def handle_cli_error(e: Exception, config: Dict[str, Any]):
    """CLI実行エラーのハンドリング"""
    print(f"\n❌ 実験が失敗しました: {e}")
    if config.get('debug', False):
        import traceback
        traceback.print_exc()
    print("🔄 データは自動的に実験前の状態に復元されました")


def create_experiment_summary(results: Any, phase: str) -> str:
    """実験結果サマリーの作成"""
    summary = f"\n📊 {phase.upper()} 結果サマリー:\n"
    
    try:
        # 結果の形式に応じて処理
        if hasattr(results, 'to_dict'):
            # pandas DataFrame
            summary += f"✅ 実験完了: {len(results)} レコード\n"
        elif isinstance(results, dict):
            summary += f"✅ 実験完了: {len(results.get('results', []))} レコード\n"
        else:
            summary += "✅ 実験完了\n"
        
        summary += "📁 結果は以下に保存されています:\n"
        summary += "  - experiment_data/ (実験結果)\n"
        summary += "  - data_backups/ (バックアップ)\n"
        
    except Exception as e:
        summary += f"⚠️  サマリー生成エラー: {e}\n"
    
    return summary
