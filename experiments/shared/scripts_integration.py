"""
Scripts統合機能
==============================

scripts/experiments/のCLI機能と実験管理機能を統合
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# scripts/experiments/への参照を追加
scripts_experiments_path = Path(__file__).parent.parent.parent / "scripts" / "experiments"
if scripts_experiments_path.exists():
    sys.path.append(str(scripts_experiments_path))


def get_experiment_cli_manager():
    """scripts/experiments/のExperimentCLIを取得"""
    try:
        from experiment_cli import ExperimentCLI
        return ExperimentCLI()
    except ImportError:
        return None


def get_experiment_runner():
    """scripts/experiments/のExperimentRunnerを取得"""
    try:
        from experiment_runner import ExperimentRunner
        return ExperimentRunner()
    except ImportError:
        return None


def integrate_with_scripts_cli(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """scripts/experiments/のCLI機能との統合"""
    
    # ExperimentCLIとの統合
    cli_manager = get_experiment_cli_manager()
    if cli_manager:
        print("✅ scripts/experiments/ExperimentCLI統合済み")
        
        # データディレクトリの同期
        if hasattr(cli_manager, 'data_dir'):
            experiment_config['scripts_data_dir'] = str(cli_manager.data_dir)
        
        # 設定の統合
        if hasattr(cli_manager, 'config'):
            experiment_config.update(cli_manager.config)
    
    # ExperimentRunnerとの統合
    runner = get_experiment_runner()
    if runner:
        print("✅ scripts/experiments/ExperimentRunner統合済み")
        
        # LLMプロバイダーとの統合
        if hasattr(runner, 'llm_provider'):
            experiment_config['llm_provider'] = runner.llm_provider
    
    return experiment_config


def run_with_scripts_integration(experiment_func, experiment_config: Dict[str, Any]):
    """scripts/experiments/機能統合での実験実行"""
    
    # 統合設定
    integrated_config = integrate_with_scripts_cli(experiment_config)
    
    print("🔗 scripts/experiments/機能統合モードで実行")
    
    try:
        # scripts/experiments/のデータ管理機能を活用
        cli_manager = get_experiment_cli_manager()
        if cli_manager and hasattr(cli_manager, 'backup_data'):
            print("📦 scripts/experiments/のバックアップ機能を使用")
            backup_id = cli_manager.backup_data()
            integrated_config['scripts_backup_id'] = backup_id
        
        # 実験実行
        results = experiment_func(integrated_config)
        
        # scripts/experiments/のレポート機能を活用
        if cli_manager and hasattr(cli_manager, 'generate_report'):
            print("📊 scripts/experiments/のレポート機能でレポート生成")
            cli_manager.generate_report(results)
        
        return results
        
    except Exception as e:
        print(f"❌ scripts統合モードでエラー: {e}")
        
        # scripts/experiments/のエラー復旧機能
        if cli_manager and hasattr(cli_manager, 'restore_backup'):
            backup_id = integrated_config.get('scripts_backup_id')
            if backup_id:
                print("🔄 scripts/experiments/のバックアップから復旧")
                cli_manager.restore_backup(backup_id)
        
        raise


class ScriptsIntegratedExperiment:
    """scripts/experiments/統合実験クラス"""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any] = None):
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # scripts/experiments/機能の初期化
        self.cli_manager = get_experiment_cli_manager()
        self.runner = get_experiment_runner()
        
        if self.cli_manager:
            print(f"✅ {experiment_name}: ExperimentCLI統合完了")
        
        if self.runner:
            print(f"✅ {experiment_name}: ExperimentRunner統合完了")
    
    def run_experiment(self, experiment_func, **kwargs):
        """統合環境での実験実行"""
        
        if not self.cli_manager and not self.runner:
            print("⚠️  scripts/experiments/機能が利用不可 - 標準モードで実行")
            return experiment_func(self.config, **kwargs)
        
        # 統合設定の準備
        integrated_config = integrate_with_scripts_cli(self.config)
        integrated_config.update(kwargs)
        
        # scripts機能統合での実行
        return run_with_scripts_integration(
            lambda config: experiment_func(config, **kwargs),
            integrated_config
        )
    
    def get_data_directory(self) -> Optional[Path]:
        """統合データディレクトリの取得"""
        if self.cli_manager and hasattr(self.cli_manager, 'data_dir'):
            return Path(self.cli_manager.data_dir)
        return None
    
    def get_output_directory(self) -> Optional[Path]:
        """統合出力ディレクトリの取得"""
        if self.cli_manager and hasattr(self.cli_manager, 'outputs_dir'):
            return Path(self.cli_manager.outputs_dir)
        return None
    
    def create_experiment_report(self, results: Any) -> str:
        """統合レポート生成"""
        if self.cli_manager and hasattr(self.cli_manager, 'generate_detailed_report'):
            return self.cli_manager.generate_detailed_report(results)
        
        # フォールバック: 基本レポート
        return f"""
# {self.experiment_name} 実験レポート

## 実験結果
{results}

## 設定
{self.config}
        """


def check_scripts_availability() -> Dict[str, bool]:
    """scripts/experiments/機能の利用可能性チェック"""
    return {
        'experiment_cli': get_experiment_cli_manager() is not None,
        'experiment_runner': get_experiment_runner() is not None,
        'scripts_path_exists': scripts_experiments_path.exists()
    }


def print_scripts_integration_status():
    """scripts統合状況の表示"""
    status = check_scripts_availability()
    
    print("\n🔗 scripts/experiments/統合状況:")
    print(f"  📁 パス存在: {'✅' if status['scripts_path_exists'] else '❌'}")
    print(f"  🎮 ExperimentCLI: {'✅' if status['experiment_cli'] else '❌'}")
    print(f"  🏃 ExperimentRunner: {'✅' if status['experiment_runner'] else '❌'}")
    
    if all(status.values()):
        print("  🎉 完全統合モード利用可能")
    elif any(status.values()):
        print("  ⚠️  部分統合モード")
    else:
        print("  🔧 標準モード（統合機能なし）")
