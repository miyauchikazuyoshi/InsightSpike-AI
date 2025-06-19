#!/usr/bin/env python3
"""
InsightSpike-AI 実験統合コマンド
===============================

すべての実験管理機能を統合したマスターコマンド
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List


class ExperimentMaster:
    """実験統合管理クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts" / "experiments"
        
        # 利用可能なスクリプト
        self.scripts = {
            "cli": self.scripts_dir / "experiment_cli.py",
            "run": self.scripts_dir / "run_standardized_experiment.py",
            "advanced": self.scripts_dir / "advanced_experiment_manager.py",
            "workflow": self.scripts_dir / "automated_workflow.py"
        }
    
    def run_script(self, script_name: str, args: List[str]) -> int:
        """スクリプト実行"""
        if script_name not in self.scripts:
            print(f"❌ 未知のスクリプト: {script_name}")
            return 1
        
        script_path = self.scripts[script_name]
        if not script_path.exists():
            print(f"❌ スクリプトが見つかりません: {script_path}")
            return 1
        
        cmd = ["python", str(script_path)] + args
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            print(f"❌ スクリプト実行エラー: {e}")
            return 1
    
    def show_help(self):
        """ヘルプ表示"""
        help_text = """
🔬 InsightSpike-AI 実験統合コマンド
=====================================

基本データ管理:
  exp status                    # データ状態確認
  exp check                     # データ整合性チェック
  exp clean                     # データクリーンアップ
  exp backup <name>            # データバックアップ
  exp restore <backup_id>      # データ復元
  exp build-memory [options]   # 初期メモリ構築

実験セッション管理:
  exp create-session <name>    # 実験セッション作成
  exp list-backups            # バックアップ一覧

単発実験実行:
  exp run <session_id> <name> [options]  # 標準実験実行

高度な実験管理:
  exp list-templates                     # 実験テンプレート一覧
  exp create-config <name> <template>    # 実験設定作成
  exp run-sweep <session> <base> <config> # パラメータスイープ
  exp generate-report <session> <exps...> # 比較レポート生成

自動化ワークフロー:
  exp workflow list                      # ワークフロー一覧
  exp workflow <name> [--session-id ID]  # ワークフロー実行

クイックスタート例:
  # 1. 高速比較実験
  exp workflow quick_comparison
  
  # 2. パラメータ感度分析
  exp workflow parameter_sensitivity
  
  # 3. カスタム実験
  exp create-session my_test
  exp build-memory --episodes 50
  exp run my_test baseline --episodes 200

オプション詳細は各コマンドで --help を参照してください。
        """
        print(help_text)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="InsightSpike-AI 実験統合コマンド",
        add_help=False
    )
    parser.add_argument("command", nargs="?", help="実行コマンド")
    parser.add_argument("args", nargs="*", help="コマンド引数")
    parser.add_argument("--help", "-h", action="store_true", help="ヘルプ表示")
    
    args = parser.parse_args()
    
    master = ExperimentMaster()
    
    # ヘルプ表示
    if args.help or not args.command:
        master.show_help()
        return 0
    
    # コマンドルーティング
    command = args.command
    remaining_args = args.args
    
    # 基本データ管理コマンド
    basic_commands = [
        "status", "check", "clean", "backup", "restore", 
        "build-memory", "create-session", "list-backups"
    ]
    
    if command in basic_commands:
        return master.run_script("cli", [command] + remaining_args)
    
    # 単発実験実行
    elif command == "run":
        return master.run_script("run", remaining_args)
    
    # 高度な実験管理
    elif command in ["list-templates", "create-config", "run-sweep", "generate-report"]:
        return master.run_script("advanced", [command] + remaining_args)
    
    # ワークフロー実行
    elif command == "workflow":
        if not remaining_args:
            return master.run_script("workflow", ["--list-workflows"])
        elif remaining_args[0] == "list":
            return master.run_script("workflow", ["--list-workflows"])
        else:
            return master.run_script("workflow", remaining_args)
    
    else:
        print(f"❌ 未知のコマンド: {command}")
        print("利用可能なコマンドを確認するには: exp --help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
