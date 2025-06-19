#!/usr/bin/env python3
"""
完全自動化実験ワークフロー
========================

データ準備から実験実行、結果分析まで一貫して実行する
ワンクリック実験システム
"""

import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class AutomatedExperimentWorkflow:
    """自動化実験ワークフロークラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts" / "experiments"
        self.data_dir = self.project_root / "data"
        self.outputs_dir = self.project_root / "experiments" / "outputs"
        
        # ワークフロー設定
        self.workflow_configs = {
            "quick_comparison": {
                "description": "高速比較実験（デバッグ用）",
                "experiments": [
                    {"name": "baseline", "template": "quick_test", "seed": 42},
                    {"name": "high_sensitivity", "template": "high_sensitivity", "custom": {"episodes": 100}},
                    {"name": "low_sensitivity", "template": "low_sensitivity", "custom": {"episodes": 100}}
                ]
            },
            "parameter_sensitivity": {
                "description": "パラメータ感度分析",
                "experiments": [
                    {"name": "standard", "template": "standard", "seed": 42},
                    {"name": "ged_010", "template": "standard", "custom": {"ged_threshold": 0.10}, "seed": 42},
                    {"name": "ged_015", "template": "standard", "custom": {"ged_threshold": 0.15}, "seed": 42},
                    {"name": "ged_020", "template": "standard", "custom": {"ged_threshold": 0.20}, "seed": 42},
                    {"name": "ig_005", "template": "standard", "custom": {"ig_threshold": 0.05}, "seed": 42},
                    {"name": "ig_010", "template": "standard", "custom": {"ig_threshold": 0.10}, "seed": 42},
                    {"name": "ig_015", "template": "standard", "custom": {"ig_threshold": 0.15}, "seed": 42}
                ]
            },
            "memory_initialization_study": {
                "description": "初期メモリサイズの影響調査",
                "experiments": [
                    {"name": "init_0", "template": "standard", "initial_episodes": 0, "seed": 42},
                    {"name": "init_25", "template": "standard", "initial_episodes": 25, "seed": 42},
                    {"name": "init_50", "template": "standard", "initial_episodes": 50, "seed": 42},
                    {"name": "init_100", "template": "standard", "initial_episodes": 100, "seed": 42}
                ]
            },
            "comprehensive_evaluation": {
                "description": "包括的評価実験",
                "experiments": [
                    {"name": "baseline", "template": "comprehensive", "seed": 42},
                    {"name": "optimized", "template": "comprehensive", 
                     "custom": {"ged_threshold": 0.12, "ig_threshold": 0.08, "topk": 12}, "seed": 42},
                    {"name": "high_precision", "template": "comprehensive",
                     "custom": {"ged_threshold": 0.20, "ig_threshold": 0.15, "topk": 8}, "seed": 42}
                ]
            }
        }
    
    def run_cli_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """CLIコマンド実行"""
        try:
            print(f"📋 実行中: {' '.join(command)}")
            result = subprocess.run(
                command, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.project_root,
                timeout=1800  # 30分タイムアウト
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"⏰ コマンドタイムアウト: {' '.join(command)}")
            raise
        except Exception as e:
            print(f"❌ コマンド実行エラー: {e}")
            raise
    
    def prepare_environment(self, session_id: str, clean_data: bool = True) -> bool:
        """実験環境準備"""
        print(f"🛠️ 実験環境準備開始: {session_id}")
        
        try:
            # 1. データ整合性チェック
            print("1️⃣ データ整合性チェック...")
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "experiment_cli.py"), "check"
            ])
            
            # 2. データクリーンアップ（必要な場合）
            if clean_data:
                print("2️⃣ データクリーンアップ...")
                result = self.run_cli_command([
                    "python", str(self.scripts_dir / "experiment_cli.py"), "clean"
                ])
                if result.returncode != 0:
                    print(f"❌ データクリーンアップ失敗: {result.stderr}")
                    return False
            
            # 3. 実験セッション作成
            print("3️⃣ 実験セッション作成...")
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "experiment_cli.py"), 
                "create-session", session_id, 
                "--description", f"自動化ワークフロー実験 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            if result.returncode != 0:
                print(f"❌ セッション作成失敗: {result.stderr}")
                return False
            
            print("✅ 実験環境準備完了")
            return True
            
        except Exception as e:
            print(f"❌ 環境準備エラー: {e}")
            return False
    
    def setup_initial_memory(self, episodes_count: int, seed: int = 42) -> bool:
        """初期メモリ構築"""
        if episodes_count == 0:
            print("⏭️ 初期メモリ構築をスキップ（0エピソード）")
            return True
        
        print(f"🧠 初期メモリ構築: {episodes_count}エピソード (seed={seed})")
        
        try:
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "experiment_cli.py"), 
                "build-memory", 
                "--episodes", str(episodes_count),
                "--seed", str(seed)
            ])
            
            if result.returncode != 0:
                print(f"❌ 初期メモリ構築失敗: {result.stderr}")
                return False
            
            print("✅ 初期メモリ構築完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期メモリ構築エラー: {e}")
            return False
    
    def run_single_experiment_workflow(self, session_id: str, experiment_config: Dict) -> Dict[str, Any]:
        """単一実験ワークフロー実行"""
        exp_name = experiment_config["name"]
        template = experiment_config["template"]
        seed = experiment_config.get("seed", 42)
        custom_params = experiment_config.get("custom", {})
        initial_episodes = experiment_config.get("initial_episodes", None)
        
        print(f"\n🚀 実験開始: {exp_name}")
        print(f"   テンプレート: {template}")
        print(f"   シード: {seed}")
        if custom_params:
            print(f"   カスタム設定: {custom_params}")
        if initial_episodes is not None:
            print(f"   初期エピソード: {initial_episodes}")
        
        workflow_result = {
            "experiment_name": exp_name,
            "template": template,
            "seed": seed,
            "custom_params": custom_params,
            "initial_episodes": initial_episodes,
            "start_time": datetime.now().isoformat(),
            "status": "unknown",
            "steps": {}
        }
        
        try:
            # ステップ1: データクリーンアップ
            print("1️⃣ データクリーンアップ...")
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "experiment_cli.py"), "clean"
            ])
            workflow_result["steps"]["cleanup"] = {"status": "success" if result.returncode == 0 else "failed"}
            
            # ステップ2: 初期メモリ構築（必要な場合）
            if initial_episodes is not None:
                if not self.setup_initial_memory(initial_episodes, seed):
                    workflow_result["status"] = "failed"
                    workflow_result["steps"]["initial_memory"] = {"status": "failed"}
                    return workflow_result
                workflow_result["steps"]["initial_memory"] = {"status": "success", "episodes": initial_episodes}
            
            # ステップ3: 実験設定作成
            print("2️⃣ 実験設定作成...")
            config_name = f"{exp_name}_config"
            custom_json = json.dumps(custom_params) if custom_params else "{}"
            
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "advanced_experiment_manager.py"),
                "create-config", config_name, template,
                "--custom", custom_json
            ])
            
            if result.returncode != 0:
                workflow_result["status"] = "config_failed"
                workflow_result["steps"]["config"] = {"status": "failed", "error": result.stderr}
                return workflow_result
            
            workflow_result["steps"]["config"] = {"status": "success", "config_name": config_name}
            
            # ステップ4: 実験実行
            print("3️⃣ 実験実行...")
            config = {
                "episodes": custom_params.get("episodes", 500),
                "memory_dim": custom_params.get("memory_dim", 384),
                "topk": custom_params.get("topk", 10),
                "ged_threshold": custom_params.get("ged_threshold", 0.15),
                "ig_threshold": custom_params.get("ig_threshold", 0.10),
                "similarity_threshold": custom_params.get("similarity_threshold", 0.3)
            }
            
            result = self.run_cli_command([
                "python", str(self.scripts_dir / "run_standardized_experiment.py"),
                session_id, exp_name,
                "--episodes", str(config["episodes"]),
                "--seed", str(seed),
                "--memory-dim", str(config["memory_dim"]),
                "--topk", str(config["topk"]),
                "--ged-threshold", str(config["ged_threshold"]),
                "--ig-threshold", str(config["ig_threshold"]),
                "--similarity-threshold", str(config["similarity_threshold"])
            ])
            
            if result.returncode == 0:
                workflow_result["status"] = "success"
                workflow_result["steps"]["experiment"] = {"status": "success"}
                print(f"✅ 実験完了: {exp_name}")
            else:
                workflow_result["status"] = "experiment_failed"
                workflow_result["steps"]["experiment"] = {"status": "failed", "error": result.stderr}
                print(f"❌ 実験失敗: {exp_name}")
                print(f"エラー: {result.stderr}")
            
            workflow_result["end_time"] = datetime.now().isoformat()
            return workflow_result
            
        except Exception as e:
            workflow_result["status"] = "error"
            workflow_result["error"] = str(e)
            workflow_result["end_time"] = datetime.now().isoformat()
            print(f"❌ ワークフローエラー: {e}")
            return workflow_result
    
    def run_workflow(self, workflow_name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """完全自動化ワークフロー実行"""
        if workflow_name not in self.workflow_configs:
            raise ValueError(f"未知のワークフロー: {workflow_name}")
        
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{workflow_name}_{timestamp}"
        
        workflow_config = self.workflow_configs[workflow_name]
        
        print(f"🎯 自動化ワークフロー開始: {workflow_name}")
        print(f"📋 説明: {workflow_config['description']}")
        print(f"🆔 セッションID: {session_id}")
        print(f"🔢 実験数: {len(workflow_config['experiments'])}")
        print("=" * 60)
        
        workflow_result = {
            "workflow_name": workflow_name,
            "session_id": session_id,
            "description": workflow_config["description"],
            "start_time": datetime.now().isoformat(),
            "experiments": [],
            "status": "unknown"
        }
        
        try:
            # 環境準備
            if not self.prepare_environment(session_id, clean_data=True):
                workflow_result["status"] = "environment_failed"
                return workflow_result
            
            # 各実験実行
            successful_experiments = []
            failed_experiments = []
            
            for i, exp_config in enumerate(workflow_config["experiments"]):
                print(f"\n{'='*20} {i+1}/{len(workflow_config['experiments'])} {'='*20}")
                
                exp_result = self.run_single_experiment_workflow(session_id, exp_config)
                workflow_result["experiments"].append(exp_result)
                
                if exp_result["status"] == "success":
                    successful_experiments.append(exp_result["experiment_name"])
                else:
                    failed_experiments.append(exp_result["experiment_name"])
                
                # 実験間の休憩（リソース回復）
                if i < len(workflow_config["experiments"]) - 1:
                    print("⏸️ 次の実験まで5秒待機...")
                    time.sleep(5)
            
            # レポート生成
            if successful_experiments:
                print(f"\n📊 比較レポート生成...")
                try:
                    result = self.run_cli_command([
                        "python", str(self.scripts_dir / "advanced_experiment_manager.py"),
                        "generate-report", session_id,
                        *successful_experiments,
                        "--report-name", f"{workflow_name}_report"
                    ])
                    
                    if result.returncode == 0:
                        workflow_result["report_generated"] = True
                        print("✅ レポート生成完了")
                    else:
                        workflow_result["report_generated"] = False
                        print(f"⚠️ レポート生成失敗: {result.stderr}")
                        
                except Exception as e:
                    workflow_result["report_generated"] = False
                    print(f"⚠️ レポート生成エラー: {e}")
            
            # 最終ステータス決定
            if len(successful_experiments) == len(workflow_config["experiments"]):
                workflow_result["status"] = "all_success"
            elif len(successful_experiments) > 0:
                workflow_result["status"] = "partial_success"
            else:
                workflow_result["status"] = "all_failed"
            
            workflow_result["successful_experiments"] = successful_experiments
            workflow_result["failed_experiments"] = failed_experiments
            workflow_result["end_time"] = datetime.now().isoformat()
            
            # 結果サマリー
            print(f"\n🎉 ワークフロー完了: {workflow_name}")
            print(f"   セッションID: {session_id}")
            print(f"   成功実験: {len(successful_experiments)}/{len(workflow_config['experiments'])}")
            if successful_experiments:
                print(f"   成功: {', '.join(successful_experiments)}")
            if failed_experiments:
                print(f"   失敗: {', '.join(failed_experiments)}")
            
            # レポートファイル保存
            report_file = self.outputs_dir / session_id / "workflow_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_result, f, indent=2, ensure_ascii=False)
            
            print(f"   📄 ワークフローレポート: {report_file}")
            
            return workflow_result
            
        except Exception as e:
            workflow_result["status"] = "workflow_error"
            workflow_result["error"] = str(e)
            workflow_result["end_time"] = datetime.now().isoformat()
            print(f"❌ ワークフローエラー: {e}")
            return workflow_result


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="完全自動化実験ワークフロー")
    parser.add_argument("workflow", nargs="?", help="ワークフロー名")
    parser.add_argument("--session-id", help="セッションID (自動生成される場合は省略可)")
    parser.add_argument("--list-workflows", action="store_true", help="利用可能なワークフロー一覧")
    
    args = parser.parse_args()
    
    workflow_manager = AutomatedExperimentWorkflow()
    
    if args.list_workflows or not args.workflow:
        print("📋 利用可能なワークフロー:")
        for name, config in workflow_manager.workflow_configs.items():
            print(f"   🔸 {name}: {config['description']}")
            print(f"      実験数: {len(config['experiments'])}")
            for exp in config['experiments']:
                print(f"        - {exp['name']} ({exp['template']})")
            print()
        return
    
    try:
        result = workflow_manager.run_workflow(args.workflow, args.session_id)
        
        if result["status"] in ["all_success", "partial_success"]:
            print(f"\n🌐 結果を確認するには:")
            if "session_id" in result:
                reports_dir = workflow_manager.outputs_dir / "reports"
                print(f"   レポートディレクトリ: {reports_dir}")
                html_files = list(reports_dir.glob("**/report.html"))
                if html_files:
                    print(f"   HTMLレポート: open {html_files[-1]}")
        
        sys.exit(0 if result["status"] in ["all_success", "partial_success"] else 1)
        
    except Exception as e:
        print(f"❌ ワークフロー実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
