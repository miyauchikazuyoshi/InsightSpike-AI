#!/usr/bin/env python3
"""
データフォルダ初期状態復元スクリプト
===================================

実験後にdataフォルダをgit管理下の初期状態に戻します。
実験で生成された一時ファイルを削除し、リポジトリの正しい状態に復元します。
"""

import sys
import subprocess
import shutil
import json
from pathlib import Path
from typing import List, Set
from datetime import datetime

class DataInitializer:
    """データフォルダ初期化クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        
        # git管理下の正規ファイル一覧（実際のリポジトリ状態）
        self.git_tracked_files = {
            "data/cache/.gitkeep",
            "data/embedding/.gitkeep", 
            "data/graph_pyg.pt",
            "data/insight_facts.db",
            "data/integrated_rag_memory_experiments/integrated_rag_memory_results_20250613_005205.json",
            "data/integrated_rag_memory_experiments/visualizations/integrated_rag_memory_analysis.png",
            "data/logs/.gitkeep",
            "data/models/.gitkeep",
            "data/processed/.gitkeep",
            "data/processed/comprehensive_rag_analysis.json",
            "data/processed/experiment_results.json", 
            "data/processed/graph_visualization_results.json",
            "data/processed/simple_metadata.json",
            "data/processed/test_questions.json",
            "data/raw/.gitkeep",
            "data/raw/indirect_knowledge.txt",
            "data/raw/insight_dataset.txt", 
            "data/raw/simple_dataset.txt",
            "data/raw/test_sentences.txt",
            "data/samples/benchmark_data.json",
            "data/unknown_learning.db"
        }
        
        # 実験で生成される一時ファイルパターン
        self.temp_file_patterns = [
            "*_experiment_*",
            "*_temp_*", 
            "*_backup_*",
            "index*.faiss",
            "episodes*.json",
            "*_seed*.json",
            "*.tmp",
            "*.temp"
        ]
        
    def get_current_git_status(self) -> dict:
        """現在のgit状態を取得"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "data/"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            status = {"deleted": [], "modified": [], "untracked": []}
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                state = line[:2]
                file_path = line[3:]
                
                if state == " D" or state == "D ":
                    status["deleted"].append(file_path)
                elif state == " M" or state == "M ":
                    status["modified"].append(file_path)
                elif state == "??":
                    status["untracked"].append(file_path)
                    
            return status
            
        except Exception as e:
            print(f"⚠️ git状態取得エラー: {e}")
            return {"deleted": [], "modified": [], "untracked": []}
    
    def restore_deleted_files(self, deleted_files: List[str]) -> None:
        """削除されたgit管理ファイルを復元"""
        if not deleted_files:
            print("📁 削除されたファイルはありません")
            return
            
        print(f"🔄 {len(deleted_files)}個の削除ファイルを復元中...")
        
        for file_path in deleted_files:
            try:
                # git restoreコマンドを使用
                result = subprocess.run(
                    ["git", "restore", file_path],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    print(f"   ✅ 復元: {file_path}")
                else:
                    # git checkoutで試す
                    result2 = subprocess.run(
                        ["git", "checkout", "HEAD", "--", file_path],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root
                    )
                    
                    if result2.returncode == 0:
                        print(f"   ✅ 復元 (checkout): {file_path}")
                    else:
                        print(f"   ❌ 復元失敗: {file_path} - restore: {result.stderr.strip()}, checkout: {result2.stderr.strip()}")
                    
            except Exception as e:
                print(f"   ❌ 復元エラー: {file_path} - {e}")
    
    def clean_temp_files(self) -> None:
        """実験一時ファイルを削除"""
        print("🧹 実験一時ファイル削除中...")
        
        deleted_count = 0
        
        for pattern in self.temp_file_patterns:
            for file_path in self.data_dir.rglob(pattern):
                if file_path.is_file():
                    try:
                        # git管理下でないことを確認
                        relative_path = str(file_path.relative_to(self.project_root))
                        if relative_path not in self.git_tracked_files:
                            file_path.unlink()
                            print(f"   🗑️ 削除: {relative_path}")
                            deleted_count += 1
                        else:
                            print(f"   ⚠️ スキップ（git管理下）: {relative_path}")
                            
                    except Exception as e:
                        print(f"   ❌ 削除エラー: {file_path} - {e}")
        
        print(f"✅ {deleted_count}個の一時ファイルを削除しました")
    
    def remove_untracked_files(self, untracked_files: List[str]) -> None:
        """未追跡ファイル（実験生成物）を削除"""
        if not untracked_files:
            print("📁 未追跡ファイルはありません")
            return
            
        print(f"🗑️ {len(untracked_files)}個の未追跡ファイルを削除中...")
        
        for file_path in untracked_files:
            try:
                full_path = self.project_root / file_path
                if full_path.exists():
                    if full_path.is_file():
                        full_path.unlink()
                        print(f"   🗑️ ファイル削除: {file_path}")
                    elif full_path.is_dir() and not any(full_path.iterdir()):
                        full_path.rmdir()
                        print(f"   🗑️ 空ディレクトリ削除: {file_path}")
                        
            except Exception as e:
                print(f"   ❌ 削除エラー: {file_path} - {e}")
    
    def verify_initial_state(self) -> bool:
        """初期状態復元の検証"""
        print("🔍 初期状態復元を検証中...")
        
        # git status再チェック
        status = self.get_current_git_status()
        
        if not status["deleted"] and not status["untracked"]:
            print("✅ データフォルダが正しい初期状態に復元されました")
            return True
        else:
            if status["deleted"]:
                print(f"⚠️ まだ削除されているファイル: {len(status['deleted'])}個")
            if status["untracked"]:
                print(f"⚠️ まだ未追跡ファイル: {len(status['untracked'])}個")
            return False
    
    def create_initialization_report(self) -> dict:
        """初期化レポート作成"""
        status = self.get_current_git_status()
        
        report = {
            "initialization_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "data_directory": str(self.data_dir),
            "git_status": status,
            "git_tracked_files_count": len(self.git_tracked_files),
            "temp_patterns_cleaned": self.temp_file_patterns,
            "is_clean_state": not status["deleted"] and not status["untracked"]
        }
        
        return report
    
    def reset_to_initial_state(self) -> dict:
        """データフォルダを初期状態にリセット"""
        print("🔄 データフォルダ初期状態復元開始")
        print("=" * 60)
        
        # 1. 現在の状態確認
        print("1️⃣ 現在のgit状態確認...")
        initial_status = self.get_current_git_status()
        print(f"   削除ファイル: {len(initial_status['deleted'])}個")
        print(f"   変更ファイル: {len(initial_status['modified'])}個") 
        print(f"   未追跡ファイル: {len(initial_status['untracked'])}個")
        
        # 2. 削除されたgit管理ファイルを復元
        print("\n2️⃣ 削除ファイル復元...")
        self.restore_deleted_files(initial_status["deleted"])
        
        # 3. 実験一時ファイルを削除
        print("\n3️⃣ 一時ファイル削除...")
        self.clean_temp_files()
        
        # 4. 未追跡ファイルを削除
        print("\n4️⃣ 未追跡ファイル削除...")
        self.remove_untracked_files(initial_status["untracked"])
        
        # 5. 初期状態検証
        print("\n5️⃣ 初期状態検証...")
        is_clean = self.verify_initial_state()
        
        # 6. レポート作成
        report = self.create_initialization_report()
        report["initialization_success"] = is_clean
        
        print(f"\n{'✅' if is_clean else '⚠️'} データフォルダ初期化{'完了' if is_clean else '部分完了'}")
        
        return report


def main():
    """メイン実行関数"""
    print("🔄 データフォルダ初期状態復元スクリプト")
    print("=" * 50)
    
    try:
        initializer = DataInitializer()
        report = initializer.reset_to_initial_state()
        
        # レポート保存
        report_path = Path("data_initialization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 初期化レポート保存: {report_path}")
        
        if report["initialization_success"]:
            print("\n🎉 データフォルダが正しい初期状態に復元されました！")
            return 0
        else:
            print("\n⚠️ 一部問題があります。手動での確認が必要かもしれません。")
            return 1
            
    except Exception as e:
        print(f"\n❌ 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
