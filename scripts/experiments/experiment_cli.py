#!/usr/bin/env python3
"""
InsightSpike-AI 実験管理CLI
==============================

同一条件での対照実験を可能にする包括的な実験管理ツール
"""

import sys
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3

# InsightSpike-AIコンポーネントを読み込み
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from insightspike.core.config import get_config
    from insightspike.utils.embedder import get_model
    from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
    from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
except ImportError as e:
    print(f"❌ InsightSpike-AIコンポーネント読み込みエラー: {e}")
    sys.exit(1)


class ExperimentCLI:
    """実験管理CLIクラス"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.outputs_dir = Path("experiments/outputs")
        
        # データディレクトリ構造
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.embedding_dir = self.data_dir / "embedding"
        self.logs_dir = self.data_dir / "logs"
        self.cache_dir = self.data_dir / "cache"
        
        # 必要なディレクトリを作成
        for dir_path in [self.raw_dir, self.processed_dir, self.embedding_dir, 
                        self.logs_dir, self.cache_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_session(self, session_name: str, description: str = "") -> str:
        """実験セッション作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{session_name}_{timestamp}"
        
        # セッションディレクトリ作成
        session_dir = self.outputs_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # セッション情報保存
        session_info = {
            "session_id": session_id,
            "session_name": session_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "experiments": []
        }
        
        with open(session_dir / "session_info.json", 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 実験セッション作成: {session_id}")
        print(f"📁 セッションディレクトリ: {session_dir}")
        
        return session_id
    
    def backup_data_state(self, backup_name: str) -> Path:
        """データ状態をバックアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{backup_name}_{timestamp}"
        backup_path = self.data_dir / "cache" / "backups" / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 重要ファイルをバックアップ
        important_files = [
            "index.faiss",
            "graph_pyg.pt", 
            "insight_facts.db",
            "unknown_learning.db"
        ]
        
        backed_up_files = []
        for file_name in important_files:
            src = self.data_dir / file_name
            if src.exists():
                dst = backup_path / file_name
                shutil.copy2(src, dst)
                backed_up_files.append(file_name)
                print(f"📦 バックアップ: {file_name}")
        
        # バックアップ情報を保存
        backup_info = {
            "backup_id": backup_id,
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "backed_up_files": backed_up_files,
            "backup_path": str(backup_path)
        }
        
        with open(backup_path / "backup_info.json", 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ データバックアップ完了: {backup_id}")
        return backup_path
    
    def restore_data_state(self, backup_id: str) -> bool:
        """データ状態を復元"""
        backup_path = self.data_dir / "cache" / "backups" / backup_id
        
        if not backup_path.exists():
            print(f"❌ バックアップが見つかりません: {backup_id}")
            return False
        
        # バックアップ情報を読み込み
        backup_info_path = backup_path / "backup_info.json"
        if not backup_info_path.exists():
            print(f"❌ バックアップ情報が見つかりません: {backup_info_path}")
            return False
        
        with open(backup_info_path, 'r', encoding='utf-8') as f:
            backup_info = json.load(f)
        
        # ファイルを復元
        restored_files = []
        for file_name in backup_info["backed_up_files"]:
            src = backup_path / file_name
            dst = self.data_dir / file_name
            if src.exists():
                shutil.copy2(src, dst)
                restored_files.append(file_name)
                print(f"🔄 復元: {file_name}")
        
        print(f"✅ データ復元完了: {backup_id}")
        print(f"   復元ファイル数: {len(restored_files)}")
        return True
    
    def clean_data_folder(self, keep_structure: bool = True, preserve_graph: bool = True, preserve_index: bool = True) -> None:
        """dataフォルダをクリーンアップ"""
        print("🧹 dataフォルダクリーンアップ開始...")
        
        # 削除対象ファイル（条件付き削除）
        cleanup_files = [
            "insight_facts.db", 
            "unknown_learning.db"
        ]
        
        # index.faissの処理
        if not preserve_index:
            cleanup_files.append("index.faiss")
        else:
            print("🔒 index.faiss を保持します")
        
        # graph_pyg.ptの処理を分離
        if not preserve_graph:
            cleanup_files.append("graph_pyg.pt")
        else:
            print("🔒 graph_pyg.pt を保持します")
        
        # 削除対象ディレクトリの中身
        cleanup_dirs = ["cache", "processed"]
        
        removed_files = []
        
        # ファイル削除
        for file_name in cleanup_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                file_path.unlink()
                removed_files.append(file_name)
                print(f"🗑️ 削除: {file_name}")
        
        # ディレクトリの中身削除
        for dir_name in cleanup_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        removed_files.append(f"{dir_name}/{item.name}")
                    elif item.is_dir() and item.name != "backups":  # backupsディレクトリは保持
                        shutil.rmtree(item)
                        removed_files.append(f"{dir_name}/{item.name}/")
                print(f"🗑️ クリア: {dir_name}/")
        
        print(f"✅ クリーンアップ完了: {len(removed_files)}項目削除")
        
        if keep_structure:
            # ディレクトリ構造を再作成
            for dir_path in [self.raw_dir, self.processed_dir, self.embedding_dir, 
                            self.logs_dir, self.cache_dir]:
                dir_path.mkdir(exist_ok=True)
            print("📁 ディレクトリ構造を再作成")
    
    def clean_temp_files(self) -> None:
        """実験用一時ファイルのみをクリーンアップ（重要ファイルは保持）"""
        print("🧹 実験用一時ファイルクリーンアップ開始...")
        
        # 実験中に作成される一時ファイルのみ削除
        temp_files = [
            "insight_facts.db", 
            "unknown_learning.db"
        ]
        
        # 一時ディレクトリ（キャッシュのみ）
        temp_dirs = ["cache"]
        
        removed_files = []
        
        # 一時ファイル削除
        for file_name in temp_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                file_path.unlink()
                removed_files.append(file_name)
                print(f"🗑️ 削除: {file_name}")
        
        # 一時ディレクトリの中身削除
        for dir_name in temp_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        removed_files.append(f"{dir_name}/{item.name}")
                    elif item.is_dir() and item.name != "backups":
                        shutil.rmtree(item)
                        removed_files.append(f"{dir_name}/{item.name}/")
                print(f"🗑️ クリア: {dir_name}/")
        
        print(f"✅ 一時ファイルクリーンアップ完了: {len(removed_files)}項目削除")
        print("🔒 重要ファイル保持: graph_pyg.pt, index.faiss, index.json, episodes.json, processed/, raw/, samples/, embedding/")
    
    def build_initial_memory(self, episodes_count: int = 50, seed: int = 42) -> Dict[str, Any]:
        """初期メモリ/グラフを構築"""
        print(f"🧠 初期メモリ構築開始 ({episodes_count}エピソード, seed={seed})")
        
        import random
        import numpy as np
        
        # シード設定
        random.seed(seed)
        np.random.seed(seed)
        
        # コンポーネント初期化
        config = get_config()
        model = get_model()
        memory_manager = L2MemoryManager(dim=384)
        knowledge_graph = KnowledgeGraphMemory(embedding_dim=384, similarity_threshold=0.3)
        
        # 基礎エピソード生成（再現可能）
        research_areas = [
            "Large Language Models", "Computer Vision", "Reinforcement Learning",
            "Graph Neural Networks", "Federated Learning", "Explainable AI",
            "Multimodal Learning", "Few-shot Learning", "Transfer Learning",
            "Adversarial Machine Learning"
        ]
        
        activity_types = [
            "achieves breakthrough performance on",
            "introduces novel architecture for", 
            "demonstrates significant improvements in",
            "reveals new insights about",
            "establishes new benchmarks for"
        ]
        
        domains = [
            "medical diagnosis", "autonomous systems", "natural language processing",
            "computer vision", "robotics", "cybersecurity", "climate modeling",
            "drug discovery", "financial prediction", "educational technology"
        ]
        
        episodes = []
        start_time = time.time()
        
        for i in range(1, episodes_count + 1):
            research_area = research_areas[(i - 1) % len(research_areas)]
            activity_type = activity_types[(i - 1) % len(activity_types)]
            domain = domains[(i - 1) % len(domains)]
            
            text = f"Initial research in {research_area} {activity_type} {domain}, establishing foundational knowledge for future insights."
            
            # メモリに保存
            memory_manager.store_episode(
                text=text,
                c_value=0.5,  # 初期エピソードは中程度の重要度
                metadata={
                    'id': i,
                    'type': 'initial',
                    'domain': domain,
                    'research_area': research_area,
                    'seed': seed
                }
            )
            
            episodes.append({
                'id': i,
                'text': text,
                'research_area': research_area,
                'activity_type': activity_type,
                'domain': domain,
                'type': 'initial',
                'timestamp': datetime.now().isoformat()
            })
            
            if i % 10 == 0:
                print(f"📝 {i}/{episodes_count}エピソード構築完了")
        
        build_time = time.time() - start_time
        
        # 初期状態情報を保存
        initial_state_info = {
            "episodes_count": episodes_count,
            "seed": seed,
            "build_time_seconds": build_time,
            "timestamp": datetime.now().isoformat(),
            "episodes": episodes
        }
        
        # rawディレクトリに保存
        with open(self.raw_dir / f"initial_episodes_seed{seed}_count{episodes_count}.json", 'w', encoding='utf-8') as f:
            json.dump(initial_state_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 初期メモリ構築完了!")
        print(f"   エピソード数: {episodes_count}")
        print(f"   構築時間: {build_time:.2f}秒")
        print(f"   シード値: {seed}")
        print(f"   データ保存: data/raw/initial_episodes_seed{seed}_count{episodes_count}.json")
        
        return initial_state_info
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """利用可能なバックアップ一覧"""
        backups_dir = self.data_dir / "cache" / "backups"
        if not backups_dir.exists():
            return []
        
        backups = []
        for backup_dir in backups_dir.iterdir():
            if backup_dir.is_dir():
                info_file = backup_dir / "backup_info.json"
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        backup_info = json.load(f)
                    backups.append(backup_info)
        
        # 作成日時で降順ソート
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups
    
    def show_data_status(self) -> Dict[str, Any]:
        """現在のデータ状態を表示"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "data_files": {},
            "directories": {},
            "memory_info": {},
            "experiments_info": {}
        }
        
        # データファイルの状態
        data_files = ["index.faiss", "graph_pyg.pt", "insight_facts.db", "unknown_learning.db"]
        for file_name in data_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                stat = file_path.stat()
                status["data_files"][file_name] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                status["data_files"][file_name] = {"exists": False}
        
        # ディレクトリの状態
        for dir_name in ["raw", "processed", "embedding", "logs", "cache"]:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                file_count = len([f for f in files if f.is_file()])
                dir_count = len([f for f in files if f.is_dir()])
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                status["directories"][dir_name] = {
                    "exists": True,
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size_mb": total_size / (1024 * 1024)
                }
            else:
                status["directories"][dir_name] = {"exists": False}
        
        # メモリ情報（DBから取得）
        try:
            if (self.data_dir / "insight_facts.db").exists():
                conn = sqlite3.connect(self.data_dir / "insight_facts.db")
                cursor = conn.cursor()
                
                # エピソード数
                cursor.execute("SELECT COUNT(*) FROM episodes")
                episode_count = cursor.fetchone()[0]
                
                # テーブル情報
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                status["memory_info"] = {
                    "episodes_count": episode_count,
                    "database_tables": tables
                }
                conn.close()
        except Exception as e:
            status["memory_info"]["error"] = str(e)
        
        # 最近の実験情報
        try:
            if self.outputs_dir.exists():
                recent_experiments = []
                for exp_dir in self.outputs_dir.iterdir():
                    if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                        session_info_file = exp_dir / "session_info.json"
                        if session_info_file.exists():
                            with open(session_info_file, 'r', encoding='utf-8') as f:
                                session_info = json.load(f)
                            recent_experiments.append({
                                "session_id": exp_dir.name,
                                "created_at": session_info.get("created_at", "unknown"),
                                "experiment_count": len(list(exp_dir.glob("*/06_experiment_results.json")))
                            })
                
                # 作成日時で降順ソート
                recent_experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                status["experiments_info"]["recent_sessions"] = recent_experiments[:5]  # 最新5件
                
        except Exception as e:
            status["experiments_info"]["error"] = str(e)
        
        return status
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """データ整合性チェック"""
        print("🔍 データ整合性チェック開始...")
        
        integrity_report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "overall_status": "unknown"
        }
        
        # 1. 必須ファイルの存在チェック
        required_files = {
            "index.faiss": "FAISS インデックス",
            "insight_facts.db": "洞察データベース"
        }
        
        for file_name, description in required_files.items():
            file_path = self.data_dir / file_name
            if file_path.exists():
                integrity_report["checks"][file_name] = {"status": "OK", "description": description}
            else:
                integrity_report["errors"].append(f"必須ファイルが見つかりません: {file_name} ({description})")
                integrity_report["checks"][file_name] = {"status": "MISSING", "description": description}
        
        # 2. データベース整合性チェック
        try:
            if (self.data_dir / "insight_facts.db").exists():
                conn = sqlite3.connect(self.data_dir / "insight_facts.db")
                cursor = conn.cursor()
                
                # テーブル存在チェック
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                if "episodes" in tables:
                    cursor.execute("SELECT COUNT(*) FROM episodes")
                    episode_count = cursor.fetchone()[0]
                    integrity_report["checks"]["database_episodes"] = {
                        "status": "OK", 
                        "count": episode_count,
                        "description": f"エピソードテーブル ({episode_count}件)"
                    }
                else:
                    integrity_report["errors"].append("エピソードテーブルが見つかりません")
                
                conn.close()
        except Exception as e:
            integrity_report["errors"].append(f"データベースチェックエラー: {e}")
        
        # 3. ディレクトリ構造チェック
        required_dirs = ["raw", "processed", "embedding", "logs", "cache"]
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                integrity_report["checks"][f"dir_{dir_name}"] = {"status": "OK", "description": f"{dir_name}ディレクトリ"}
            else:
                integrity_report["warnings"].append(f"推奨ディレクトリが見つかりません: {dir_name}")
                integrity_report["checks"][f"dir_{dir_name}"] = {"status": "MISSING", "description": f"{dir_name}ディレクトリ"}
        
        # 4. ファイルサイズ異常チェック
        for file_name in ["index.faiss", "graph_pyg.pt", "insight_facts.db"]:
            file_path = self.data_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < 0.001:  # 1KB未満
                    integrity_report["warnings"].append(f"ファイルサイズが小さすぎます: {file_name} ({size_mb:.3f}MB)")
                elif size_mb > 1000:  # 1GB超
                    integrity_report["warnings"].append(f"ファイルサイズが大きすぎます: {file_name} ({size_mb:.1f}MB)")
        
        # 総合ステータス決定
        if integrity_report["errors"]:
            integrity_report["overall_status"] = "ERROR"
        elif integrity_report["warnings"]:
            integrity_report["overall_status"] = "WARNING"  
        else:
            integrity_report["overall_status"] = "OK"
        
        # 結果表示
        if integrity_report["overall_status"] == "OK":
            print("✅ データ整合性チェック完了: 問題なし")
        elif integrity_report["overall_status"] == "WARNING":
            print("⚠️ データ整合性チェック完了: 警告あり")
            for warning in integrity_report["warnings"]:
                print(f"   ⚠️ {warning}")
        else:
            print("❌ データ整合性チェック完了: エラーあり")
            for error in integrity_report["errors"]:
                print(f"   ❌ {error}")
        
        return integrity_report


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="InsightSpike-AI 実験管理CLI")
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # セッション作成
    session_parser = subparsers.add_parser("create-session", help="実験セッション作成")
    session_parser.add_argument("name", help="セッション名")
    session_parser.add_argument("--description", default="", help="セッション説明")
    
    # データバックアップ
    backup_parser = subparsers.add_parser("backup", help="データ状態バックアップ")
    backup_parser.add_argument("name", help="バックアップ名")
    
    # データ復元
    restore_parser = subparsers.add_parser("restore", help="データ状態復元")
    restore_parser.add_argument("backup_id", help="バックアップID")
    
    # データクリーンアップ
    clean_parser = subparsers.add_parser("clean", help="dataフォルダクリーンアップ")
    clean_parser.add_argument("--no-structure", action="store_true", help="ディレクトリ構造も削除")
    clean_parser.add_argument("--delete-graph", action="store_true", help="graph_pyg.ptも削除する（デフォルトは保持）")
    clean_parser.add_argument("--delete-index", action="store_true", help="index.faissも削除する（デフォルトは保持）")
    
    # 一時ファイルクリーンアップ
    temp_clean_parser = subparsers.add_parser("clean-temp", help="実験用一時ファイルのみクリーンアップ（graph_pyg.ptは保持）")
    
    # 初期メモリ構築
    memory_parser = subparsers.add_parser("build-memory", help="初期メモリ構築")
    memory_parser.add_argument("--episodes", type=int, default=50, help="エピソード数 (default: 50)")
    memory_parser.add_argument("--seed", type=int, default=42, help="ランダムシード (default: 42)")
    
    # バックアップ一覧
    subparsers.add_parser("list-backups", help="バックアップ一覧表示")
    
    # データ状態表示
    subparsers.add_parser("status", help="現在のデータ状態表示")
    
    # データ整合性チェック
    subparsers.add_parser("check", help="データ整合性チェック")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ExperimentCLI()
    
    try:
        if args.command == "create-session":
            cli.create_experiment_session(args.name, args.description)
        
        elif args.command == "backup":
            cli.backup_data_state(args.name)
        
        elif args.command == "restore":
            cli.restore_data_state(args.backup_id)
        
        elif args.command == "clean":
            cli.clean_data_folder(
                keep_structure=not args.no_structure,
                preserve_graph=not args.delete_graph,
                preserve_index=not args.delete_index
            )
        
        elif args.command == "clean-temp":
            cli.clean_temp_files()
        
        elif args.command == "build-memory":
            cli.build_initial_memory(args.episodes, args.seed)
        
        elif args.command == "list-backups":
            backups = cli.list_backups()
            if not backups:
                print("📦 利用可能なバックアップはありません")
            else:
                print(f"📦 利用可能なバックアップ ({len(backups)}個):")
                for backup in backups:
                    print(f"   🔸 {backup['backup_id']}")
                    print(f"      名前: {backup['backup_name']}")
                    print(f"      作成: {backup['timestamp']}")
                    print(f"      ファイル: {len(backup['backed_up_files'])}個")
                    print()
        
        elif args.command == "status":
            status = cli.show_data_status()
            print("📊 現在のデータ状態:")
            print(f"   確認時刻: {status['timestamp']}")
            print()
            
            print("📄 データファイル:")
            for file_name, info in status["data_files"].items():
                if info["exists"]:
                    print(f"   ✅ {file_name}: {info['size_mb']:.2f}MB (更新: {info['modified']})")
                else:
                    print(f"   ❌ {file_name}: 存在せず")
            print()
            
            print("📁 ディレクトリ:")
            for dir_name, info in status["directories"].items():
                if info["exists"]:
                    print(f"   ✅ {dir_name}/: {info['file_count']}ファイル, {info['dir_count']}ディレクトリ ({info['total_size_mb']:.2f}MB)")
                else:
                    print(f"   ❌ {dir_name}/: 存在せず")
            print()
            
            if "episodes_count" in status["memory_info"]:
                print(f"🧠 メモリ情報:")
                print(f"   エピソード数: {status['memory_info']['episodes_count']}")
                print(f"   DBテーブル: {', '.join(status['memory_info']['database_tables'])}")
            
            if "recent_sessions" in status.get("experiments_info", {}):
                print(f"\n🔬 最近の実験セッション:")
                for session in status["experiments_info"]["recent_sessions"]:
                    print(f"   📋 {session['session_id']}: {session['experiment_count']}実験 ({session['created_at']})")
        
        elif args.command == "check":
            cli.validate_data_integrity()
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
