"""
データ管理・バックアップ・ロールバックユーティリティ

実験実行時のデータフォルダの安全な管理機能
- 実験前の自動バックアップ
- 実験後の自動ロールバック
- 実験データの分離管理
- 状態復元機能
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import logging
import tarfile
import tempfile
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DataStateManager:
    """データ状態管理クラス"""
    
    def __init__(self, workspace_dir: str = "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI"):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = self.workspace_dir / "data"
        self.backup_dir = self.workspace_dir / "data_backups"
        self.experiment_data_dir = self.workspace_dir / "experiment_data"
        
        # バックアップディレクトリ作成
        self.backup_dir.mkdir(exist_ok=True)
        self.experiment_data_dir.mkdir(exist_ok=True)
        
        # 状態管理ファイル
        self.state_file = self.backup_dir / "data_state_history.json"
        self.load_state_history()
        
        logger.info(f"DataStateManager initialized for {self.data_dir}")
    
    def load_state_history(self):
        """状態履歴の読み込み"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state_history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state history: {e}")
                self.state_history = {"backups": [], "current_state": None}
        else:
            self.state_history = {"backups": [], "current_state": None}
    
    def save_state_history(self):
        """状態履歴の保存"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state_history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Failed to save state history: {e}")
    
    def calculate_data_checksum(self, directory: Path) -> str:
        """ディレクトリのチェックサム計算"""
        hasher = hashlib.sha256()
        
        if not directory.exists():
            return ""
        
        # ファイルを再帰的に処理
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                    # ファイルパスも含める（構造の変化を検出）
                    hasher.update(str(file_path.relative_to(directory)).encode())
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return hasher.hexdigest()
    
    def create_backup(self, backup_name: str, description: str = "") -> str:
        """データフォルダのバックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{backup_name}_{timestamp}"
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        logger.info(f"Creating backup: {backup_id}")
        
        try:
            # データディレクトリの存在確認
            if not self.data_dir.exists():
                logger.warning(f"Data directory does not exist: {self.data_dir}")
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # チェックサム計算
            checksum_before = self.calculate_data_checksum(self.data_dir)
            
            # tar.gz形式でバックアップ作成
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(self.data_dir, arcname="data")
            
            # バックアップ情報記録
            backup_info = {
                "backup_id": backup_id,
                "backup_name": backup_name,
                "description": description,
                "timestamp": timestamp,
                "backup_path": str(backup_path),
                "checksum": checksum_before,
                "size_mb": backup_path.stat().st_size / (1024**2) if backup_path.exists() else 0,
                "file_count": sum(1 for _ in self.data_dir.rglob("*") if _.is_file())
            }
            
            self.state_history["backups"].append(backup_info)
            self.state_history["current_state"] = backup_id
            self.save_state_history()
            
            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Backup size: {backup_info['size_mb']:.2f} MB")
            logger.info(f"Files backed up: {backup_info['file_count']}")
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_id: str) -> bool:
        """バックアップからの復元"""
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup_path = Path(backup_info["backup_path"])
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        logger.info(f"Restoring backup: {backup_id}")
        
        try:
            # 現在のデータを一時退避（安全のため）
            temp_backup = self.create_temporary_backup()
            
            # データディレクトリ削除
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            
            # バックアップから復元
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(self.workspace_dir)
            
            # チェックサム検証
            restored_checksum = self.calculate_data_checksum(self.data_dir)
            if restored_checksum != backup_info["checksum"]:
                logger.warning("Checksum mismatch after restoration")
            
            # 状態更新
            self.state_history["current_state"] = backup_id
            self.save_state_history()
            
            # 一時バックアップ削除
            if temp_backup and Path(temp_backup).exists():
                Path(temp_backup).unlink()
            
            logger.info(f"Backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            
            # 復元失敗時は一時バックアップから復旧を試みる
            if temp_backup and Path(temp_backup).exists():
                try:
                    with tarfile.open(temp_backup, "r:gz") as tar:
                        tar.extractall(self.workspace_dir)
                    logger.info("Restored from temporary backup after failure")
                except Exception as restore_error:
                    logger.error(f"Failed to restore from temporary backup: {restore_error}")
            
            return False
    
    def create_temporary_backup(self) -> Optional[str]:
        """一時バックアップ作成"""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            with tarfile.open(temp_path, "w:gz") as tar:
                tar.add(self.data_dir, arcname="data")
            
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temporary backup: {e}")
            return None
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """バックアップ情報取得"""
        for backup in self.state_history["backups"]:
            if backup["backup_id"] == backup_id:
                return backup
        return None
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """バックアップ一覧取得"""
        return self.state_history["backups"]
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """古いバックアップの清理"""
        backups = sorted(self.state_history["backups"], key=lambda x: x["timestamp"], reverse=True)
        
        if len(backups) <= keep_count:
            return
        
        backups_to_remove = backups[keep_count:]
        
        for backup in backups_to_remove:
            backup_path = Path(backup["backup_path"])
            if backup_path.exists():
                try:
                    backup_path.unlink()
                    logger.info(f"Removed old backup: {backup['backup_id']}")
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup['backup_id']}: {e}")
        
        # 状態履歴更新
        self.state_history["backups"] = backups[:keep_count]
        self.save_state_history()
    
    def get_current_state(self) -> Dict[str, Any]:
        """現在のデータ状態取得"""
        return {
            "current_backup_id": self.state_history.get("current_state"),
            "data_directory": str(self.data_dir),
            "exists": self.data_dir.exists(),
            "current_checksum": self.calculate_data_checksum(self.data_dir),
            "file_count": sum(1 for _ in self.data_dir.rglob("*") if _.is_file()) if self.data_dir.exists() else 0,
            "size_mb": sum(f.stat().st_size for f in self.data_dir.rglob("*") if f.is_file()) / (1024**2) if self.data_dir.exists() else 0
        }


class ExperimentDataManager:
    """実験専用データ管理クラス"""
    
    def __init__(self, experiment_name: str, data_state_manager: DataStateManager):
        self.experiment_name = experiment_name
        self.state_manager = data_state_manager
        self.experiment_data_dir = data_state_manager.experiment_data_dir / experiment_name
        self.experiment_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験固有のデータパス
        self.experiment_data_paths = {
            "input": self.experiment_data_dir / "input",
            "processed": self.experiment_data_dir / "processed", 
            "embeddings": self.experiment_data_dir / "embeddings",
            "indices": self.experiment_data_dir / "indices",
            "cache": self.experiment_data_dir / "cache",
            "models": self.experiment_data_dir / "models",
            "temp": self.experiment_data_dir / "temp"
        }
        
        # ディレクトリ作成
        for path in self.experiment_data_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExperimentDataManager initialized for: {experiment_name}")
    
    def get_data_path(self, data_type: str) -> Path:
        """実験用データパス取得"""
        if data_type in self.experiment_data_paths:
            return self.experiment_data_paths[data_type]
        else:
            # カスタムパス
            custom_path = self.experiment_data_dir / data_type
            custom_path.mkdir(parents=True, exist_ok=True)
            return custom_path
    
    def copy_base_data(self, selective_copy: List[str] = None):
        """ベースデータの実験用コピー"""
        base_data_dir = self.state_manager.data_dir
        
        if not base_data_dir.exists():
            logger.warning(f"Base data directory does not exist: {base_data_dir}")
            return
        
        # 選択的コピー（指定がない場合は全てコピー）
        if selective_copy is None:
            selective_copy = ["processed", "raw", "models", "embedding"]
        
        for item_name in selective_copy:
            source_path = base_data_dir / item_name
            if source_path.exists():
                dest_path = self.experiment_data_dir / item_name
                
                if source_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                else:
                    shutil.copy2(source_path, dest_path)
                
                logger.info(f"Copied {item_name} to experiment data")
    
    def cleanup_experiment_data(self):
        """実験データの清理"""
        if self.experiment_data_dir.exists():
            shutil.rmtree(self.experiment_data_dir)
            logger.info(f"Cleaned up experiment data: {self.experiment_name}")


@contextmanager
def safe_experiment_environment(experiment_name: str, backup_description: str = "",
                               auto_rollback: bool = True, selective_copy: List[str] = None):
    """安全な実験環境コンテキストマネージャー"""
    
    # データ状態管理初期化
    state_manager = DataStateManager()
    experiment_data_manager = ExperimentDataManager(experiment_name, state_manager)
    
    # 実験前バックアップ作成
    backup_id = state_manager.create_backup(
        f"pre_{experiment_name}",
        backup_description or f"Pre-experiment backup for {experiment_name}"
    )
    
    logger.info(f"=== Starting safe experiment: {experiment_name} ===")
    logger.info(f"Backup created: {backup_id}")
    
    try:
        # ベースデータを実験用ディレクトリにコピー
        experiment_data_manager.copy_base_data(selective_copy)
        
        # 実験環境情報をyield
        yield {
            "experiment_name": experiment_name,
            "backup_id": backup_id,
            "state_manager": state_manager,
            "data_manager": experiment_data_manager,
            "experiment_data_dir": experiment_data_manager.experiment_data_dir,
            "data_paths": experiment_data_manager.experiment_data_paths
        }
        
        logger.info(f"=== Experiment completed successfully: {experiment_name} ===")
        
    except Exception as e:
        logger.error(f"=== Experiment failed: {experiment_name} ===")
        logger.error(f"Error: {e}")
        
        if auto_rollback:
            logger.info("Performing automatic rollback...")
            rollback_success = state_manager.restore_backup(backup_id)
            if rollback_success:
                logger.info("Rollback completed successfully")
            else:
                logger.error("Rollback failed - manual intervention required")
        
        raise
    
    finally:
        # 実験データ清理（実験が成功した場合）
        try:
            if auto_rollback:
                # 自動ロールバックが有効な場合は、メインデータを元に戻す
                logger.info("Performing post-experiment rollback...")
                state_manager.restore_backup(backup_id)
            
            # 実験用データは常に清理
            experiment_data_manager.cleanup_experiment_data()
            
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        
        # 古いバックアップの清理
        state_manager.cleanup_old_backups(keep_count=5)
        
        logger.info(f"=== Experiment environment cleaned up: {experiment_name} ===")


def create_experiment_data_config(experiment_env: Dict[str, Any]) -> Dict[str, str]:
    """実験用データ設定生成"""
    data_paths = experiment_env["data_paths"]
    
    return {
        "DATA_INPUT_DIR": str(data_paths["input"]),
        "DATA_PROCESSED_DIR": str(data_paths["processed"]),
        "DATA_EMBEDDINGS_DIR": str(data_paths["embeddings"]),
        "DATA_INDICES_DIR": str(data_paths["indices"]),
        "DATA_CACHE_DIR": str(data_paths["cache"]),
        "DATA_MODELS_DIR": str(data_paths["models"]),
        "DATA_TEMP_DIR": str(data_paths["temp"]),
        "EXPERIMENT_NAME": experiment_env["experiment_name"],
        "BACKUP_ID": experiment_env["backup_id"]
    }


# 使用例デコレーター
def with_data_safety(experiment_name: str, backup_description: str = "", 
                    auto_rollback: bool = True, selective_copy: List[str] = None):
    """データ安全性デコレーター"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with safe_experiment_environment(
                experiment_name, backup_description, auto_rollback, selective_copy
            ) as experiment_env:
                # 実験関数にデータパス情報を渡す
                kwargs["experiment_env"] = experiment_env
                return func(*args, **kwargs)
        return wrapper
    return decorator
