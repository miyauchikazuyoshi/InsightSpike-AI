"""
実験環境構築・監視ユーティリティ

実験実行環境の準備、リソース監視、依存関係管理機能
"""

import os
import sys
import psutil
import time
import logging
import subprocess
import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import threading
import queue
import resource
import platform

logger = logging.getLogger(__name__)


class ExperimentEnvironment:
    """実験環境構築・管理クラス"""
    
    def __init__(self, experiment_name: str, workspace_dir: str = "./experiment_workspace"):
        self.experiment_name = experiment_name
        self.workspace_dir = Path(workspace_dir) / experiment_name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.dependencies = []
        self.environment_vars = {}
        
        # 環境情報収集
        self.system_info = self._collect_system_info()
        
        # ロギング設定
        self._setup_logging()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """システム情報収集"""
        try:
            return {
                "platform": platform.platform(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "hostname": platform.node(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
            return {"error": str(e)}
    
    def _setup_logging(self):
        """実験用ロギング設定"""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # ファイルハンドラー設定
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # フォーマッター設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # ルートロガーに追加
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)
        
        logger.info(f"Logging setup complete for experiment: {self.experiment_name}")
    
    def set_config(self, config: Dict[str, Any]):
        """実験設定の指定"""
        self.config = config
        
        # 設定ファイル保存
        config_file = self.workspace_dir / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment config set: {list(config.keys())}")
    
    def add_dependencies(self, dependencies: List[str]):
        """依存関係追加"""
        self.dependencies.extend(dependencies)
        logger.info(f"Dependencies added: {dependencies}")
    
    def set_environment_variables(self, env_vars: Dict[str, str]):
        """環境変数設定"""
        self.environment_vars.update(env_vars)
        for key, value in env_vars.items():
            os.environ[key] = value
        logger.info(f"Environment variables set: {list(env_vars.keys())}")
    
    def install_dependencies(self, force_reinstall: bool = False) -> bool:
        """依存関係インストール"""
        if not self.dependencies:
            logger.info("No dependencies to install")
            return True
        
        requirements_file = self.workspace_dir / "requirements.txt"
        
        # requirements.txt作成
        with open(requirements_file, 'w') as f:
            for dep in self.dependencies:
                f.write(f"{dep}\n")
        
        try:
            # pip install実行
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            if force_reinstall:
                cmd.append("--force-reinstall")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Dependencies installed successfully")
            logger.debug(f"Install output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def prepare_data_directories(self, directories: List[str]):
        """データディレクトリ準備"""
        for dir_name in directories:
            dir_path = self.workspace_dir / "data" / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory created: {dir_path}")
    
    def create_output_directories(self, output_types: List[str] = None):
        """出力ディレクトリ作成"""
        if output_types is None:
            output_types = ["results", "visualizations", "reports", "models", "logs"]
        
        for output_type in output_types:
            output_dir = self.workspace_dir / output_type
            output_dir.mkdir(exist_ok=True)
            logger.info(f"Output directory created: {output_dir}")
    
    def validate_environment(self) -> Dict[str, Any]:
        """実験環境検証"""
        validation_results = {
            "system_requirements": True,
            "dependencies": True,
            "disk_space": True,
            "memory": True,
            "errors": []
        }
        
        # 最小システム要件チェック
        min_memory_gb = self.config.get("min_memory_gb", 4)
        min_disk_gb = self.config.get("min_disk_gb", 10)
        
        # メモリチェック
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < min_memory_gb:
            validation_results["memory"] = False
            validation_results["errors"].append(
                f"Insufficient memory: {available_memory_gb:.1f}GB available, {min_memory_gb}GB required"
            )
        
        # ディスク容量チェック
        available_disk_gb = psutil.disk_usage(str(self.workspace_dir)).free / (1024**3)
        if available_disk_gb < min_disk_gb:
            validation_results["disk_space"] = False
            validation_results["errors"].append(
                f"Insufficient disk space: {available_disk_gb:.1f}GB available, {min_disk_gb}GB required"
            )
        
        # 依存関係チェック
        try:
            for dep in self.dependencies:
                __import__(dep.split('==')[0].split('>=')[0].split('<=')[0])
        except ImportError as e:
            validation_results["dependencies"] = False
            validation_results["errors"].append(f"Missing dependency: {e}")
        
        # 全体評価
        validation_results["overall"] = all([
            validation_results["system_requirements"],
            validation_results["dependencies"],
            validation_results["disk_space"],
            validation_results["memory"]
        ])
        
        logger.info(f"Environment validation completed: {'PASSED' if validation_results['overall'] else 'FAILED'}")
        return validation_results
    
    def save_environment_snapshot(self) -> str:
        """実験環境スナップショット保存"""
        snapshot = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "config": self.config,
            "dependencies": self.dependencies,
            "environment_vars": dict(os.environ),
            "python_path": sys.path,
            "working_directory": str(Path.cwd()),
            "workspace_directory": str(self.workspace_dir)
        }
        
        snapshot_file = self.workspace_dir / "environment_snapshot.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Environment snapshot saved: {snapshot_file}")
        return str(snapshot_file)


class ResourceMonitor:
    """リソース監視クラス"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        self.resource_data = []
        self.start_time = None
    
    def start_monitoring(self):
        """リソース監視開始"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """リソース監視停止・結果取得"""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return {}
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        # キューからデータ収集
        while not self.data_queue.empty():
            self.resource_data.append(self.data_queue.get())
        
        # 統計計算
        summary = self._calculate_resource_summary()
        
        logger.info("Resource monitoring stopped")
        return summary
    
    def _monitor_loop(self):
        """監視ループ"""
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                timestamp = time.time() - self.start_time
                
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # メモリ使用量
                memory_info = psutil.virtual_memory()
                process_memory = process.memory_info()
                
                # ディスクI/O
                disk_io = psutil.disk_io_counters()
                
                # ネットワークI/O
                network_io = psutil.net_io_counters()
                
                # GPUメモリ（利用可能な場合）
                gpu_memory = self._get_gpu_memory()
                
                resource_snapshot = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available": memory_info.available,
                    "process_memory_rss": process_memory.rss,
                    "process_memory_vms": process_memory.vms,
                    "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                    "network_sent": network_io.bytes_sent if network_io else 0,
                    "network_recv": network_io.bytes_recv if network_io else 0,
                    "gpu_memory": gpu_memory
                }
                
                self.data_queue.put(resource_snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
    
    def _get_gpu_memory(self) -> Optional[Dict[str, float]]:
        """GPU メモリ使用量取得（可能な場合）"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info = {}
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info[f"gpu_{i}"] = {
                    "total": memory_info.total,
                    "used": memory_info.used,
                    "free": memory_info.free
                }
            
            return gpu_info
            
        except ImportError:
            # pynvml not available
            return None
        except Exception as e:
            logger.debug(f"GPU monitoring failed: {e}")
            return None
    
    def _calculate_resource_summary(self) -> Dict[str, Any]:
        """リソース使用量サマリー計算"""
        if not self.resource_data:
            return {"error": "No monitoring data collected"}
        
        # データフレーム形式での分析
        import pandas as pd
        df = pd.DataFrame(self.resource_data)
        
        summary = {
            "monitoring_duration": df['timestamp'].max() if len(df) > 0 else 0,
            "data_points": len(df),
            "cpu_usage": {
                "mean": df['cpu_percent'].mean(),
                "max": df['cpu_percent'].max(),
                "min": df['cpu_percent'].min(),
                "std": df['cpu_percent'].std()
            },
            "memory_usage": {
                "mean_percent": df['memory_percent'].mean(),
                "max_percent": df['memory_percent'].max(),
                "mean_process_mb": df['process_memory_rss'].mean() / (1024**2),
                "max_process_mb": df['process_memory_rss'].max() / (1024**2)
            },
            "disk_io": {
                "total_read_mb": (df['disk_read_bytes'].max() - df['disk_read_bytes'].min()) / (1024**2),
                "total_write_mb": (df['disk_write_bytes'].max() - df['disk_write_bytes'].min()) / (1024**2)
            },
            "network_io": {
                "total_sent_mb": (df['network_sent'].max() - df['network_sent'].min()) / (1024**2),
                "total_recv_mb": (df['network_recv'].max() - df['network_recv'].min()) / (1024**2)
            }
        }
        
        # GPU情報（利用可能な場合）
        if 'gpu_memory' in df.columns and df['gpu_memory'].notna().any():
            summary["gpu_usage"] = "GPU monitoring data available"
        
        return summary
    
    def save_monitoring_data(self, output_file: str):
        """監視データ保存"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        monitoring_data = {
            "metadata": {
                "monitoring_interval": self.monitoring_interval,
                "start_time": self.start_time,
                "data_points": len(self.resource_data)
            },
            "data": self.resource_data,
            "summary": self._calculate_resource_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Monitoring data saved: {output_path}")


def measure_execution_time(func: Callable) -> Callable:
    """実行時間測定デコレーター"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        # 結果に実行時間情報を追加
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    
    return wrapper


def setup_experiment_environment(experiment_name: str, config: Dict[str, Any],
                                dependencies: List[str] = None) -> ExperimentEnvironment:
    """実験環境簡単セットアップ"""
    env = ExperimentEnvironment(experiment_name)
    env.set_config(config)
    
    if dependencies:
        env.add_dependencies(dependencies)
        env.install_dependencies()
    
    # 標準ディレクトリ作成
    env.create_output_directories()
    env.prepare_data_directories(["input", "processed", "temp"])
    
    # 環境検証
    validation = env.validate_environment()
    if not validation["overall"]:
        logger.warning(f"Environment validation failed: {validation['errors']}")
    
    # スナップショット保存
    env.save_environment_snapshot()
    
    return env
