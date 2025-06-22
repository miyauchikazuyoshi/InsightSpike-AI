"""
Colab用ユーティリティ関数
GPU環境での実験実行をサポート
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_colab_environment() -> Dict[str, Any]:
    """Colab環境のセットアップと環境情報の取得"""
    
    # GPU利用可能性チェック
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if device_info['cuda_available']:
        device_info['gpu_name'] = torch.cuda.get_device_name(0)
        device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    # 環境変数設定
    os.environ['TORCH_DEVICE'] = device_info['device']
    os.environ['USE_CUDA'] = '1' if device_info['cuda_available'] else '0'
    os.environ['INSIGHTSPIKE_GPU_MODE'] = '1'
    
    # メモリ最適化設定
    if device_info['cuda_available']:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return device_info


def install_colab_dependencies():
    """Colab環境での依存関係インストール"""
    import subprocess
    import sys
    
    # pyproject_colab.tomlを使用してPoetryでインストール
    commands = [
        "pip install poetry",
        "poetry config virtualenvs.create false",  # Colabでは仮想環境不要
        "poetry install -f pyproject_colab.toml --no-dev"
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
            print(f"✅ {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {cmd}: {e}")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Colab用ログ設定"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def print_environment_info(device_info: Dict[str, Any]):
    """環境情報の表示"""
    print("🚀 Colab環境情報")
    print("=" * 50)
    print(f"Device: {device_info['device']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']}")
        print(f"GPU Memory: {device_info['gpu_memory']:.1f} GB")
        print(f"CUDA Devices: {device_info['cuda_device_count']}")
    
    print("=" * 50)


def clone_repository(repo_url: str = "https://github.com/miyauchikazuyoshi/InsightSpike-AI.git",
                    target_dir: str = "InsightSpike-AI") -> Path:
    """リポジトリのクローン"""
    import subprocess
    
    if not os.path.exists(target_dir):
        try:
            subprocess.run(["git", "clone", repo_url, target_dir], check=True)
            print(f"✅ Repository cloned: {target_dir}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Clone failed: {e}")
            raise
    else:
        print(f"📁 Repository already exists: {target_dir}")
    
    return Path(target_dir)


def setup_pythonpath(repo_path: Path):
    """Python パスの設定"""
    import sys
    
    src_path = str(repo_path / "src")
    experiments_path = str(repo_path / "experiments")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if experiments_path not in sys.path:
        sys.path.insert(0, experiments_path)
    
    print(f"✅ Python path configured: {src_path}, {experiments_path}")


def get_gpu_memory_usage() -> Dict[str, float]:
    """GPU メモリ使用量の取得"""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
    }


def clear_gpu_cache():
    """GPU キャッシュのクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")


def optimize_for_gpu(model, enable_mixed_precision: bool = True):
    """モデルのGPU最適化"""
    if torch.cuda.is_available():
        model = model.cuda()
        
        if enable_mixed_precision:
            # 自動混合精度の設定
            model = torch.compile(model) if hasattr(torch, 'compile') else model
            
        print("🚀 Model optimized for GPU")
    
    return model
