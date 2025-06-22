"""
GPU最適化ヘルパー関数
大規模実験のための高性能計算サポート
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from contextlib import contextmanager


class GPUOptimizer:
    """GPU最適化管理クラス"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """モデルの最適化"""
        model = model.to(self.device)
        
        if self.device == 'cuda':
            # CUDAメモリ最適化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # モデルコンパイル（PyTorch 2.0+）
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='max-autotune')
                    print("🚀 Model compiled with torch.compile")
                except Exception as e:
                    print(f"⚠️ Compilation failed: {e}")
        
        return model
    
    def optimize_dataloader(self, dataset, batch_size: int = 32, 
                          num_workers: int = 2) -> DataLoader:
        """DataLoaderの最適化"""
        
        # GPU環境では大きなバッチサイズを使用
        if self.device == 'cuda':
            batch_size = min(batch_size * 2, 128)
            num_workers = min(num_workers * 2, 8)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device == 'cuda',
            persistent_workers=num_workers > 0
        )
    
    @contextmanager
    def mixed_precision_context(self):
        """自動混合精度コンテキスト"""
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def get_memory_stats(self) -> Dict[str, float]:
        """メモリ使用統計"""
        if self.device != 'cuda':
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'utilization_%': (torch.cuda.memory_allocated() / 
                            torch.cuda.get_device_properties(0).total_memory) * 100
        }


class BatchProcessor:
    """大規模データのバッチ処理"""
    
    def __init__(self, batch_size: int = 1000, device: str = 'cuda'):
        self.batch_size = batch_size
        self.device = device
    
    def process_embeddings(self, texts: List[str], model, 
                         show_progress: bool = True) -> torch.Tensor:
        """大規模テキストの埋め込み処理"""
        embeddings = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), self.batch_size), 
                              desc="Processing embeddings")
            except ImportError:
                iterator = range(0, len(texts), self.batch_size)
        else:
            iterator = range(0, len(texts), self.batch_size)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = model.encode(batch_texts, 
                                              convert_to_tensor=True,
                                              device=self.device)
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def process_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """大規模類似度行列の計算"""
        n = embeddings.size(0)
        similarity_matrix = torch.zeros(n, n, device=self.device)
        
        # チャンク単位で計算（メモリ効率化）
        chunk_size = min(self.batch_size, 1000)
        
        for i in range(0, n, chunk_size):
            for j in range(0, n, chunk_size):
                chunk_i = embeddings[i:i+chunk_size]
                chunk_j = embeddings[j:j+chunk_size]
                
                similarity_chunk = torch.mm(chunk_i, chunk_j.t())
                similarity_matrix[i:i+chunk_size, j:j+chunk_size] = similarity_chunk
        
        return similarity_matrix


def optimize_pytorch_settings():
    """PyTorch最適化設定"""
    if torch.cuda.is_available():
        # CUDAメモリ管理最適化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # メモリフラグメンテーション対策
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("🚀 PyTorch GPU settings optimized")
    
    # CPUスレッド数最適化
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    print(f"🔧 CPU threads: {torch.get_num_threads()}")


def benchmark_operations(device: str = 'cuda', size: int = 1000) -> Dict[str, float]:
    """GPU/CPU性能ベンチマーク"""
    import time
    
    # テストデータ生成
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # 行列乗算ベンチマーク
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(10):
        result = torch.mm(x, y)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    matrix_multiply_time = (end_time - start_time) / 10
    
    # メモリ転送ベンチマーク
    if device == 'cuda':
        cpu_tensor = torch.randn(size, size)
        
        start_time = time.time()
        for _ in range(10):
            gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        end_time = time.time()
        
        memory_transfer_time = (end_time - start_time) / 10
    else:
        memory_transfer_time = 0.0
    
    return {
        'matrix_multiply_ms': matrix_multiply_time * 1000,
        'memory_transfer_ms': memory_transfer_time * 1000,
        'device': device
    }
