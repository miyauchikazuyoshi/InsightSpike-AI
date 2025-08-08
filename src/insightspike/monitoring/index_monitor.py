"""
統合インデックスのパフォーマンスモニタリング
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class IndexPerformanceMonitor:
    """統合インデックスのパフォーマンスを監視"""
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: メトリクス保持のウィンドウサイズ
        """
        self.window_size = window_size
        
        # メトリクス保存用
        self.search_times = deque(maxlen=window_size)
        self.add_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        
        # カウンタ
        self.total_searches = 0
        self.total_adds = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # アラート設定
        self.alert_thresholds = {
            'search_time_ms': 10.0,
            'add_time_ms': 5.0,
            'memory_mb': 1000.0
        }
        
    def record_search(self, elapsed_time: float, hit_cache: bool = False):
        """検索操作を記録"""
        self.search_times.append(elapsed_time)
        self.total_searches += 1
        
        if hit_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # アラートチェック
        if elapsed_time * 1000 > self.alert_thresholds['search_time_ms']:
            logger.warning(
                f"検索時間がしきい値を超過: {elapsed_time*1000:.2f}ms > "
                f"{self.alert_thresholds['search_time_ms']}ms"
            )
    
    def record_add(self, elapsed_time: float):
        """追加操作を記録"""
        self.add_times.append(elapsed_time)
        self.total_adds += 1
        
        if elapsed_time * 1000 > self.alert_thresholds['add_time_ms']:
            logger.warning(
                f"追加時間がしきい値を超過: {elapsed_time*1000:.2f}ms > "
                f"{self.alert_thresholds['add_time_ms']}ms"
            )
    
    def record_memory(self, memory_mb: float):
        """メモリ使用量を記録"""
        self.memory_usage.append(memory_mb)
        
        if memory_mb > self.alert_thresholds['memory_mb']:
            logger.warning(
                f"メモリ使用量がしきい値を超過: {memory_mb:.1f}MB > "
                f"{self.alert_thresholds['memory_mb']}MB"
            )
    
    def get_metrics(self) -> Dict:
        """現在のメトリクスを取得"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_searches': self.total_searches,
            'total_adds': self.total_adds,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            )
        }
        
        # 検索時間統計
        if self.search_times:
            search_times_ms = [t * 1000 for t in self.search_times]
            metrics['search_time'] = {
                'avg_ms': np.mean(search_times_ms),
                'p50_ms': np.percentile(search_times_ms, 50),
                'p95_ms': np.percentile(search_times_ms, 95),
                'p99_ms': np.percentile(search_times_ms, 99),
                'max_ms': max(search_times_ms)
            }
        
        # 追加時間統計
        if self.add_times:
            add_times_ms = [t * 1000 for t in self.add_times]
            metrics['add_time'] = {
                'avg_ms': np.mean(add_times_ms),
                'p50_ms': np.percentile(add_times_ms, 50),
                'p95_ms': np.percentile(add_times_ms, 95),
                'max_ms': max(add_times_ms)
            }
        
        # メモリ使用量
        if self.memory_usage:
            metrics['memory'] = {
                'current_mb': self.memory_usage[-1],
                'avg_mb': np.mean(self.memory_usage),
                'max_mb': max(self.memory_usage)
            }
        
        return metrics
    
    def check_health(self) -> Dict:
        """ヘルスチェック"""
        metrics = self.get_metrics()
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        # 検索時間チェック
        if 'search_time' in metrics:
            if metrics['search_time']['p95_ms'] > self.alert_thresholds['search_time_ms']:
                health['status'] = 'degraded'
                health['issues'].append(
                    f"検索時間P95が高い: {metrics['search_time']['p95_ms']:.2f}ms"
                )
        
        # キャッシュヒット率チェック
        if metrics['cache_hit_rate'] < 0.8:
            health['status'] = 'degraded'
            health['issues'].append(
                f"キャッシュヒット率が低い: {metrics['cache_hit_rate']:.1%}"
            )
        
        # メモリチェック
        if 'memory' in metrics:
            if metrics['memory']['current_mb'] > self.alert_thresholds['memory_mb']:
                health['status'] = 'unhealthy'
                health['issues'].append(
                    f"メモリ使用量が高い: {metrics['memory']['current_mb']:.1f}MB"
                )
        
        return health


class IndexMonitoringDecorator:
    """統合インデックスにモニタリング機能を追加するデコレータ"""
    
    def __init__(self, index, monitor: Optional[IndexPerformanceMonitor] = None):
        self.index = index
        self.monitor = monitor or IndexPerformanceMonitor()
        
    def search(self, query_vector, **kwargs):
        """モニタリング付き検索"""
        start = time.time()
        try:
            result = self.index.search(query_vector, **kwargs)
            elapsed = time.time() - start
            self.monitor.record_search(elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start
            self.monitor.record_search(elapsed)
            raise
    
    def add_episode(self, episode):
        """モニタリング付きエピソード追加"""
        start = time.time()
        try:
            result = self.index.add_episode(episode)
            elapsed = time.time() - start
            self.monitor.record_add(elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start
            self.monitor.record_add(elapsed)
            raise
    
    def add_vector(self, vec, metadata):
        """モニタリング付きベクトル追加"""
        start = time.time()
        try:
            result = self.index.add_vector(vec, metadata)
            elapsed = time.time() - start
            self.monitor.record_add(elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start
            self.monitor.record_add(elapsed)
            raise
    
    def get_metrics(self) -> Dict:
        """メトリクス取得"""
        # メモリ使用量を推定
        if hasattr(self.index, 'normalized_vectors'):
            n = len(self.index.normalized_vectors)
            d = self.index.dimension if hasattr(self.index, 'dimension') else 768
            memory_mb = (n * d * 4) / (1024 * 1024)  # float32と仮定
            self.monitor.record_memory(memory_mb)
        
        return self.monitor.get_metrics()
    
    def check_health(self) -> Dict:
        """ヘルスチェック"""
        return self.monitor.check_health()
    
    def __getattr__(self, name):
        """その他のメソッドは直接転送"""
        return getattr(self.index, name)