"""
Core Metrics for All Experiments
================================

統一された評価指標を全実験で収集するためのユーティリティ。
これにより実験間の横断比較が可能になる。
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import psutil
import logging

logger = logging.getLogger(__name__)


class CoreMetrics:
    """全実験共通のコアメトリクス収集器"""
    
    # 必須メトリクス（全実験で必ず記録）
    REQUIRED_METRICS = {
        'delta_ged_mean',      # ΔGED平均
        'delta_ged_std',       # ΔGED標準偏差
        'delta_ig_mean',       # ΔIG平均
        'delta_ig_std',        # ΔIG標準偏差
        'avg_search_depth',    # 平均検索深度
        'search_k',            # 検索パラメータk
        'total_episodes',      # 総エピソード数
        'runtime_seconds',     # 実行時間（秒）
        'memory_mb',          # メモリ使用量（MB）
    }
    
    # オプションメトリクス（実験タイプによって追加）
    OPTIONAL_METRICS = {
        # 迷路系
        'success_rate',
        'avg_steps_to_goal',
        'wall_hit_rate',
        'unique_cells_visited',
        
        # QA系
        'f1_score',
        'precision',
        'recall',
        'avg_response_time_ms',
        
        # グラフ系
        'graph_nodes',
        'graph_edges',
        'avg_node_degree',
        'clustering_coefficient',
    }
    
    def __init__(self, experiment_name: str, seed: Optional[int] = None):
        """
        Args:
            experiment_name: 実験名
            seed: ランダムシード（再現性のため）
        """
        self.experiment_name = experiment_name
        self.seed = seed
        self.start_time = time.time()
        
        # メトリクス記録
        self.metrics: Dict[str, Any] = {
            'experiment_name': experiment_name,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 時系列データ
        self.time_series: Dict[str, List[float]] = {
            'delta_ged': [],
            'delta_ig': [],
            'search_depths': [],
        }
        
        # メモリベースライン
        self.baseline_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def record_gedig(self, delta_ged: float, delta_ig: float):
        """geDIG関連メトリクスを記録"""
        self.time_series['delta_ged'].append(delta_ged)
        self.time_series['delta_ig'].append(delta_ig)
    
    def record_search(self, depth: int, k: int):
        """検索関連メトリクスを記録"""
        self.time_series['search_depths'].append(depth)
        self.metrics['search_k'] = k  # 最新のkを保持
    
    def record_custom(self, key: str, value: Any):
        """カスタムメトリクスを記録"""
        if key in self.REQUIRED_METRICS or key in self.OPTIONAL_METRICS:
            self.metrics[key] = value
        else:
            # カスタムメトリクスは'custom_'プレフィックスを付ける
            self.metrics[f'custom_{key}'] = value
    
    def finalize(self) -> Dict[str, Any]:
        """最終メトリクスを計算して返す"""
        # 実行時間
        self.metrics['runtime_seconds'] = time.time() - self.start_time
        
        # メモリ使用量
        current_memory = self._get_memory_usage()
        self.metrics['memory_mb'] = current_memory - self.baseline_memory
        
        # geDIG統計
        if self.time_series['delta_ged']:
            self.metrics['delta_ged_mean'] = float(np.mean(self.time_series['delta_ged']))
            self.metrics['delta_ged_std'] = float(np.std(self.time_series['delta_ged']))
        else:
            self.metrics['delta_ged_mean'] = 0.0
            self.metrics['delta_ged_std'] = 0.0
        
        if self.time_series['delta_ig']:
            self.metrics['delta_ig_mean'] = float(np.mean(self.time_series['delta_ig']))
            self.metrics['delta_ig_std'] = float(np.std(self.time_series['delta_ig']))
        else:
            self.metrics['delta_ig_mean'] = 0.0
            self.metrics['delta_ig_std'] = 0.0
        
        # 検索深度
        if self.time_series['search_depths']:
            self.metrics['avg_search_depth'] = float(np.mean(self.time_series['search_depths']))
        else:
            self.metrics['avg_search_depth'] = 0.0
        
        # 必須メトリクスの欠損チェック
        missing = []
        for key in self.REQUIRED_METRICS:
            if key not in self.metrics:
                missing.append(key)
                self.metrics[key] = None  # 欠損は明示的にNull
        
        if missing:
            logger.warning(f"Missing required metrics: {missing}")
        
        return self.metrics
    
    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """メトリクスを標準形式で保存"""
        if output_dir is None:
            output_dir = Path('results') / self.experiment_name
        else:
            output_dir = Path(output_dir)
        
        # ディレクトリ作成
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        run_id = f"{timestamp}{seed_suffix}"
        
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # メトリクス保存
        metrics_path = run_dir / 'metrics.json'
        final_metrics = self.finalize()
        final_metrics['run_id'] = run_id
        
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {metrics_path}")
        return metrics_path
    
    @classmethod
    def load(cls, metrics_path: Union[str, Path]) -> Dict[str, Any]:
        """保存されたメトリクスを読み込む"""
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def compare(cls, *metrics_paths: Union[str, Path]) -> Dict[str, Any]:
        """複数の実験結果を比較"""
        all_metrics = []
        for path in metrics_paths:
            all_metrics.append(cls.load(path))
        
        comparison = {
            'experiments': len(all_metrics),
            'metrics': {}
        }
        
        # 各メトリクスの統計を計算
        for key in cls.REQUIRED_METRICS:
            values = [m.get(key) for m in all_metrics if m.get(key) is not None]
            if values:
                comparison['metrics'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
        
        return comparison


class ExperimentRunner:
    """実験実行の標準フレームワーク"""
    
    def __init__(self, experiment_name: str, config: Optional[Dict] = None):
        """
        Args:
            experiment_name: 実験名
            config: 実験設定
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.seed = self.config.get('seed', None)
        
        # メトリクス収集器を初期化
        self.metrics = CoreMetrics(experiment_name, self.seed)
        
        # 出力ディレクトリ設定
        self.output_dir = Path('results') / experiment_name
    
    def run(self, main_func, *args, **kwargs) -> Dict[str, Any]:
        """
        実験を実行してメトリクスを収集
        
        Args:
            main_func: 実験のメイン関数
            *args, **kwargs: main_funcへの引数
        
        Returns:
            実験結果とメトリクス
        """
        logger.info(f"Starting experiment: {self.experiment_name}")
        
        try:
            # シード設定
            if self.seed is not None:
                np.random.seed(self.seed)
                logger.info(f"Random seed set to {self.seed}")
            
            # 実験実行
            result = main_func(self.metrics, *args, **kwargs)
            
            # メトリクス保存
            metrics_path = self.metrics.save(self.output_dir)
            
            return {
                'result': result,
                'metrics': self.metrics.finalize(),
                'metrics_path': str(metrics_path)
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            # エラーもメトリクスに記録
            self.metrics.record_custom('error', str(e))
            self.metrics.record_custom('status', 'failed')
            metrics_path = self.metrics.save(self.output_dir)
            
            return {
                'result': None,
                'error': str(e),
                'metrics': self.metrics.finalize(),
                'metrics_path': str(metrics_path)
            }


# 使用例
if __name__ == "__main__":
    # 実験での使い方
    def sample_experiment(metrics: CoreMetrics):
        """サンプル実験"""
        # geDIG値を記録
        for i in range(10):
            metrics.record_gedig(
                delta_ged=np.random.random(),
                delta_ig=np.random.random()
            )
            metrics.record_search(depth=np.random.randint(1, 6), k=30)
        
        # カスタムメトリクスを記録
        metrics.record_custom('success_rate', 0.75)
        metrics.record_custom('total_episodes', 100)
        
        return "実験完了"
    
    # 実験実行
    runner = ExperimentRunner('sample_experiment', config={'seed': 42})
    result = runner.run(sample_experiment)
    
    print(f"Result: {result}")