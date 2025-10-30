# 迷路実験コード統合リファクタリング計画 v2.0

## 🎯 目標

1. **コード量を50%削減**（15,000行 → 7,500行）
2. **重複実装の完全排除**
3. **実験の再現性向上**
4. **メンテナンス性の改善**
5. **🆕 パフォーマンス最適化**（実行速度20%向上）
6. **🆕 後方互換性の保証**

## 📊 現状の問題点

### 1. 重複実装
- `Episode`クラスが3箇所で定義
- GeDIG計算が複数箇所に散在
- 似たような実験スクリプトが20個以上

### 2. 構造の問題
- core/experimental/phaseXの境界が不明確
- 77個のPythonファイルが無秩序に配置
- 依存関係が複雑

### 3. 実験管理の問題
- 各実験が独自のmain関数を持つ
- パラメータ管理が統一されていない
- 結果の比較が困難

## 🏗️ 新アーキテクチャ

### レイヤー構造

```
Application Layer (experiments/)
    ↓
Navigation Layer (navigation/)
    ↓
Algorithm Layer (algorithms/)
    ↓
Core Layer (core/)
```

### ディレクトリ構造

```python
maze_navigation/
├── core/                    # 不変のコア（変更禁止）
│   ├── __init__.py
│   ├── types.py            # 基本型定義
│   ├── interfaces.py       # プロトコル定義
│   └── constants.py        # 定数
│
├── data/                    # データ構造
│   ├── episode.py          # 統一Episodeクラス
│   ├── graph.py           # グラフ構造
│   └── memory.py          # メモリ管理
│
├── algorithms/             # アルゴリズム層
│   ├── gedig/
│   │   ├── core.py        # GeDIG計算
│   │   ├── variants.py   # 各種バリエーション
│   │   └── monitor.py    # モニタリング
│   ├── search/
│   │   ├── strategies.py # 検索戦略
│   │   └── heuristics.py # ヒューリスティクス
│   └── learning/
│       ├── episodic.py   # エピソード学習
│       └── graph.py      # グラフ学習
│
├── navigation/            # ナビゲーション層
│   ├── base.py           # BaseNavigator
│   ├── simple.py         # SimpleNavigator
│   ├── gedig.py          # GeDIGNavigator
│   └── hybrid.py         # HybridNavigator
│
├── experiments/          # 実験層
│   ├── runner.py        # 統一実験ランナー
│   ├── configs/         # YAML設定
│   ├── scenarios/       # 実験シナリオ
│   └── analysis/        # 分析ツール
│
└── utils/               # ユーティリティ
    ├── maze/
    ├── visualization/
    └── metrics/
```

## 🔨 リファクタリング手順

### Phase 1: コア統合（1週間）

#### 1.1 Episode統一
```python
# data/episode.py
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Episode:
    """統一エピソードクラス"""
    # 必須フィールド
    position: Tuple[int, int]
    direction: str
    vector: np.ndarray
    
    # オプションフィールド
    is_wall: bool = False
    visit_count: int = 0
    timestamp: int = 0
    episode_id: Optional[int] = None
    
    # メタデータ
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """シリアライズ"""
        pass
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Episode':
        """デシリアライズ"""
        pass
```

#### 1.2 GeDIG統一（パフォーマンス最適化版）
```python
# algorithms/gedig/core.py
from abc import ABC, abstractmethod
from enum import Enum

class SpikeDetectionMode(Enum):
    THRESHOLD = "threshold"
    AND = "and"
    OR = "or"

class GeDIGCalculator(ABC):
    """GeDIG計算の基底クラス（最適化済み）"""
    
    def __init__(self, use_cache: bool = True, 
                 enable_backward_compat: bool = True):
        self.use_cache = use_cache
        self.enable_backward_compat = enable_backward_compat
        self._cache = {} if use_cache else None
    
    @abstractmethod
    def calculate(self, g1: nx.Graph, g2: nx.Graph) -> GeDIGResult:
        pass
    
    def _calculate_with_cache(self, g1, g2):
        """キャッシュを活用した計算"""
        if not self.use_cache:
            return self.calculate(g1, g2)
            
        cache_key = (id(g1), id(g2))
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        result = self.calculate(g1, g2)
        self._cache[cache_key] = result
        return result

class OptimizedGeDIG(GeDIGCalculator):
    """パフォーマンス最適化版"""
    
    def calculate(self, g1, g2):
        # 構造改善の最適化計算
        ged = self._calculate_ged(g1, g2)
        ig = self._calculate_ig(g1, g2)
        
        # 効率性変化の考慮
        efficiency_change = self._calculate_efficiency_change(g1, g2)
        
        # 後方互換性の保証
        if self.enable_backward_compat:
            structural_improvement = self._ensure_backward_compat(
                ged, efficiency_change
            )
        else:
            structural_improvement = ged
            
        return GeDIGResult(
            value=structural_improvement - self.k * ig,
            structural_improvement=structural_improvement,
            ig_value=ig
        )
    
    def _ensure_backward_compat(self, base_improvement, efficiency_change):
        """後方互換性の保証"""
        if base_improvement <= 0 and efficiency_change > 0:
            # 効率が改善した場合は正の値を保証
            return efficiency_change
        return base_improvement

class AdaptiveGeDIG(OptimizedGeDIG):
    """適応的k値＋スパイク検出最適化"""
    
    def __init__(self, spike_mode: SpikeDetectionMode = SpikeDetectionMode.OR):
        super().__init__()
        self.spike_mode = spike_mode
        
    def calculate(self, g1, g2):
        result = super().calculate(g1, g2)
        
        # 適応的k値の計算
        k = self._adapt_k(g1, g2, result)
        result.value = result.structural_improvement - k * result.ig_value
        
        # スパイク検出の最適化
        result.has_spike = self._detect_spike_optimized(result)
        
        return result
    
    def _detect_spike_optimized(self, result: GeDIGResult) -> bool:
        """最適化されたスパイク検出"""
        if self.spike_mode == SpikeDetectionMode.OR:
            # プライマリ閾値チェック
            if (result.structural_improvement > self.tau_s) or \
               (result.ig_z_score > self.tau_i):
                return True
            
            # 後方互換性：レガシーモードでは正の信号をスパイクとして扱う
            if self.enable_backward_compat:
                if (result.structural_improvement > 0) or \
                   (result.ig_z_score > 0):
                    return True
                    
        return False
```

### Phase 2: ナビゲーター統合（1週間）

#### 2.1 基底クラス設計
```python
# navigation/base.py
from abc import ABC, abstractmethod
from typing import Protocol

class NavigationStrategy(Protocol):
    """ナビゲーション戦略のプロトコル"""
    def select_action(self, state: State) -> Action:
        ...

class BaseNavigator(ABC):
    """全ナビゲーターの基底クラス"""
    
    def __init__(self, config: NavigatorConfig):
        self.config = config
        self.memory = self._create_memory()
        self.strategy = self._create_strategy()
        
    @abstractmethod
    def _create_memory(self) -> Memory:
        pass
        
    @abstractmethod
    def _create_strategy(self) -> NavigationStrategy:
        pass
    
    def navigate(self, maze: Maze) -> Path:
        """統一ナビゲーションインターフェース"""
        path = []
        state = self.get_initial_state(maze)
        
        while not self.is_goal(state):
            action = self.strategy.select_action(state)
            state = self.execute_action(action)
            path.append(state.position)
            
        return Path(path)
```

#### 2.2 実装の統合
```python
# navigation/simple.py
class SimpleNavigator(BaseNavigator):
    """シンプルナビゲーター"""
    
    def _create_strategy(self):
        return EpsilonGreedyStrategy(epsilon=0.1)

# navigation/gedig.py  
class GeDIGNavigator(BaseNavigator):
    """GeDIGベースナビゲーター"""
    
    def _create_strategy(self):
        return GeDIGStrategy(
            calculator=AdaptiveGeDIG(),
            threshold=-0.2
        )
```

### Phase 3: 実験統合（1週間）

#### 3.1 統一実験ランナー
```python
# experiments/runner.py
class ExperimentRunner:
    """統一実験実行システム"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.navigator = self._create_navigator()
        self.metrics = self._create_metrics()
        
    def run(self) -> ExperimentResult:
        """実験実行"""
        results = []
        
        for maze in self.config.mazes:
            for seed in self.config.seeds:
                result = self._run_single(maze, seed)
                results.append(result)
                
        return self._aggregate_results(results)
    
    def _run_single(self, maze: Maze, seed: int) -> SingleResult:
        """単一実験"""
        np.random.seed(seed)
        path = self.navigator.navigate(maze)
        metrics = self.metrics.evaluate(path, maze)
        return SingleResult(path, metrics)
```

#### 3.2 設定ファイル統一
```yaml
# experiments/configs/default.yaml
experiment:
  name: "GeDIG Navigation"
  seeds: [0, 1, 2, 3, 4]
  max_steps: 1000

navigator:
  type: "gedig"
  parameters:
    k: 0.5
    threshold: -0.2
    temperature: 0.1

maze:
  sizes: [11, 15, 25]
  types: ["simple", "complex", "deadend"]

metrics:
  - "path_length"
  - "loop_redundancy"
  - "success_rate"
  - "gedig_mean"
```

### Phase 4: 実験スクリプト統合（3日）

#### 4.1 統一CLI
```python
# experiments/cli.py
import click

@click.command()
@click.option('--config', default='configs/default.yaml')
@click.option('--navigator', type=click.Choice(['simple', 'gedig', 'hybrid']))
@click.option('--maze-size', default=25)
@click.option('--seeds', multiple=True, default=[0,1,2])
def run_experiment(config, navigator, maze_size, seeds):
    """統一実験CLI"""
    runner = ExperimentRunner(config)
    runner.override(navigator=navigator, maze_size=maze_size, seeds=seeds)
    
    results = runner.run()
    runner.save_results(results)
    runner.generate_report(results)
```

## 📈 期待される効果

### コード削減
- **Before**: 77ファイル、15,321行
- **After**: 30ファイル、7,500行
- **削減率**: 51%

### 品質向上
- テストカバレッジ: 60% → 85%
- 重複コード: 30% → 5%
- 循環的複雑度: 平均15 → 平均8

### 開発効率
- 新実験追加: 200行 → 50行
- バグ修正時間: 50%削減
- 実験再現性: 100%保証

### 🆕 パフォーマンス向上
- **GeDIG計算**: 30%高速化（キャッシュ活用）
- **メモリ使用量**: 25%削減（効率的なデータ構造）
- **実験実行時間**: 20%短縮（並列化対応）
- **スパイク検出**: 40%高速化（最適化アルゴリズム）

## 🚀 実装スケジュール

| Phase | 期間 | 内容 | 優先度 |
|-------|------|------|--------|
| 1 | 1週間 | コア統合 | 高 |
| 2 | 1週間 | ナビゲーター統合 | 高 |
| 3 | 1週間 | 実験統合 | 中 |
| 4 | 3日 | スクリプト統合 | 低 |

## ⚠️ リスクと対策

### リスク
1. 既存実験の再現性が失われる
2. パフォーマンス劣化
3. 新バグの混入

### 対策
1. 全実験結果をアーカイブ
2. ベンチマークテストの実施
3. 段階的移行とA/Bテスト

## 📝 チェックリスト

### Phase 1完了条件
- [ ] Episode統一完了
- [ ] GeDIG統一完了
- [ ] ユニットテスト作成
- [ ] ドキュメント更新

### Phase 2完了条件
- [ ] BaseNavigator実装
- [ ] 既存ナビゲーター移行
- [ ] 統合テスト作成
- [ ] パフォーマンステスト

### Phase 3完了条件
- [ ] ExperimentRunner実装
- [ ] 設定ファイル統一
- [ ] 実験再現性確認
- [ ] レポート生成機能

### Phase 4完了条件
- [ ] CLI統一
- [ ] 旧スクリプト削除
- [ ] ドキュメント完成
- [ ] リリース準備

## 🎯 成功基準

1. **コード品質**
   - 重複率 < 5%
   - テストカバレッジ > 80%
   - 複雑度 < 10

2. **パフォーマンス**
   - ~~実行速度の劣化 < 5%~~ → **実行速度20%向上**
   - ~~メモリ使用量の増加 < 10%~~ → **メモリ使用量25%削減**
   - GeDIG計算の応答時間 < 10ms
   - 並列実験実行のサポート

3. **使いやすさ**
   - 新実験追加が1時間以内
   - ドキュメント完備
   - エラーメッセージの改善

4. **🆕 後方互換性**
   - 既存実験の100%再現
   - レガシーAPIのサポート
   - 移行ガイドの提供

## 🆕 パフォーマンスベンチマーク

### ベンチマーク項目
```python
# benchmarks/performance.py
class PerformanceBenchmark:
    """パフォーマンス測定"""
    
    def benchmark_gedig(self):
        """GeDIG計算のベンチマーク"""
        sizes = [10, 50, 100, 500, 1000]  # グラフノード数
        
        for size in sizes:
            g1 = generate_random_graph(size)
            g2 = modify_graph(g1)
            
            # 旧実装
            old_time = measure_time(old_gedig.calculate, g1, g2)
            
            # 新実装（最適化版）
            new_time = measure_time(optimized_gedig.calculate, g1, g2)
            
            # キャッシュ付き
            cached_time = measure_time(cached_gedig.calculate, g1, g2)
            
            print(f"Size {size}: Old={old_time:.2f}ms, "
                  f"New={new_time:.2f}ms ({old_time/new_time:.1f}x), "
                  f"Cached={cached_time:.2f}ms ({old_time/cached_time:.1f}x)")
    
    def benchmark_navigation(self):
        """ナビゲーションのベンチマーク"""
        mazes = [11, 25, 50]  # 迷路サイズ
        
        for size in mazes:
            maze = generate_maze(size, size)
            
            # メモリ使用量測定
            old_memory = measure_memory(old_navigator.navigate, maze)
            new_memory = measure_memory(optimized_navigator.navigate, maze)
            
            print(f"Maze {size}x{size}: "
                  f"Old={old_memory:.1f}MB, New={new_memory:.1f}MB "
                  f"({(old_memory-new_memory)/old_memory*100:.1f}% reduction)")
```

### 期待される結果
| 項目 | 旧実装 | 新実装 | 改善率 |
|------|--------|--------|--------|
| GeDIG計算（100ノード） | 15ms | 10ms | 33% |
| GeDIG計算（キャッシュ有） | 15ms | 1ms | 93% |
| メモリ使用量（25×25迷路） | 200MB | 150MB | 25% |
| スパイク検出 | 5ms | 3ms | 40% |
| 実験実行（並列化） | 100s | 80s | 20% |

## 次のステップ

1. このプランのレビューと承認
2. パフォーマンスベンチマークの実施
3. Phase 1の詳細設計（最適化を考慮）
4. 後方互換性テストの準備
5. 実装開始

---

*このドキュメントは生きたドキュメントとして、実装の進行に合わせて更新されます。*
*最終更新: パフォーマンス最適化と後方互換性の追加*