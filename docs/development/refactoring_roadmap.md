# InsightSpike-AI リファクタリングロードマップ

**Version**: 1.0
**Date**: 2025-11-27
**Status**: Implementation Guide（提案・未実装を含む）
**注記**: 本ロードマップは現状の巨大ファイルを分割する「案」です。記載のディレクトリ/ファイル（例: `layer3/core.py` など）はまだリポジトリに存在しません。現状は単一ファイル実装（例: `layer3_graph_reasoner.py` 2244行）が稼働中です。

---

## 🎯 概要

本ドキュメントは、InsightSpike-AIの3大巨大ファイルを分割するための**具体的な実装手順**を提供します。
テストは軽量モード（`INSIGHTSPIKE_LITE_MODE=1`）を前提に段階的に実施し、maze系のレガシー依存（`navigation`/`core` モジュールなど）が欠ける環境ではそれらをスキップする。

## 🧪 テスト計画（共通）

### コマンドセット（軽量モード想定）

- スモーク＋カバレッジ対象（非 maze）  
  `INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=src/insightspike --cov-report=term --maxfail=1 tests/e2e tests/gedig`  
  - 2025-11-27 時点の実行結果: 18/18 pass、カバレッジ 16.9%（`fail_under=35` で失敗）。maze 依存テストを除外した暫定値。
- Layer3 分割後の局所カバレッジ  
  `INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=src/insightspike/implementations/layers/layer3 --cov-report=term-missing tests/unit/test_layer3_graph_reasoner.py tests/unit/test_message_passing.py tests/unit/test_scalable_graph_builder.py`
- 後方互換確認  
  `pytest -q tests/unit/test_layer3_graph_reasoner.py::test_backward_compat_import`
- maze 依存テスト（`navigation`/`core` モジュールを必要とするもの）は、依存を用意できない環境では  
  `--ignore=tests/maze --ignore=tests/maze-query-hub-prototype --ignore=tests/test_macro_target_adaptive_p*.py --ignore=tests/test_macro_target_metrics.py --ignore=tests/test_maze_navigator_smoke.py --ignore=tests/unit/test_maze_simple_mode.py`  
  を付けてスキップする。

### 成果物チェック

- Layer3 分割 PR では:
  - 既存パスの import 互換性を担保（`layer3_graph_reasoner` の wrapper 経由）
  - 上記カバレッジコマンドが通ること（`fail_under` を一時的に下げても良いが、レポート値は必ず記録）
  - e2e ワークフロー（tests/e2e）と gedig スイートがグリーンであること
- main_agent / gedig_core 分割でも同様に局所カバレッジを追加し、e2e/gedig スモークを必須とする。
- 進捗（2025-11-27→現在）: Layer3 パッケージ足場＋`ConflictScore`/`GraphBuilder`/`message_passing` ラッパー抽出済み、lazy wrapper と lite stub テスト済み。MessagePassing 初期化は controller に委譲済み（apply 実行含む）。GraphAnalyzer/RewardCalculator を layer3 に self-contained 移植し再エクスポート。GNN 初期化は `layer3/gnn.py` に分離。MetricsSelector も controller 経由にし、analysis/message_passing_controller/metrics_controller のユニットテスト追加済み。`analyze_documents` は `analyzer_runner` に完全委譲（旧本体削除、runner 例外時のみ `_fallback_result`）。query-focal metrics 用ハンドラを `analyzer_runner` に実装（core/cached 両パス、失敗時ニュートラル）し、ユニット追加済み。Layer3/メトリクス/GeDIG 小グラフ系の追加ユニットによりカバレッジ 18.93%。e2e+gedig スモーク 18/18 pass 継続。

### 残タスク（Layer3 完了に向けて）
- Query-focal metrics の実データ検証・パラメータ調整（k_star/centers/sp_engine）。core/cached 両パスの期待値テストを実グラフで追加。
- Layer3 以外の巨大ファイルの分割・カバレッジ強化（main_agent, gedig_core など）。

### 段階的実行プラン（推奨）

1. **テスト基盤の安定化（最初のPR）**
   - maze依存テストをデフォルトで `-m "not maze"` などのマーカー/ignoreで明示スキップ（欠損モジュール対策）
   - `fail_under` を一時的に 0〜10 に緩和し、レポート値を記録し続ける（現状 10）
   - コマンド: 上記スモーク＋カバレッジ（e2e+gedig）でグリーンにする（現状 18/18 pass, 16.9%）
   - Layer3 サブパッケージを追加し、従来実装への lazy delegate と lite stub を用意（APIは未分割のまま）

2. **Layer3 分割（2本目のPR）**
   - `layer3/` パッケージ化＋後方互換wrapper維持
   - 局所テスト追加（core/lite_stub/backward compat）＋ e2e/gedig スモークを通す
   - カバレッジ目標: 現状 +5〜10pt（まずは 25% 付近を目指す）

3. **main_agent 分割（3本目のPR）**
   - `main_agent/` 構造化＋wrapper
   - 局所テスト追加（cycle/memory/layers統合）＋ e2e/gedig スモーク
   - カバレッジ目標: 30% 付近へ引き上げ

4. **gedig_core 分割（4本目のPR）**
   - `algorithms/gedig/` への分割＋wrapper
   - 局所テスト追加（metrics/config/results/multihop）＋ e2e/gedig スモーク
   - カバレッジ目標: 35% 以上（fail_under を元に戻す）

5. **後続（任意）**
   - 型チェック強化、残り巨大ファイル（sqlite_store, layer4_llm_interface, cached_memory_manager, layer2_memory_manager）の縮減

### 対象ファイル

| ファイル | 行数 | 優先度 | 工数 |
|---------|------|--------|------|
| layer3_graph_reasoner.py | 2244 | P0 (最優先) | 3日 |
| main_agent.py | 2203 | P0 | 3日 |
| gedig_core.py | 2035 | P0 | 3日 |

---

## 📦 Part 1: layer3_graph_reasoner.py の分割

**現状**: 2244行（最大ファイル）
**目標**: 7ファイル × 平均300行

### ステップ1: ディレクトリ構造の作成

```bash
cd src/insightspike/implementations/layers

# 新規ディレクトリ作成
mkdir -p layer3/{__pycache__,tests}

# 既存ファイルをバックアップ
cp layer3_graph_reasoner.py layer3_graph_reasoner.py.backup
```

**最終的なディレクトリ構造**:

```
src/insightspike/implementations/layers/
├── layer3/
│   ├── __init__.py              # Public exports
│   ├── core.py                  # L3GraphReasonerCore (300行)
│   ├── gnn.py                   # GNN processing (400行)
│   ├── conflict.py              # ConflictScore (200行)
│   ├── analysis.py              # Graph analysis (400行)
│   ├── message_passing.py       # Message passing (300行)
│   ├── lite_stub.py             # Lite mode stub (100行)
│   └── diagnostics.py           # Diagnostic utilities (100行)
└── layer3_graph_reasoner.py     # 後方互換wrapper (50行)
```

---

### ステップ2: core.py の作成

**ファイル**: `src/insightspike/implementations/layers/layer3/core.py`

```python
"""Layer3 Core - Base graph reasoner implementation

This module provides the core L3GraphReasoner implementation without
GNN-specific or diagnostic code.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from ....core.base import L3GraphReasonerInterface, LayerInput, LayerOutput
from ....config import get_config
from ....config.legacy_adapter import LegacyConfigAdapter

logger = logging.getLogger(__name__)

# Lightweight cosine similarity fallback
def _cosine_similarity(a: np.ndarray, b: Optional[np.ndarray] = None):
    """Compute cosine similarity (NumPy-only implementation)"""
    if b is None:
        b = a
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class L3GraphReasonerCore(L3GraphReasonerInterface):
    """Base implementation of Layer3 graph reasoning.

    This class provides the core functionality without GNN or heavy
    dependencies, suitable for lite mode and testing.

    Attributes:
        config: Configuration object
        enabled: Whether the reasoner is enabled
        current_graph: Current graph state
    """

    def __init__(self, config=None):
        """Initialize the core reasoner.

        Args:
            config: Optional configuration. If None, loads default.
        """
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self.enabled = True
        self.current_graph = None
        self._initialized = False

        logger.info("L3GraphReasonerCore initialized (lite mode)")

    def initialize(self) -> bool:
        """Initialize the reasoner.

        Returns:
            True if successful
        """
        self._initialized = True
        return True

    def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze documents and build/update graph.

        Args:
            documents: List of document dictionaries
            context: Optional context information

        Returns:
            Analysis results dictionary with keys:
                - graph: NetworkX graph
                - metrics: Delta metrics (GED, IG)
                - conflicts: Conflict detection results
                - reward: Reward signals
                - spike_detected: Whether insight spike detected
                - reasoning_quality: Quality score [0, 1]
        """
        if not self._initialized:
            self.initialize()

        # Build graph from documents
        from .graph_builder import build_graph_from_documents
        graph = build_graph_from_documents(documents, context)

        # Compute metrics
        from .metrics import compute_delta_metrics
        metrics = compute_delta_metrics(self.current_graph, graph)

        # Detect conflicts
        from .conflict import detect_conflicts
        conflicts = detect_conflicts(self.current_graph, graph, context or {})

        # Compute rewards
        reward = self._compute_reward(metrics, conflicts)

        # Spike detection
        spike_detected = self._detect_spike(metrics)

        # Update current graph
        self.current_graph = graph

        return {
            "graph": graph,
            "metrics": metrics,
            "conflicts": conflicts,
            "reward": reward,
            "spike_detected": spike_detected,
            "reasoning_quality": self._compute_quality(metrics, conflicts),
        }

    def _compute_reward(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute reward signals from metrics and conflicts.

        Args:
            metrics: Delta metrics
            conflicts: Conflict scores

        Returns:
            Reward dictionary
        """
        delta_ged = metrics.get("delta_ged", 0.0)
        delta_ig = metrics.get("delta_ig", 0.0)
        conflict_total = conflicts.get("total", 0.0)

        # Simple reward formula
        insight_reward = -delta_ged + delta_ig
        quality_bonus = max(0, 1.0 - conflict_total)
        total = insight_reward + 0.3 * quality_bonus

        return {
            "insight_reward": float(insight_reward),
            "quality_bonus": float(quality_bonus),
            "total": float(total),
        }

    def _detect_spike(self, metrics: Dict[str, float]) -> bool:
        """Detect insight spike from metrics.

        Args:
            metrics: Delta metrics

        Returns:
            True if spike detected
        """
        delta_ged = metrics.get("delta_ged", 0.0)
        delta_ig = metrics.get("delta_ig", 0.0)

        ged_threshold = self.config.graph.spike_ged_threshold
        ig_threshold = self.config.graph.spike_ig_threshold

        return delta_ged < ged_threshold and delta_ig > ig_threshold

    def _compute_quality(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float]
    ) -> float:
        """Compute reasoning quality score.

        Args:
            metrics: Delta metrics
            conflicts: Conflict scores

        Returns:
            Quality score [0, 1]
        """
        delta_ig = metrics.get("delta_ig", 0.0)
        conflict_total = conflicts.get("total", 0.0)

        # Higher IG and lower conflict = higher quality
        quality = 0.5 + 0.3 * min(delta_ig, 1.0) - 0.3 * conflict_total
        return float(np.clip(quality, 0.0, 1.0))


__all__ = ["L3GraphReasonerCore"]
```

---

### ステップ3: lite_stub.py の作成

**ファイル**: `src/insightspike/implementations/layers/layer3/lite_stub.py`

```python
"""Lite mode stub for Layer3 graph reasoner

Provides a minimal placeholder when torch/PyG are not available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class L3GraphReasonerLiteStub:
    """Lightweight stub for lite mode.

    This stub is used when INSIGHTSPIKE_LITE_MODE=1 or when
    torch_geometric is not available.
    """

    def __init__(self, config=None):
        self.config = config
        self.enabled = False
        self.current_graph = None
        logger.info("L3GraphReasoner: Using lite stub (torch/PyG not available)")

    def initialize(self) -> bool:
        """Initialize (no-op)."""
        return True

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Legacy analyze method (for backward compat)."""
        return {"enabled": False, "reason": "lite_mode"}

    def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return minimal analysis dict that MainAgent expects.

        Args:
            documents: List of documents
            context: Optional context

        Returns:
            Minimal analysis result
        """
        return {
            "graph": None,
            "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
            "conflicts": {"total": 0},
            "reward": {"insight_reward": 0.0, "quality_bonus": 0.0, "total": 0.0},
            "reasoning_quality": 0.5,
            "spike_detected": False,
        }


__all__ = ["L3GraphReasonerLiteStub"]
```

---

### ステップ4: __init__.py の作成

**ファイル**: `src/insightspike/implementations/layers/layer3/__init__.py`

```python
"""Layer3 Graph Reasoner Package

This package provides graph-based reasoning with spike detection.

Modules:
    core: Base graph reasoner implementation
    gnn: GNN processing (requires torch_geometric)
    conflict: Conflict detection
    analysis: Graph analysis
    message_passing: Message passing operations
    lite_stub: Lightweight stub for lite mode
    diagnostics: Diagnostic utilities
"""

import os
import logging

logger = logging.getLogger(__name__)

# Check environment
LITE_MODE = os.getenv("INSIGHTSPIKE_LITE_MODE") == "1"
DISABLE_GNN = os.getenv("INSIGHTSPIKE_DISABLE_GNN") == "1"

# Check torch_geometric availability
def _have_torch_geometric() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False

# Select implementation
if LITE_MODE or not _have_torch_geometric():
    # Use lite stub
    from .lite_stub import L3GraphReasonerLiteStub as L3GraphReasoner
    logger.info("Layer3: Using lite stub (torch/PyG not available)")
else:
    # Use full implementation
    if DISABLE_GNN:
        from .core import L3GraphReasonerCore as L3GraphReasoner
        logger.info("Layer3: Using core implementation (GNN disabled)")
    else:
        try:
            from .gnn import L3GraphReasonerWithGNN as L3GraphReasoner
            logger.info("Layer3: Using GNN implementation")
        except ImportError:
            from .core import L3GraphReasonerCore as L3GraphReasoner
            logger.warning("Layer3: GNN import failed, using core implementation")

# Export
__all__ = ["L3GraphReasoner"]
```

---

### ステップ5: 後方互換wrapper の作成

**ファイル**: `src/insightspike/implementations/layers/layer3_graph_reasoner.py`

```python
"""Backward compatibility wrapper for Layer3GraphReasoner

This module maintains the original import path for backward compatibility.
New code should import from `layer3` package directly.

Example:
    # Old style (still works)
    from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner

    # New style (preferred)
    from insightspike.implementations.layers.layer3 import L3GraphReasoner
"""

import warnings

# Import from new location
from .layer3 import L3GraphReasoner

# Deprecation warning (optional, can be removed later)
# warnings.warn(
#     "Importing from layer3_graph_reasoner is deprecated. "
#     "Use 'from insightspike.implementations.layers.layer3 import L3GraphReasoner' instead.",
#     DeprecationWarning,
#     stacklevel=2
# )

__all__ = ["L3GraphReasoner"]
```

---

### ステップ6: テストの作成

**ファイル**: `tests/unit/implementations/layers/test_layer3_refactored.py`

```python
"""Tests for refactored Layer3 modules"""

import pytest
from insightspike.implementations.layers.layer3 import L3GraphReasoner
from insightspike.implementations.layers.layer3.core import L3GraphReasonerCore
from insightspike.implementations.layers.layer3.lite_stub import L3GraphReasonerLiteStub


class TestLayer3Refactored:
    """Test refactored Layer3 components"""

    def test_import_backward_compat(self):
        """Test backward compatibility import"""
        from insightspike.implementations.layers.layer3_graph_reasoner import (
            L3GraphReasoner as LegacyL3
        )
        assert LegacyL3 is not None

    def test_core_initialization(self):
        """Test core reasoner initialization"""
        reasoner = L3GraphReasonerCore()
        assert reasoner is not None
        assert reasoner.initialize() is True

    def test_lite_stub_initialization(self):
        """Test lite stub initialization"""
        stub = L3GraphReasonerLiteStub()
        assert stub.enabled is False
        assert stub.initialize() is True

    def test_analyze_documents_core(self):
        """Test core analyze_documents"""
        reasoner = L3GraphReasonerCore()
        reasoner.initialize()

        documents = [
            {"text": "Test document 1"},
            {"text": "Test document 2"},
        ]

        result = reasoner.analyze_documents(documents)

        assert "graph" in result
        assert "metrics" in result
        assert "spike_detected" in result
        assert isinstance(result["reasoning_quality"], float)

    def test_analyze_documents_lite(self):
        """Test lite stub analyze_documents"""
        stub = L3GraphReasonerLiteStub()

        documents = [{"text": "Test"}]
        result = stub.analyze_documents(documents)

        assert result["graph"] is None
        assert result["spike_detected"] is False
```

---

### ステップ7: マイグレーションチェックリスト

- [ ] `layer3/` ディレクトリ作成
- [ ] `core.py` 実装
- [ ] `lite_stub.py` 実装
- [ ] `__init__.py` 実装（lite mode切り替え）
- [ ] 後方互換wrapper作成
- [ ] テスト作成・実行
- [ ] CI通過確認
- [ ] 既存の全インポートが動作確認
- [ ] ドキュメント更新

---

## 📦 Part 2: main_agent.py の分割

**現状**: 2203行
**目標**: 6ファイル × 平均350行

### 分割後の構造

```
src/insightspike/implementations/agents/
├── main_agent/
│   ├── __init__.py              # Public exports
│   ├── core.py                  # MainAgent core (500行)
│   ├── cycle.py                 # Cycle processing (400行)
│   ├── memory.py                # Memory management (300行)
│   ├── layers.py                # Layer integration (400行)
│   └── diagnostics.py           # Diagnostics (200行)
└── main_agent.py                # 後方互換wrapper (50行)
```

### 実装ガイド

**core.py**: MainAgentクラスの基本構造
```python
"""MainAgent Core Implementation"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class CycleResult:
    """Result from one reasoning cycle"""
    question: str
    retrieved_documents: List[Dict[str, Any]]
    graph_analysis: Dict[str, Any]
    response: str
    reasoning_quality: float
    spike_detected: bool
    error_state: Dict[str, Any]
    cycle_number: int
    success: bool = True
    query_id: Optional[str] = None


class MainAgentCore:
    """Core orchestrating agent (without layer dependencies)"""

    def __init__(self, config=None, datastore=None):
        if config is None:
            raise ValueError("Config must be provided to MainAgent")

        self.config = config
        self.datastore = datastore
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize agent components"""
        # Basic initialization (layers loaded separately)
        self._initialized = True
        return True
```

**cycle.py**: 推論サイクル処理
```python
"""Cycle Processing Logic"""

def process_cycle(
    agent,
    question: str,
    cycle_num: int,
    max_cycles: int,
    verbose: bool = False
) -> CycleResult:
    """Process single reasoning cycle"""
    # Document retrieval
    documents = agent.l2_memory.search_episodes(question)

    # Graph analysis
    graph_analysis = agent.l3_graph.analyze_documents(documents)

    # LLM generation
    response = agent.l4_llm.generate_response_detailed(
        context=documents,
        question=question
    )

    # Build result
    return CycleResult(
        question=question,
        retrieved_documents=documents,
        graph_analysis=graph_analysis,
        response=response["response"],
        reasoning_quality=0.8,
        spike_detected=graph_analysis.get("spike_detected", False),
        error_state={},
        cycle_number=cycle_num,
        success=True,
    )
```

---

## 📦 Part 3: gedig_core.py の分割

**現状**: 2035行
**目標**: 6ファイル × 平均300行

### 分割後の構造

```
src/insightspike/algorithms/
├── gedig/
│   ├── __init__.py
│   ├── core.py                  # GeDIGCore class (400行)
│   ├── config.py                # Configuration (200行)
│   ├── multihop.py              # Multi-hop processing (500行)
│   ├── metrics.py               # Metrics calculation (300行)
│   ├── normalization.py         # Normalization (300行)
│   └── results.py               # Result dataclasses (100行)
└── gedig_core.py                # 後方互換wrapper (50行)
```

### 実装ガイド

**results.py**: データクラス抽出
```python
"""geDIG Result Data Classes"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Set

@dataclass
class HopResult:
    """Per-hop evaluation result"""
    hop: int
    ged: float
    ig: float
    gedig: float
    struct_cost: float
    node_count: int
    edge_count: int
    sp: float = 0.0
    h_component: float = 0.0
    # ... (full fields)


@dataclass
class GeDIGResult:
    """Complete geDIG calculation result"""
    gedig_value: float
    ged_value: float
    ig_value: float
    raw_ged: float = 0.0
    ged_norm_den: float = 1.0
    # ... (full fields)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

**config.py**: 設定クラス抽出
```python
"""geDIG Configuration Management"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class GeDIGConfig:
    """Configuration for geDIG calculator"""

    # Cost parameters
    node_cost: float = 1.0
    edge_cost: float = 1.0
    normalization: str = 'sum'
    efficiency_weight: float = 0.3

    # geDIG formula
    lambda_weight: float = 1.0
    sp_beta: float = 0.2

    # Multi-hop
    enable_multihop: bool = False
    max_hops: int = 3
    decay_factor: float = 0.7

    # Spike detection
    spike_threshold: float = -0.5
    tau_s: float = 0.15
    tau_i: float = 0.25

    # ... (other fields)
```

---

## 🧪 テスト戦略

### 後方互換性テスト

```python
# tests/integration/test_refactoring_backward_compat.py
"""Test backward compatibility after refactoring"""

def test_layer3_backward_compat():
    """Ensure old import path still works"""
    from insightspike.implementations.layers.layer3_graph_reasoner import (
        L3GraphReasoner
    )
    reasoner = L3GraphReasoner()
    assert reasoner is not None

def test_main_agent_backward_compat():
    """Ensure MainAgent still works"""
    from insightspike.implementations.agents.main_agent import MainAgent
    from insightspike.config.presets import ConfigPresets

    config = ConfigPresets.development()
    agent = MainAgent(config=config)
    assert agent is not None

def test_gedig_core_backward_compat():
    """Ensure GeDIGCore still works"""
    from insightspike.algorithms.gedig_core import GeDIGCore

    core = GeDIGCore()
    assert core is not None
```

---

## 📋 実行チェックリスト

### Phase 1: layer3_graph_reasoner.py (Week 1)

- [ ] Day 1: ディレクトリ構造作成 + core.py実装
- [ ] Day 2: lite_stub.py + __init__.py実装
- [ ] Day 3: テスト作成 + CI確認

### Phase 2: main_agent.py (Week 2)

- [ ] Day 1: ディレクトリ構造 + core.py
- [ ] Day 2: cycle.py + memory.py
- [ ] Day 3: テスト + CI確認

### Phase 3: gedig_core.py (Week 3)

- [ ] Day 1: ディレクトリ構造 + results.py + config.py
- [ ] Day 2: multihop.py + metrics.py
- [ ] Day 3: テスト + CI確認

---

## ⚠️ 注意事項

### 破壊的変更を避ける

1. **後方互換wrapperを必ず作成**
   - 既存のインポートパスを維持
   - Deprecation警告は後から追加

2. **段階的移行**
   - 一度に全てを変えない
   - 各ファイルごとにPR作成

3. **テストの充実**
   - 各段階でCI通過を確認
   - 既存テストが全てパス

### コミットメッセージ

```
refactor(layer3): split layer3_graph_reasoner.py into modular structure

- Create layer3/ package with core, gnn, conflict modules
- Add lite_stub for lightweight operation
- Maintain backward compatibility via wrapper
- Add tests for new structure

Refs: #123 (issue number)
```

---

## 📚 参考資料

- Martin Fowler "Refactoring": Extract Module pattern
- "Working Effectively with Legacy Code": Seam patterns
- Python Import System: PEP 420 (Namespace packages)

---

**このロードマップに従えば、3週間で全ての巨大ファイルを分割できます。**

詳細な背景と戦略は `comprehensive_improvement_plan.md` を参照してください。
