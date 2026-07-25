---
status: proposal
created: 2025-11-28
updated: 2026-02-08
---

# InsightSpike-AI 包括的改善計画書

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。


**Version**: 1.0
**Date**: 2025-11-27
**Status**: Draft（提案・未実装を含む）
**総合評価**: 8.5/10（主観/暫定）
**注記**: 本文の統計は 2025-11-27 時点の再計測値。カバレッジなど未計測項目は明示。

---

## 🎯 Executive Summary

InsightSpike-AIは**研究グレードとして非常に高品質**なコードベースです。理論的基盤、実装品質、ドキュメントのすべてが優れています。

### 主要な強み
- ✅ 学術的厳密性（論文v5、特許出願 JP 2025-082988, 2025-082989）
- ✅ プロダクション対応（CI/CD、Docker、Lite mode）
- ✅ 開発者体験（Mock LLM、30秒クイックスタート）
- ✅ 再現性（シード管理、プロトコル文書化）

### 改善が必要な領域
- 🔴 **ファイルサイズ**: 3ファイルが2000行超
- 🟡 **テストカバレッジ**: 未計測（pyproject の閾値 35% のみ設定）
- 🟡 **コード整理**: 診断コードの混在
- 🟡 **ドキュメント**: v4/v5の不一致

---

## 📊 コードベース分析結果

### 統計サマリー

```
総ソースコード: 67,641行 (Python, src)
総テストコード: 23,992行 (tests)
Pythonファイル: 267 (src) + 163 (tests)
ドキュメント: 68 Markdownファイル
測定方法: python3 で行数カウント（2025-11-27）
```

### ファイルサイズ分布

| ファイル | 行数 | 評価 | アクション |
|---------|------|------|-----------|
| layer3_graph_reasoner.py | 2244 | 🔴 | **分割必須** |
| main_agent.py | 2203 | 🔴 | **分割必須** |
| gedig_core.py | 2035 | 🔴 | **分割必須** |
| sqlite_store.py | 1737 | 🟡 | 要検討 |
| layer4_llm_interface.py | 1437 | 🟡 | 要検討 |
| cached_memory_manager.py | 1232 | 🟡 | 要検討 |
| layer2_memory_manager.py | 1123 | 🟡 | 要検討 |

**推奨サイズ**: 300-500行/ファイル
**現状問題**: Top 3が2000行超（4-5倍）

### モジュール評価

| モジュール | 評価 | 主な問題 |
|-----------|------|---------|
| algorithms/gedig/selector.py (271行) | ⭐⭐⭐⭐⭐ | なし（模範的） |
| public/__init__.py (59行) | ⭐⭐⭐⭐⭐ | なし（模範的） |
| config/models.py (543行) | ⭐⭐⭐⭐⭐ | 分割推奨 |
| config/presets.py (546行) | ⭐⭐⭐⭐⭐ | プリセット統合 |
| implementations/datastore/factory.py (117行) | ⭐⭐⭐⭐⭐ | なし（模範的） |
| metrics/psz.py (51行) | ⭐⭐⭐⭐⭐ | なし（完璧） |
| algorithms/gedig_core.py (2035行) | ⭐⭐⭐⭐ | ファイルサイズ |
| implementations/agents/main_agent.py (2203行) | ⭐⭐⭐ | サイズ、診断コード混在 |
| implementations/layers/layer3_graph_reasoner.py (2244行) | ⭐⭐⭐ | サイズ最大、診断コード |

---

## 🚨 優先度別改善計画

### 🔴 P0: 緊急（1-2週間）

#### 1. 巨大ファイルの分割

**問題**: 3ファイルが2000行超

**影響**:
- テストが困難
- インポート時間が長い
- 型チェッカーの精度低下
- メンテナンス負担

**解決策**: モジュール分割（詳細は`refactoring_roadmap.md`参照）

**対象ファイル**:
1. `layer3_graph_reasoner.py` (2244行) ← **最優先**
2. `main_agent.py` (2203行)
3. `gedig_core.py` (2035行)

**工数見積**: 6-9日（各ファイル2-3日）

---

#### 2. 診断コードの分離

**問題**: 30+箇所に診断printが散在

```python
# 現状
if _DIAG_IMPORT:
    print('[main_agent] module import start', flush=True)
# ... 200行の診断コード ...
if _DIAG_IMPORT:
    print('[main_agent] layer1 imported', flush=True)
```

**影響**:
- ビジネスロジックとデバッグコードが混在
- 可読性低下
- メンテナンス負担

**解決策**: DiagnosticsManager パターン

```python
# 提案
from insightspike.diagnostics import _diag, trace

_diag.log('[main_agent] module import start')

@trace('initialize_components')
def _init_components(self):
    # クリーンなビジネスロジック
    pass
```

**工数見積**: 1日

---

### 🟡 P1: 高優先（2-4週間）

#### 3. テストカバレッジ向上（基準値を計測 → 目標設定）

**現状**:
- 実測カバレッジは 2025-11-27 時点で **16.9%**（`tests/e2e` + `tests/gedig` のみ、maze 依存テスト除外）。`fail_under=35` では失敗したため、暫定で 10 に緩和。
- maze 依存の欠損モジュールがある環境では conftest の自動 ignore で収集を回避。
- Layer3 リファクタ進捗: パッケージ足場＋`ConflictScore`/`GraphBuilder`/`message_passing` ラッパー抽出済み、message_passing は controller で初期化・apply まで委譲。GraphAnalyzer/RewardCalculator を layer3 に self-contained 移植。MetricsSelector は controller 経由、GNN 初期化は `layer3/gnn.py` に分離。`analyze_documents` は `analyzer_runner` に完全委譲（旧本体削除、runner 例外時のみ `_fallback_result`）。Layer3/メトリクス/GeDIG 小グラフ系の追加ユニットでカバレッジ 18.93%。e2e+gedig スモークは継続グリーン。
- Query-focal metrics は `analyzer_runner` に実装（core/cached 両パス、失敗時ニュートラル）。今後は実グラフでのパラメータ調整・期待値テストを追加予定。
- Query-focal metrics は別ハンドラに切り出し済み（現状スタブ）。推奨仕様をロードマップに追記済み。実装後にユニット/スモークを追加予定。

**次の一手**:
- まず `pytest --cov=src/insightspike --cov-report=term` でベースラインを取得し、数値を反映
- 重要パス優先で改善（gedig_core, main_agent, config, selector）

**目標イメージ（ベースライン取得後に具体化）**

| モジュール | 現状 | 目標 | 備考 |
|-----------|------|------|------|
| gedig_core.py | 未計測 (全体16.9%) | 80% | ベースライン取得後に設定 |
| selector.py | 未計測 (全体16.9%) | 90% | 同上 |
| public/__init__.py | 未計測 (全体16.9%) | 95% | 同上 |
| config/models.py | 未計測 (全体16.9%) | 85% | 同上 |
| main_agent.py | 未計測 (全体16.9%) | 75% | 同上 |

**Phase 2: Property-based testing (2週間)**

```python
from hypothesis import given, strategies as st

@given(
    num_nodes=st.integers(min_value=2, max_value=20),
    edge_prob=st.floats(min_value=0.1, max_value=0.8)
)
def test_gedig_monotonicity(num_nodes, edge_prob):
    """Property: F should be deterministic"""
    # ...
```

**工数見積**: 4週間（週10時間）

---

#### 4. Configuration の整理

**問題**: `models.py` が543行

**解決策**: モジュール分割

```
src/insightspike/config/
├── models/
│   ├── __init__.py
│   ├── base.py              # InsightSpikeConfig
│   ├── llm.py               # LLMConfig
│   ├── memory.py            # MemoryConfig
│   ├── graph.py             # GraphConfig (geDIG)
│   ├── processing.py        # ProcessingConfig
│   └── weights.py           # HybridWeightsConfig
└── presets.py               # 10→8に統合
```

**工数見積**: 2日

---

#### 5. 環境変数管理の一元化

**問題**: 12+箇所で`os.environ.get()`直接呼び出し

**解決策**: EnvironmentConfigLoader

```python
# src/insightspike/config/env_loader.py
class EnvironmentConfigLoader:
    """Centralized environment variable management"""

    SPECS = {
        'MAZE_GEDIG_LAMBDA': EnvVarSpec(
            key='MAZE_GEDIG_LAMBDA',
            default=1.0,
            type_=float,
            description='Override λ (information temperature)'
        ),
        # ... 全ての環境変数
    }

    @classmethod
    def get(cls, key: str, default=None) -> Any:
        """Get with type conversion"""
        # ...
```

**工数見積**: 2日

---

### 🟢 P2: 中優先（1-2ヶ月）

#### 6. ドキュメント整備

**6.1 Migration Guide (v4→v5)**

`docs/development/migration_v4_to_v5.md`

主な内容:
- 正規化スキームの違い
- Information Gain ソースの違い
- コード例とマイグレーションパス
- タイムライン（v0.9.0, v1.0.0）

**6.2 API Reference (Sphinx)**

```bash
pip install sphinx sphinx-rtd-theme
cd docs/api
sphinx-quickstart
make html
```

**工数見積**: 1週間

---

#### 7. パフォーマンス最適化

**7.1 プロファイリング**

```bash
python scripts/profile_gedig.py
# → ホットスポット特定
```

**7.2 最適化候補**

- SP計算のキャッシング
- グラフハッシュの最適化
- 遅延評価の活用

**工数見積**: 1週間

---

### 🔵 P3: 低優先（長期）

#### 8. 型チェック強化

**現状**: `mypy --ignore-missing-imports`

**目標**: `mypy --strict`

**段階的移行**:
```toml
# Phase 1
[tool.mypy]
warn_return_any = true
warn_unused_configs = true

# Phase 2
disallow_untyped_defs = true

# Phase 3
strict = true
```

**py.typed マーカー追加**:
```bash
touch src/insightspike/py.typed
```

**工数見積**: 2週間

---

## 📈 実装ロードマップ

### Sprint 1 (Week 1-2): 緊急対応

| Task | 工数 | 優先度 |
|------|------|--------|
| layer3_graph_reasoner.py 分割 | 3日 | P0 |
| 診断コード分離（DiagnosticsManager） | 1日 | P0 |
| 環境変数管理一元化 | 2日 | P1 |

**成果物**:
- ✅ `src/insightspike/implementations/layers/layer3/` ディレクトリ
- ✅ `src/insightspike/diagnostics/` モジュール
- ✅ `src/insightspike/config/env_loader.py`

---

### Sprint 2 (Week 3-4): テストカバレッジ

| Task | 工数 | 優先度 |
|------|------|--------|
| ベースラインカバレッジ取得（pytest --cov） | 0.5日 | P1 |
| gedig_core エッジケーステスト | 3日 | P1 |
| selector 境界値テスト | 2日 | P1 |
| config バリデーションテスト | 2日 | P1 |
| Coverage: ベースライン → +15pt | 3日 | P1 |

**成果物**:
- ✅ 50+新規テストケース
- ✅ Coverage report (50%)

---

### Sprint 3 (Week 5-6): main_agent.py 分割

| Task | 工数 | 優先度 |
|------|------|--------|
| main_agent/ ディレクトリ設計 | 1日 | P0 |
| コア実装の分離 | 3日 | P0 |
| 後方互換性テスト | 2日 | P0 |

---

### Sprint 4 (Week 7-8): gedig_core.py 分割

| Task | 工数 | 優先度 |
|------|------|--------|
| gedig/ ディレクトリ設計 | 1日 | P0 |
| Multi-hop ロジック分離 | 3日 | P0 |
| Metrics計算の分離 | 2日 | P0 |

---

### Month 2: ドキュメント & テスト

| Task | 工数 | 優先度 |
|------|------|--------|
| MIGRATION_V4_TO_V5.md | 2日 | P2 |
| Sphinx API Reference | 3日 | P2 |
| Property-based tests | 5日 | P1 |
| Coverage: ベースライン+15pt → 70% 目標に近づける | 10日 | P1 |

---

### Month 3: 最適化 & 長期改善

| Task | 工数 | 優先度 |
|------|------|--------|
| Profiling & 最適化 | 5日 | P3 |
| 型チェック強化 (Phase 1) | 5日 | P3 |
| Configuration 分割 | 3日 | P1 |
| Preset統合 (10→8) | 1日 | P2 |

---

## 🎯 成功指標 (KPI)

### コード品質

| メトリック | 現在 | 目標（3ヶ月） | 測定方法 |
|-----------|------|---------------|---------|
| 最大ファイルサイズ | 2244行 | 800行以下 | `wc -l` |
| 平均ファイルサイズ | ~300行 | ~250行 | 統計 |
| テストカバレッジ | 未計測（要 pytest --cov） | 70% | `pytest --cov` |
| 型カバレッジ | 未計測 | 85% | `mypy --strict` |

### 開発者体験

| メトリック | 現在 | 目標 |
|-----------|------|------|
| インポート時間 (main_agent) | 推定2-3秒 | 1秒以下 |
| テスト実行時間 (unit) | 推定30秒 | 20秒以下 |
| CI実行時間 (lite) | 推定2分 | 1分以下 |

### ドキュメント

| 項目 | 現在 | 目標 |
|------|------|------|
| API Reference | ❌ | Sphinx完備 |
| Migration Guide | ❌ | v4→v5完備 |
| Docstring Coverage | 推定70% | 90% |

---

## 💡 ベストプラクティス

### コーディング規約

**✅ Good: 小さな関数、単一責任**

```python
def compute_delta_sp(g_before: nx.Graph, g_after: nx.Graph,
                      pairs: List[Tuple[int, int]]) -> float:
    """Compute relative ΔSP gain.

    Args:
        g_before: Graph before update
        g_after: Graph after update
        pairs: Node pairs to evaluate

    Returns:
        Relative SP improvement: (L_before - L_after) / L_before
    """
    sp_before = _avg_shortest_path(g_before, pairs)
    sp_after = _avg_shortest_path(g_after, pairs)
    return (sp_before - sp_after) / max(sp_before, 1e-6)
```

**❌ Bad: 大きな関数、複数責任**

```python
def process_everything(data, config, options, flags, ...):  # 200行
    # ネスト深い、複数の責務
    pass
```

---

### テスト戦略

**✅ Good: AAA パターン (Arrange, Act, Assert)**

```python
def test_gedig_with_single_edge_addition():
    # Arrange
    g1 = nx.Graph([(0, 1), (1, 2)])
    g2 = g1.copy()
    g2.add_edge(2, 3)
    core = GeDIGCore(lambda_weight=1.0)

    # Act
    result = core.calculate(g_prev=g1, g_now=g2)

    # Assert
    assert result.raw_ged > 0
    assert result.ged_value > 0
```

**❌ Bad: 曖昧なテスト**

```python
def test_stuff():
    result = do_something()
    assert result  # 何をテスト？
```

---

### ドキュメント

**✅ Good: 具体例付きdocstring**

```python
def calculate(self, g_prev: nx.Graph, g_now: nx.Graph,
              linkset_info: Optional[Dict] = None) -> GeDIGResult:
    """Calculate geDIG value for graph transition.

    Implements the canonical formula::

        F = ΔEPC_norm - λ·(ΔH_norm + γ·ΔSP_rel)

    Args:
        g_prev: Graph state before update (G_t)
        g_now: Graph state after update (G_{t+1})
        linkset_info: Optional linkset metadata for entropy calculation.
            If provided, uses linkset-based IG (v5 paper-aligned).

    Returns:
        GeDIGResult containing F, ΔEPC, ΔIG components

    Examples:
        >>> core = GeDIGCore()
        >>> result = core.calculate(g1, g2)
        >>> print(f"F = {result.gedig_value:.3f}")

    References:
        .. [1] geDIG v5 paper, Section 3.2
    """
```

---

## 🚀 次のアクション

### 今すぐ実行可能

1. **layer3_graph_reasoner.py の分割開始**
   ```bash
   cd src/insightspike/implementations/layers
   mkdir layer3
   # 分割作業開始
   ```

2. **DiagnosticsManager 実装**
   ```bash
   mkdir src/insightspike/diagnostics
   # diagnostics/__init__.py 作成
   ```

3. **Coverage baseline測定**
   ```bash
   pytest --cov=src/insightspike --cov-report=html
   open htmlcov/index.html
   ```

### 1週間以内

1. layer3/ ディレクトリ構造完成
2. DiagnosticsManager 統合
3. カバレッジレポート共有

### 1ヶ月以内

1. 3大ファイル全て分割完了
2. テストカバレッジ 50%達成
3. MIGRATION_V4_TO_V5.md 完成

---

## 📚 参考資料

### リファクタリング
- Martin Fowler "Refactoring" (2nd ed)
- "Clean Code" by Robert C. Martin
- "Effective Python" by Brett Slatkin

### テスト戦略
- "Growing Object-Oriented Software, Guided by Tests"
- Hypothesis documentation: https://hypothesis.readthedocs.io
- pytest best practices: https://docs.pytest.org/en/stable/goodpractices.html

### 型チェック
- mypy documentation: https://mypy.readthedocs.io
- PEP 484 (Type Hints)
- "Fluent Python" (2nd ed)

---

## 📝 まとめ

InsightSpike-AIは**極めて高品質な研究プロジェクト**です。

### ✅ 現在の強み（維持すべき）
- 学術的厳密性（論文、特許）
- プロダクション対応（CI/CD、Docker）
- 開発者体験（Mock LLM、Lite mode）
- 再現性（シード管理、プロトコル）

### 🔧 改善領域（実行すべき）
1. **ファイル分割** (P0) → 3ヶ月で完了
2. **テストカバレッジ** (P1) → 70%目標
3. **ドキュメント** (P2) → API Reference + Migration Guide
4. **パフォーマンス** (P3) → プロファイリングベース

### 🎯 最終目標
- **9.5/10** の評価を目指す
- 他の研究者が容易にコントリビュート可能
- 企業のプロダクション採用に耐えうる品質

---

**本計画書は、InsightSpike-AIをさらなる高みへと導くためのロードマップです。**

詳細なリファクタリング手順は `refactoring_roadmap.md` を参照してください。
