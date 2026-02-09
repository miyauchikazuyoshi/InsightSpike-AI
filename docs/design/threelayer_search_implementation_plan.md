# 三層検索アーキテクチャ実装計画

**作成日**: 2026-02-09
**Status**: Draft（実行前）
**対象**: `experiments/maze/qhlib/` + `experiments/maze/run_experiment_query.py`
**根拠仕様**: `docs/research/thinking/memory_search_implementation_20260208.md`

---

## 1. 目的

現行の「SpatialGrid/L1プレフィルタ → 全候補距離計算 → 選択」フローを、
**不確実性に応じた三層検索**に置き換える。

```
計算量は不確実性に比例する。知っていることに計算を使わない。
```

| Layer | 判定 | 計算量 | 発火条件 |
|-------|------|--------|----------|
| Layer 0 | 完全一致（再訪判定） | O(1) | ハッシュヒット |
| Layer 1 | グラフ走査（attention参照） | O(degree) | 再訪 + attention > θ |
| Layer 2 | 記憶全ソート（現行ロジック相当） | O(N log N) | 新規 or L1候補不足 |

**期待効果:**
- 再訪時のLayer 2スキップ → 検索高速化
- attention重みによる候補品質向上
- β₁とattention閾値の連動 → 探索/活用の自動制御
- RAG等の大規模環境へのスケーラビリティ基盤

---

## 2. 現行アーキテクチャとの対応

### 2.1 現行フロー

```
クエリ到着
  ↓
[A] 観察候補生成（4方向、O(4)）           ← run_experiment_query.py:507-598
  ↓
[B] 記憶候補生成                          ← run_experiment_query.py:609-741
  ├─ SpatialGridIndex or L1 WeightedL2Index（プレフィルタ）
  └─ 全候補に weighted L2 距離計算
  ↓
[C] TwoThresholdSelector（θ_cand / θ_link） ← selector.py
  ↓
[D] build_ecand（候補エッジ組立）          ← edges.py
  ↓
[E] evaluate_multihop（マルチホップ評価）   ← evaluator.py
  ↓
[F] apply_commit_policy（グラフ更新）      ← commit.py
```

### 2.2 三層化後のフロー

```
クエリ到着: query_vector 生成
  ↓
[A] 観察候補生成（変更なし）
  ↓
[B'] 三層検索エンジン（[B]を置換）
  ├─ [Layer 0] VectorHashIndex.lookup() → 再訪判定 O(1)
  │     ├─ ヒット → [Layer 1]
  │     └─ ミス  → [Layer 2]
  │
  ├─ [Layer 1] AttentionGraphWalker.get_candidates() O(degree)
  │     ├─ 候補 ≥ min_layer1 → 候補確定、L2スキップ
  │     └─ 候補不足 → フォールスルー → [Layer 2]
  │
  └─ [Layer 2] 現行ロジック（SpatialGrid/L1 + weighted距離）
  ↓
[C] TwoThresholdSelector（変更なし、L2時のみ通過）
  ↓
[D'] build_ecand（L1候補 or L2候補を統合）
  ↓
[E] evaluate_multihop（変更なし）
  ↓
[F'] apply_commit_policy + AttentionManager
  ├─ 新エッジに attention=1.0 を付与
  └─ 全エッジ attention *= decay_rate（毎ステップ）
```

---

## 3. 新規ファイル

| ファイル | クラス | 行数見込み | 説明 |
|----------|--------|-----------|------|
| `qhlib/hash_index.py` | `VectorHashIndex` | ~80行 | Layer 0: 量子化ハッシュ再訪検出 |
| `qhlib/graph_walker.py` | `AttentionGraphWalker` | ~70行 | Layer 1: attention重み付きグラフ走査 |
| `qhlib/attention.py` | `AttentionManager` | ~60行 | エッジattentionライフサイクル管理 |
| `qhlib/search_engine.py` | `ThreeLayerSearchEngine` | ~100行 | 統合コントローラ（L0→L1→L2） |

**合計: ~310行の新規コード**

---

## 4. 既存ファイル変更

### 4.1 `qhlib/commit.py`（小変更）

**変更内容:** `add_edge()` 時に `attention=1.0` を付与

```python
# Phase 1: Hop-0 commit
if not graph.has_edge(current_query_node, dir_node):
    graph.add_edge(current_query_node, dir_node, attention=1.0)  # ← 追加

# Phase 2: Multi-hop commit
graph.add_edge(eu, ev, attention=1.0)  # ← 追加
```

**影響範囲:** 既存テストは `attention` 属性を参照しないため破壊なし。
**変更量:** ~5行

### 4.2 `qhlib/edges.py`（小変更）

**変更内容:** L1候補をecandに統合するパスを追加

```python
def build_ecand(
    *,
    # ... 既存パラメータ ...
    layer1_candidates: list[dict] | None = None,  # ← 新規パラメータ
) -> tuple[list, int, int]:
    # L1候補がある場合はそちらを優先
    if layer1_candidates:
        return _build_ecand_from_layer1(layer1_candidates, current_query_node, prev_graph)
    # 既存ロジック（L2フォールバック時）
    ...
```

**変更量:** ~20行

### 4.3 `qhlib/models.py`（小変更）

**変更内容:** StepRecord に三層検索診断フィールド追加

```python
@dataclass
class StepRecord:
    # ... 既存フィールド ...

    # 三層検索診断（デフォルト値あり、後方互換）
    search_layer_used: int = -1          # 0, 1, 2 (-1 = legacy)
    search_is_revisit: bool = False
    search_revisit_similarity: float = 0.0
    search_time_ms: float = 0.0
    search_l1_candidates: int = 0
```

**変更量:** ~10行

### 4.4 `run_experiment_query.py`（中変更）

**変更内容:** `--search-mode` フラグで三層検索を有効化

```python
# CLI引数追加
parser.add_argument("--search-mode", choices=["legacy", "threelayer"],
                    default="legacy")

# 初期化（search-mode=threelayer時のみ）
if config.search_mode == "threelayer":
    search_engine = ThreeLayerSearchEngine(
        hash_resolution=1.0 / config.maze_size,
        theta_revisit=0.95,
        theta_attention=config.theta_attention,
        weight_vector=WEIGHT_VECTOR,
        top_k=config.candidate_cap,
    )
    attention_mgr = AttentionManager(
        decay_rate=config.attention_decay,
        use_boost=config.attention_boost,
        theta=config.theta_attention,
    )

# ステップループ内（lines 609-741 の置換）
if config.search_mode == "threelayer":
    search_result = search_engine.search(query_vec, graph, memory_pool)
    if search_result.layer_used <= 1:
        # L0/L1ヒット: セレクタ・build_ecandをショートカット
        ecand = _ecand_from_search_result(search_result, current_query_node)
    else:
        # L2: 現行ロジックにフォールスルー
        ... (既存コード)
else:
    ... (既存コード、変更なし)

# コミット後: attention更新
if config.search_mode == "threelayer":
    attention_mgr.on_step(graph)           # 全エッジ減衰
    for eu, ev in committed_edges:
        attention_mgr.on_traverse(graph, eu, ev)  # 通過エッジ強化
    search_engine.register(current_query_node, query_vec)  # ハッシュ登録
```

**変更量:** ~80行（ただし既存コードの構造変更なし、if分岐の追加のみ）

### 4.5 `qhlib/models.py` — QueryHubConfig（小変更）

```python
@dataclass
class QueryHubConfig:
    # ... 既存フィールド ...
    search_mode: str = "legacy"
    theta_attention: float = 0.3
    attention_decay: float = 0.95
    attention_boost: float = 0.1
    attention_alpha: float = 0.5
    min_layer1_candidates: int = 2
```

**変更量:** ~10行

---

## 5. 実装フェーズ

### Phase 1: 独立コンポーネント実装（Day 1）

```
[1-1] qhlib/hash_index.py     — VectorHashIndex
[1-2] qhlib/graph_walker.py   — AttentionGraphWalker
[1-3] qhlib/attention.py      — AttentionManager
[1-4] qhlib/search_engine.py  — ThreeLayerSearchEngine
[1-5] 単体テスト
```

**依存関係:** なし。4ファイル並行作成可能。
**テスト方針:** 小規模グラフ（10ノード程度）でL0→L1→L2の各パスを検証。

### Phase 2: 統合（Day 2）

```
[2-1] commit.py に attention=1.0 付与追加
[2-2] edges.py に L1候補統合パス追加
[2-3] models.py に診断フィールド追加
[2-4] QueryHubConfig にパラメータ追加
[2-5] run_experiment_query.py に --search-mode threelayer 追加
```

**依存関係:** Phase 1 完了後。
**テスト方針:** `--search-mode legacy` で既存テスト全パスを確認後、`--search-mode threelayer` で9x9迷路スモークテスト。

### Phase 3: 検証実験（Day 3）

```
[3-1] 9x9 迷路: legacy vs threelayer 比較（3 seeds）
[3-2] 25x25 迷路: legacy vs threelayer 比較（5 seeds）
[3-3] dfs_loops 迷路: threelayer でβ₁/attention動態確認
```

**計測指標:**
- L1ヒット率（再訪時のL2スキップ率）
- 検索時間（L1 vs L2 の実測速度差）
- 候補品質（effective_score上位の候補が実際に良い選択か）
- β₁推移（探索↑ Sleep↓ のサイクルが出るか）

---

## 6. リスク管理

| リスク | 対策 |
|--------|------|
| 既存実験の再現性が壊れる | `--search-mode legacy` がデフォルト。既存コード変更なし |
| attention閾値のチューニング不足 | 初期値は仕様書準拠（θ=0.3, decay=0.95, boost=0.1）。実験で調整 |
| AttentionManager.on_step() のO(E)コスト | 25x25迷路のE=数百程度なら無視可能。大規模時はバッチ更新に移行 |
| L1候補品質が低い | min_layer1_candidates=2 をパラメータ化。不足時はL2フォールスルー |
| 切り戻し | `pre-betti1-baseline` ブランチが既存。追加で実装前にブランチ作成 |

---

## 7. 変更量サマリ

| カテゴリ | ファイル数 | 行数 |
|----------|-----------|------|
| 新規作成 | 4 | ~310行 |
| 既存変更 | 5 | ~125行 |
| テスト | 1-2 | ~100行 |
| **合計** | **10-11** | **~535行** |

---

## 8. 成功基準

1. `--search-mode legacy` で既存テスト全パス（回帰なし）
2. `--search-mode threelayer` で9x9/25x25スモークテスト通過
3. L1ヒット率 > 0%（再訪時にL2スキップが発生する）
4. threelayer の候補品質が legacy と同等以上（ゴール到達率で比較）
5. 25x25迷路でステップあたり検索時間が legacy 以下

---

## 9. テスト計画

### 9.1 単体テスト（Phase 1 完了時）

各コンポーネントを独立にテストする。テストファイル: `experiments/maze/test/test_threelayer.py`

#### VectorHashIndex

| # | テストケース | 入力 | 期待結果 |
|---|-------------|------|----------|
| H1 | 空インデックスからのlookup | 任意ベクトル | 空リスト |
| H2 | 登録→同一ベクトルlookup | vec=[0.5, 0.5] → lookup([0.5, 0.5]) | ヒット、sim ≈ 1.0 |
| H3 | 近傍ベクトルlookup | vec=[0.5, 0.5] → lookup([0.51, 0.49]) | resolution依存。resolution=0.05ならヒット |
| H4 | 遠方ベクトルlookup | vec=[0.5, 0.5] → lookup([0.9, 0.1]) | ミス（空リスト） |
| H5 | 複数登録→正しいノードID返却 | 3ノード登録、1つにヒット | 正しいnode_idのみ返却 |
| H6 | 量子化境界テスト | vec=[0.049, 0.0] → lookup([0.051, 0.0]) | lookup_with_neighbors使用時のみヒット |

#### AttentionGraphWalker

| # | テストケース | 入力 | 期待結果 |
|---|-------------|------|----------|
| W1 | 全エッジattention > θ | 3ノードグラフ、全att=1.0 | 全隣接ノード返却 |
| W2 | 全エッジattention < θ | 3ノードグラフ、全att=0.1 | 空リスト |
| W3 | 混在 | att=[1.0, 0.1, 0.5]、θ=0.3 | att>θの2エッジのみ |
| W4 | 再訪ノードがグラフに存在しない | 不在ノード指定 | 空リスト |
| W5 | effective_scoreソート順 | att=[0.9, 0.5]、sim=[0.3, 0.8] | score計算後に正しい降順 |

#### AttentionManager

| # | テストケース | 入力 | 期待結果 |
|---|-------------|------|----------|
| A1 | on_new_edge | 新エッジ追加 | attention=1.0, use_count=0 |
| A2 | on_step（減衰） | att=1.0, decay=0.95 | att=0.95 |
| A3 | on_step×10 | att=1.0, decay=0.95 | att≈0.5987 |
| A4 | on_traverse（強化） | att=0.5, boost=0.1 | att=0.6 |
| A5 | on_traverse上限 | att=0.95, boost=0.1 | att=1.0（min制限） |
| A6 | beta1計算 | 三角形グラフ、全att>θ | β₁=1 |
| A7 | beta1（att < θのエッジ除外） | 三角形、1辺att<θ | β₁=0（サイクル消滅） |

#### ThreeLayerSearchEngine

| # | テストケース | 入力 | 期待結果 |
|---|-------------|------|----------|
| E1 | 新規地点（L0ミス） | 未登録ベクトルでsearch | layer_used=2, is_revisit=False |
| E2 | 再訪（L0ヒット→L1十分） | 登録済みベクトル、隣接att>θが2以上 | layer_used=1, is_revisit=True |
| E3 | 再訪（L0ヒット→L1不足→L2） | 登録済みだが隣接att<θ | layer_used=2, is_revisit=True |
| E4 | register→lookup往復 | register後に同一ベクトルでsearch | L0ヒット |
| E5 | stats正確性 | L0→L1を3回、L2を2回 | stats['L1']=3, stats['L2']=2, L1_skip_rate=0.6 |

### 9.2 統合テスト（Phase 2 完了時）

テストファイル: `experiments/maze/test/test_threelayer_integration.py`

| # | テストケース | 手順 | 期待結果 |
|---|-------------|------|----------|
| I1 | 9x9迷路スモーク（threelayer） | `--maze-size 9 --max-steps 50 --seeds 1 --search-mode threelayer` | 完走、JSON出力にsearch_layer_usedあり |
| I2 | 9x9迷路スモーク（legacy） | `--maze-size 9 --max-steps 50 --seeds 1 --search-mode legacy` | 完走、search_layer_used=-1 |
| I3 | L1ヒット発生確認 | I1の出力JSONでsearch_layer_used=1のステップ数 | > 0（再訪が起きる迷路ならL1ヒットあり） |
| I4 | commit後のattention属性 | I1完了後のグラフエッジ | 全エッジにattention属性あり |
| I5 | attention減衰確認 | I1の途中ステップでattention < 1.0のエッジ | 存在する（on_stepで減衰） |

### 9.3 エンドツーエンドテスト（Phase 3）

テストスクリプト: `experiments/maze/test/run_threelayer_comparison.sh`

```bash
# 自動比較: legacy vs threelayer を同一seed/パラメータで実行し結果を比較
for MODE in legacy threelayer; do
  python run_experiment_query.py \
    --maze-size 25 --max-steps 500 --seeds 5 --seed-start 0 \
    --max-hops 10 --sp-cand-topk 5 \
    --search-mode $MODE \
    --output results/e2e_${MODE}.json
done
python compare_results.py results/e2e_legacy.json results/e2e_threelayer.json
```

比較スクリプト `compare_results.py` の出力項目:

| 指標 | 比較方法 | 許容基準 |
|------|----------|----------|
| ゴール到達率 | 成功seed数 | threelayer ≥ legacy |
| 平均ステップ数 | 成功seedの平均steps | threelayer ≤ legacy × 1.2 |
| 平均検索時間/step | time_ms_candidates の平均 | threelayer ≤ legacy |
| L1ヒット率 | search_layer_used=1 の割合 | > 0%（threelayerのみ） |
| 最終β₁ | betti1_series[-1] の平均 | 有意差なし |
| 最終ノード数 | node_count_series[-1] | 有意差なし |

---

## 10. 後方互換性検証計画

### 10.1 原則

**`--search-mode legacy`（デフォルト）では、変更前と完全に同一の動作をすること。**

三層検索の全コードは `if config.search_mode == "threelayer"` ガードの内側にあり、
legacyモードでは一切実行されない。

### 10.2 検証手順

#### Step 1: 変更前ベースライン取得

```bash
# 実装開始前に、現行mainでベースライン結果を保存
git stash  # or checkout pre-implementation branch
python run_experiment_query.py \
  --maze-size 9 --max-steps 100 --seeds 3 \
  --max-hops 10 --sp-cand-topk 5 \
  --vector-mode extended --sp-mode both \
  --output results/compat_baseline.json
```

#### Step 2: 変更後legacy結果取得

```bash
# 実装完了後、legacyモードで同一パラメータ実行
python run_experiment_query.py \
  --maze-size 9 --max-steps 100 --seeds 3 \
  --max-hops 10 --sp-cand-topk 5 \
  --vector-mode extended --sp-mode both \
  --search-mode legacy \
  --output results/compat_after_legacy.json
```

#### Step 3: バイナリ比較

```python
# validate_compatibility.py
import json, sys

def compare(baseline_path, after_path):
    with open(baseline_path) as f: base = json.load(f)
    with open(after_path) as f: after = json.load(f)

    checks = []

    for seed_idx in range(len(base['runs'])):
        b = base['runs'][seed_idx]
        a = after['runs'][seed_idx]

        # 同一seed → 同一結果であること
        checks.append(('success', b['success'] == a['success']))
        checks.append(('steps', b['steps'] == a['steps']))
        checks.append(('final_position', b.get('final_position') == a.get('final_position')))

        # 時系列長一致
        for key in ['g0_series', 'gmin_series', 'node_count_series',
                     'edge_count_series', 'betti1_series']:
            b_len = len(b.get(key, []))
            a_len = len(a.get(key, []))
            checks.append((f'{key}_len', b_len == a_len))

        # g0値一致（浮動小数点→丸め比較）
        for i, (bg, ag) in enumerate(zip(b.get('g0_series', []),
                                          a.get('g0_series', []))):
            if abs(bg - ag) > 1e-9:
                checks.append((f'g0[{i}]', False))
                break
        else:
            checks.append(('g0_values', True))

    passed = sum(1 for _, ok in checks if ok)
    failed = [(name, ok) for name, ok in checks if not ok]
    print(f"Compatibility: {passed}/{len(checks)} passed")
    if failed:
        print("FAILURES:")
        for name, _ in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("ALL PASSED — 後方互換性確認完了")

if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
```

#### Step 4: 既存テストスイート実行

```bash
# geDIG単体テスト（40テスト）
.venv_ci313/bin/python -m pytest tests/algorithms/gedig/ -v

# 迷路テスト
.venv_ci313/bin/python -m pytest experiments/maze/tests/ -v

# β₁スモークテスト
bash experiments/maze/test/run_betti1_smoke.sh
.venv/bin/python3 experiments/maze/test/validate_betti1.py
```

**合格基準:** 変更前と同一のテスト結果（既存の事前失敗12件は許容）

### 10.3 変更影響マトリクス

| 変更ファイル | legacy影響 | 理由 |
|-------------|-----------|------|
| `hash_index.py` (新規) | なし | legacyでは import されない |
| `graph_walker.py` (新規) | なし | legacyでは import されない |
| `attention.py` (新規) | なし | legacyでは import されない |
| `search_engine.py` (新規) | なし | legacyでは import されない |
| `commit.py` | **要検証** | attention=1.0 が既存エッジに影響しないか |
| `edges.py` | なし | layer1_candidates=None がデフォルト → 既存パス |
| `models.py` | なし | 新フィールドは全てデフォルト値あり |
| `run_experiment_query.py` | **要検証** | if分岐のみだがインポート追加あり |

### 10.4 commit.py の互換性詳細

`attention=1.0` をエッジに付与する変更は、legacyモードでも実行される（commit.pyは共通パス）。

**影響分析:**
- `graph.add_edge(u, v, attention=1.0)` → NetworkXは追加属性を無視する設計
- 既存コードで `graph[u][v]` を参照する箇所は `edge_type` のみ → attentionを参照しない
- evaluator.py の SP 計算は `weight` パラメータのみ使用 → attention は無関係

**結論:** 安全。ただし Step 3 のバイナリ比較で確認。

もしcommit.pyの変更が互換性リスクと判断される場合の代替案:
```python
# commit.py を変更せず、run_experiment_query.py で事後付与
if config.search_mode == "threelayer":
    for eu, ev in committed_edges:
        if graph.has_edge(eu, ev) and 'attention' not in graph[eu][ev]:
            graph[eu][ev]['attention'] = 1.0
```

---

## 11. 詳細実行計画

### Day 0: 準備（実装開始前）

```
[0-1] 切り戻しブランチ作成
      git checkout -b pre-threelayer-baseline
      git push origin pre-threelayer-baseline
      git checkout main

[0-2] 未コミット変更の整理
      β₁バグ修正 + betti1_comparison 実験スクリプトのコミット

[0-3] ベースライン結果取得
      9x9 × 3seeds / 25x25 × 5seeds を実行し結果保存
      → results/compat_baseline_9x9.json
      → results/compat_baseline_25x25.json
```

### Day 1: Phase 1 — 独立コンポーネント

```
[1-1] qhlib/hash_index.py 作成
      - VectorHashIndex クラス
      - _quantize, add, lookup, lookup_with_neighbors, size

[1-2] qhlib/graph_walker.py 作成
      - AttentionGraphWalker クラス
      - get_candidates, _weighted_similarity

[1-3] qhlib/attention.py 作成
      - AttentionManager クラス
      - on_new_edge, on_step, on_traverse, on_ag_fire, beta1

[1-4] qhlib/search_engine.py 作成
      - SearchResult dataclass
      - ThreeLayerSearchEngine クラス
      - search, register, get_stats

[1-5] 単体テスト作成・実行
      - test/test_threelayer.py
      - テストケース H1-H6, W1-W5, A1-A7, E1-E5（Section 9.1）
      - 全パス確認

[1-6] コミット
      "feat(qhlib): add three-layer search components"
```

### Day 2: Phase 2 — 統合

```
[2-1] models.py 変更
      - StepRecord に search_layer_used 等5フィールド追加
      - QueryHubConfig に search_mode 等6フィールド追加

[2-2] commit.py 変更
      - add_edge に attention=1.0 追加（2箇所）

[2-3] edges.py 変更
      - build_ecand に layer1_candidates パラメータ追加
      - _build_ecand_from_layer1 ヘルパー追加

[2-4] run_experiment_query.py 変更
      - CLI引数 --search-mode 追加
      - ThreeLayerSearchEngine 初期化
      - ステップループ内に三層検索パス追加
      - コミット後のattention更新追加

[2-5] 後方互換性検証
      - legacyモードでベースライン比較（Section 10.2 Step 2-3）
      - 既存テストスイート実行（Section 10.2 Step 4）

[2-6] 統合テスト実行
      - テストケース I1-I5（Section 9.2）

[2-7] コミット
      "feat(maze): integrate three-layer search with --search-mode flag"
```

### Day 3: Phase 3 — 検証実験

```
[3-1] 9x9 迷路比較実験
      - legacy × 3seeds vs threelayer × 3seeds
      - compare_results.py で比較

[3-2] 25x25 迷路比較実験
      - legacy × 5seeds vs threelayer × 5seeds
      - L1ヒット率、検索時間、到達率を比較

[3-3] dfs_loops 25x25 迷路
      - threelayer × 3seeds
      - β₁/attention動態の可視化

[3-4] 結果分析レポート作成
      - experiments/maze/threelayer_comparison/REPORT.md

[3-5] コミット
      "experiment(maze): three-layer search comparison results"
```

---

## 12. ドキュメント定義変更のゲート条件

三層検索とβ₁の導入に伴い、論文・README等の定義記述をいつ更新するかの判断基準。
**早すぎる定義変更は混乱を招く。エビデンスに基づいて段階的に更新する。**

### 12.1 ゲート一覧

```
今 ──── Gate 1 ──── Gate 2 ──── Gate 3 ──── Gate 4
          │            │            │            │
     実装完了      L1効果実証   β₁/ASP相関   相転移実証
          │            │            │            │
      変更なし    README軽微   論文F定義     論文理論章
```

### Gate 1: 実装完了（Phase 2 終了時）

**条件:** 三層検索が `--search-mode threelayer` で動作する
**ドキュメント変更:** なし
**理由:** 「動く」だけでは定義変更の根拠にならない

### Gate 2: L1効果実証（Phase 3 終了時）

**条件（全て満たすこと）:**
- L1ヒット率 > 20%（再訪時の2割以上でL2スキップ）
- threelayer のゴール到達率 ≥ legacy
- 検索時間の実測で threelayer ≤ legacy

**変更対象:**
| ドキュメント | 変更内容 | 変更レベル |
|-------------|---------|-----------|
| README.md | Architecture節に「三層検索モード（オプション）」追記 | 軽微 |
| CHANGELOG | 機能追加として記載 | 軽微 |

**変更しないもの:**
- 論文のF分解定義（F = GED + IG + λ·SP）
- geDIGの数学的定義

### Gate 3: β₁/ASP相関判明（追加実験後）

**前提:** SP計算を有効にした60seed以上の実験が必要（現状delta_sp_seriesが全ゼロ）

**条件と分岐:**

| Spearman |ρ| | 判定 | 論文への影響 |
|-----------|------|----------------|
| > 0.7 | β₁でSP置換可能 | F = GED + IG + λ·β₁ に定義変更 |
| 0.4 - 0.7 | β₁はSPの部分情報 | F = GED + IG + λ·SP + μ·β₁（項追加） |
| < 0.4 | β₁はSPと独立 | β₁を新しい独立項として追加。SPは残す |

**変更対象:**
| ドキュメント | 変更内容 |
|-------------|---------|
| 論文（JSAI等） | F分解の定義式を更新 |
| README.md | Mathematical Framework節を更新 |
| `docs/research/` | β₁採用の理論的根拠文書を追加 |
| `src/` コード | gedig_core.py のcalculate()でβ₁をSP項に反映 |

### Gate 4: 相転移実証（将来）

**条件:**
- attention閾値θの掃引で β₁(θ) の相転移点が迷路で検出できる
- 相転移点が探索/活用モードの切り替えと一致する
- `beta1_navigation_routing_20260208.md` の理論が迷路で再現

**変更対象:**
| ドキュメント | 変更内容 |
|-------------|---------|
| 論文（理論セクション） | 「β₁相転移による自動階層化」の章を追加 |
| 特許出願書 | 迷路実施例 + 道路ネットワーク実施例を添付 |
| README.md | Theoretical Foundation節を追加 |

### 12.2 現在の到達予定

本計画書の Day 1-3 で到達するのは **Gate 2 まで**。

Gate 3 に到達するための追加作業:
1. SP計算を有効にした実験設定の準備（`--sp-mode` の調整）
2. 60seed × 25x25 実験の実行（推定所要時間: ~2時間）
3. `analyze_betti_sp_correlation.py`（betti1_engineering_spec.md Part C）の実行

Gate 4 は Gate 3 の結果次第。β₁がSPと高相関なら相転移理論が自然に導かれる。
低相関なら別のアプローチが必要。

---

## 13. 将来展望

- **β₁連動:** attention閾値をβ₁に応じて動的調整（β₁高→θ下げ→探索モード）
- **Sleep統合:** Sleep propagation後にattention再活性化（AG発火）
- **RAG拡張:** VectorHashIndex を LSH に置換、FullMemorySearch を HNSW に置換
- **Transformer統合:** 推論時の hidden state 検索に同一アーキテクチャを適用
