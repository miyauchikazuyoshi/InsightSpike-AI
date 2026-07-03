# Three-Layer Search Architecture (L0 / L1 / L2)

**Status**: implemented & tested (2026-02) — used by the maze line via `--search-mode threelayer`
**Implementation**: `experiments/maze/qhlib/` — `hash_index.py` / `graph_walker.py` / `attention.py` / `search_engine.py` (432 lines total)
**Tests**: `experiments/maze/test/test_threelayer.py` (23 unit tests: H1-H6, W1-W5, A1-A7, E1-E5) + `test/validate_compatibility.py` (24 backward-compat checks)
**This document**: written 2026-07-03 from the implementation.
**先行設計メモ**: [docs/research/thinking/memory_search_implementation_20260208.md](../research/thinking/memory_search_implementation_20260208.md)
(2026-02-08 — Layer0/1/2 を β₁ の大小に対応づけた原典。本書は実装後の正確な仕様、原典は設計動機を保存)

---

## 1. 目的

迷路エージェントの候補検索は、素朴には毎ステップ「全メモリを類似度ソート」(L2、O(N log N)) になり、
グラフが成長するほど遅くなる。しかし**再訪時**(すでに知っている場所に戻ったとき)は、
答えの候補はグラフ上の近傍にほぼ確実に存在する。三層検索はこの局所性を段階的に搾り取る:

| 層 | 機構 | 計算量 | 役割 |
|---|---|---|---|
| **L0** | `VectorHashIndex` — 量子化ベクトルのハッシュ照合 | **O(1)** | 再訪検出(「ここ、来たことがある?」) |
| **L1** | `AttentionGraphWalker` — 再訪ノードからの注意重み付き 1-hop 歩行 | **O(degree)** | 局所候補生成(「前回ここから何が見えた?」) |
| **L2** | 従来のフルメモリソート(呼び出し側 = legacy パス) | O(N log N) | フォールバック(新規領域・候補不足時) |

生物学的対応(比喩): L0 = 場所細胞的なパターン照合、L1 = 注意による近傍活性化、L2 = 全探索。

## 2. カスケードの正確なロジック (`ThreeLayerSearchEngine.search`)

```
query_vector ──► L0: hash_index.lookup(query, θ_revisit=0.95)
                    │
        ┌───────────┴───────────┐
     再訪あり                 再訪なし
        │                       │
        ▼                       │
   L1: graph_walker.get_candidates(graph, revisit, query, weight_vector)
        │                       │
   len(cands) ≥ min_layer1(=2)? │
        │           │           │
       YES          NO ─────────┤
        │                       ▼
        ▼               L2: SearchResult(candidates=[], layer_used=2)
   layer_used=1              ── 呼び出し側が従来のフルソートを実行
   候補を返す
```

重要: **L2 は「空の候補リスト」を返すだけ**で、フル検索そのものはエンジンの外(呼び出し側の
legacy パス)が実行する。エンジンは統計(`L0/L1/L2` カウント、`L1_skip_rate`)を持ち、
`get_stats()` で層の使用率を観測できる。

## 3. モジュール別 API

### 3.1 `VectorHashIndex` (hash_index.py, 89 行)
- `__init__(resolution=0.05)` — ベクトルを `resolution` で量子化してタプルキー化
- `add(node_id, raw_vector)` — **commit 後に登録**(`ThreeLayerSearchEngine.register` 経由)
- `lookup(vec, threshold)` — 量子化バケット内で類似度 ≥ threshold のノードを返す(再訪検出)
- `lookup_with_neighbors(...)` — 隣接バケットも見る変種

### 3.2 `AttentionGraphWalker` (graph_walker.py, 138 行)
- `__init__(theta=0.3, alpha=0.5, dg_gate_tau, tau_dg_3att, tau_reward, score_mode)`
- `get_candidates(graph, revisit, query_vector, weight_vector)` — 再訪ノードを起点に、
  attention ≥ θ のエッジを辿って候補を収集。スコアは重み付き類似度と attention の合成
  (`alpha` で配分、`score_mode` で legacy 互換式に切替可)

### 3.3 `AttentionManager` (attention.py, 78 行) — エッジ属性 `attention` のライフサイクル
- `on_new_edge(G, u, v)` — 新規エッジに **attention = 1.0 を初期化**
- `on_step(G)` — 全エッジを **×0.95 減衰**(`--attention-decay`)
- `on_traverse(G, u, v)` — 通過したエッジを強化(`--attention-boost`)
- `on_ag_fire(G, node)` — AG 発火時、当該ノード周辺の attention を返す(診断)
- `beta1(G, theta)` — **attention ≥ θ のエッジのみで β₁ を計算**(注意で刈った活性サブグラフの位相)

### 3.4 `ThreeLayerSearchEngine` (search_engine.py, 127 行)
- 主要パラメータ: `theta_revisit=0.95` / `theta_attention=0.3` / `attention_alpha=0.5` /
  `min_layer1_candidates=2` / `top_k=32` / `weight_vector`(必須)
- `search(query_vector, graph, memory_pool=None) -> SearchResult(candidates, layer_used, is_revisit, revisit_similarity, search_time_ms)`
- `register(node_id, raw_vector)` / `get_stats()`

## 4. 統合ポイント(maze ライン)

- CLI: `--search-mode legacy|threelayer`(**デフォルト legacy** — 挙動不変が既定)、
  `--theta-attention`, `--attention-decay`, `--attention-boost`, `--attention-alpha`,
  `--min-layer1-candidates`
- `run_experiment_query.py`: エンジン初期化(実行冒頭)/ search 呼び出し(候補生成部)/
  attention ライフサイクル(ステップループ内)の 3 箇所に接続

## 5. 既知の罠(実装時に踏んだもの — 回帰させない)

1. **Attention ライフサイクルの順序**: 新規エッジの `attention=1.0` 初期化は、そのステップの
   `on_step()` 減衰の**前**に行うこと(逆にすると初期値が即減衰する)
2. `numpy_array or other_array` は ValueError — 明示的な `is None` チェックを使う
3. `append_record()` はステップログ dict を手動構築している — `StepRecord` に新フィールドを
   足したら**明示的に追記**しないとログに出ない
4. **9×9 迷路は L1 の検証に小さすぎる**(~28 歩で解けて再訪が少ない)。L1 の挙動を見るなら 15×15 以上

## 6. 状態と今後

- stage-1 PoC の一部として動作確認済み(挙動互換 24/24)。層使用率の系統的なベンチマーク
  (L1 発動率 vs 迷路サイズ、速度プロファイル)は未実施
- `attention.beta1()`(注意付き β₁)は v7 の β₁ 一般化(plan.md Phase 0)との接続候補
