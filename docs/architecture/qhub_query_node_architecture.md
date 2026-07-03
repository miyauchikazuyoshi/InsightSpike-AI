# QHub Query-Node Architecture(クエリハブ型グラフ体系)

**Status**: implemented — maze ラインの現行グラフ表現(`graph_mode: query_hub`)
**Implementation**: `experiments/maze/qhlib/`(`utils.py` のノード ID / `models.py` の `QueryHubConfig`・`StepRecord`)+ `run_experiment_query.py`
**一次情報源**: 運用詳細(CLI・評価規則・スナップショット戦略)は [experiments/maze/README.md](../../experiments/maze/README.md) の「ノード体系」以下の各節。本書はその**設計の骨格**を docs/ 側から参照可能にする要約であり、詳細を重複記載しない。
**This document**: 2026-07-03 作成(鮮度監査で「独立設計文書なし」とされたギャップの解消)

---

## 1. 設計思想

従来(center_hub)は「中心ノード」に候補をぶら下げる静的な形だった。QHub は**各ステップの観測を
クエリノードとして生成し、それをハブに候補(方位ノード)を仮配線 → F 評価 → 選択的コミット**する。
グラフは「歩いた軌跡と、その場その場で検討した選択肢」の履歴として成長する。

## 2. ノード 3 種(ID 形式は実装が正)

| 種類 | ID(実装) | README 表記 | 主要属性 |
|---|---|---|---|
| クエリノード | `(row, col, -1)` — 第 3 要素は `QUERY_MARKER = -1`(`qhlib/utils.py`) | `(row, col, "query")` | `visit_count`, `abs_vector`, `anchor_positions` |
| 方位ノード | `(row, col, dir)`、`dir ∈ {0,1,2,3}` | 同じ | `target_position`, `anchor_position`, `reward`(wake 中に記録・上書き), `propagated`(sleep が付与) |
| メモリノード | リプレイからロード | 同じ | クエリハブ下に再マッピング |

10D ベクトル(`--vector-mode extended`)では、方位ノードの `abs_vector[8] = reward`(**wake 中に**同期)、
`abs_vector[9] = tanh(propagated)`(**sleep が**同期)。replay sleep の Q(s,a) は方位ノード
`(r, c, a)` と**1 対 1 対応**する(state=(r,c), action=a)。

## 3. エッジ 3 種

1. **クエリ ↔ 方位**: ステップ開始時に候補と同時に仮配線。`S_link`(低閾値)入りは確定、
   `S_cand` のみは候補タグ
2. **方位 ↔ 次クエリ**: 行動確定後、移動先クエリノードと接続
3. **クエリ ↔ クエリ**: 時間的連結(移動前後)

## 4. ステップの実行シーケンス(8 段階)

```
T0 prev_graph 退避 → 候補生成(S_link Top-L、空なら未訪問方向優先フォールバック)
→ 事前評価(pre-eval、診断のみ) → 評価用グラフ構築(0-hop 実配線)
→ 本評価(g(h) = ΔGED − λ(ΔH + γΔSP)、h=0..H) → DG ポリシー適用(実コミット、予算内)
→ 永続化(SQLite/DS 差分) → スナップショット整形
```

判定は二段: **AG**(g₀ ≥ θ_AG で曖昧さ検知)→ **DG**(g_min < θ_DG でショートカット確定)。
発火状況は `StepRecord` と DG Ledger(`--dg-ledger-log`)に記録される。

## 5. 訪問回数と報酬の記録規則(アブレーション設計に効いた事実)

- `visit_counts[(row,col)]` はセル単位でインクリメントし、ノードの `visit_count` はステップ末に同期
- 方位ノードの `reward` は**上書き方式**(最後の通過イベントが勝つ)。実装値:
  goal +1.0 / novel +0.2 / revisit −0.4 / deadend・blocked −1.0
- warmup(Wake1)は誘導・伝播と無関係の初回エピソード → **同一シードの warmup は決定的に同一**
  (paired 実験設計の基盤。sleep ablation v1–v3 の操作チェックが毎回 paired 差 0 を要求できる根拠)

## 6. 関連文書

- 検索カスケード: [threelayer_search_architecture.md](threelayer_search_architecture.md)
- sleep(値再処理): [graph_persistent_dg/SPEC.md](../../experiments/maze/graph_persistent_dg/SPEC.md) +
  意味論テスト `experiments/maze/test/test_sleep_propagate_semantics.py`
- 実験の勝敗: [docs/prereg/README.md](../prereg/README.md)
