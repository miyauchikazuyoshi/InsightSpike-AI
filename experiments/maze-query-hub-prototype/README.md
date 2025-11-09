# Maze Query-Hub Prototype

このディレクトリでは、既存の `maze-fixed-kstar` 実験を参考にしつつ、中心ノード（`CENTER_DIR`）を廃止し「クエリノード」をハブとして利用する新しいグラフ表現のプロトタイプを進める。

## 目的
- 方位ノードが常に中心ノードに束ねられる構造を見直し、クエリ自体をハブノードとして永続化する。
- ΔGED/ΔIG の評価をクエリノードを起点としたサブグラフで行い、論文 5 章の設計に近づける。
- 既存実装と並行で開発し、挙動を比較できるようにする。

## 仕様概略
### ノード体系
| 種類 | ID 例 | 属性のポイント |
| ---- | ----- | --------------- |
| クエリノード | `(row, col, "query")` | `visit_count`, `abs_vector`, `anchor_positions` を保持。ステップ冒頭で生成し、ステップ終了後に訪問回数を同期する。 |
| 方位ノード | `(row, col, dir)` (`dir ∈ {0,1,2,3}`) | 既存と同様。`target_position` は移動先セル、`anchor_position` はアンカーセル。 |
| メモリノード | 既存のリプレイからロード | クエリハブ下に接続されるよう再マッピングする。 |

### エッジ構造
1. クエリノード ↔ 方位ノード: ステップ開始時に候補生成と同時に仮配線。類似度・距離に基づき、以下のルールで保持する。
   - `S_link` に入った候補は確定エッジとして保持。
   - `S_cand` のみ通過したノードは候補エッジとしてタグ付けし、ログで別扱いする（グラフ上では一旦残す）。
2. 方位ノード ↔ 「次クエリ」: 行動確定後、移動先で生成したクエリノードと方位ノードを接続。
3. クエリノード同士: 時間的連結を明示するため、移動前後のクエリノードにもエッジを張る（従来の中心ノードエッジに相当）。

### 訪問回数管理
- `visit_counts[(row, col)]` を従来通りインクリメント。
- クエリノード/方位ノードの `visit_count` はステップ終了後に `visit_counts` を参照して同期。
- クエリ生成直後は `visit_count = 0` とし、観測時点での未訪問状態を明確化。

### geDIG 評価
- 0-hop は S_link（低しきい値）から Top-L（既定1）を暫定配線したグラフで評価し、分母（Cmax/K）は S_cand（高しきい値）の台に固定する（安定化）。
- DG（`g_min < θ_DG`）成立時のみ、S_cand から ΔSP の限界寄与が大きい枝を貪欲に選び、コミット（`--commit-budget` 本まで）。
- `--theta-ag/--theta-dg/--top-link/--commit-budget/--commit-from` で制御可能。g₀/g_min と AG/DG の発火状況は `StepRecord` に記録。
  - 0‑hopで S_link を全本自動配線したい場合は `--link-autowire-all` を付与（Top‑Lを無視してベース配線）。
  - S_link が空のとき、強制候補（未探索優先など）をベースに使いたい場合は `--link-forced-as-base`（既定ON）。無効化は `--no-link-forced-as-base`。
  - 行き止まり/バックトラック時にマルチホップをスキップする `--skip-mh-on-deadend` は、既定OFF（推奨: 無効のまま）。

#### 実行シーケンス（Phase 4, 2フェーズ）
1) T0: prev_graph 退避（前ステップまでのグラフをコピー）
2) 候補生成: `S_link` Top‑L を選択（空なら「未訪問direction優先→なければ全体最近傍direction」へフォールバック）
3) 事前評価（pre‑eval, IG/SPのみ）: graph_pre に対し `core.calculate(prev_graph, graph_pre, …)` を実施（診断用, g0/gminに未採用）
4) 評価用グラフ構築（eval_after）: 0‑hop（Q↔dir）を実配線した `graph_eval` を作成
5) 本評価（hopシリーズ）: `prev_graph` vs `graph_eval_h`（h=0..H）で g(h) を算出（ΔGED − λ·(ΔH + γ·ΔSP））
6) DGポリシー適用（実コミット）: best_hop までの仮想採用セットを実コミット（閾値/always/never + 予算）
7) 永続化（DS）: SQLite/ローカルDSにノード・エッジ差分を保存（Strict DSでは timeline は評価に混入させない）
8) HTML用スナップショット整形: snapshot-level に応じて最小メタ/診断をsteps.jsonへ、または present 層でDSから再構成

### ロギング & 可視化
- `StepRecord` にクエリノード ID（例: `query_node: [row, col]`）を出力。
- HTML ビューアではクエリノードを中心ハブとして描画し、`cand` タグは `target_position` 基準で表示する。
- 旧実装との比較のため、`graph_mode`（既存実装: `center_hub`, 本プロトタイプ: `query_hub`）で実行モードを切り替え可能にする。

#### 長時間実行対策（チェックポイント）
- `--checkpoint-interval N` を指定すると、N ステップごとに `--step-log` へ部分 JSON を原子的に書き出します（タイムアウト時の保全用）。

### スナップショット戦略（速度と再現性の両立）

2つのモードを用意しています。用途に応じて選択してください。

- A) 実行軽量化優先（超軽量ステップ）
  - フラグ: `--steps-ultra-light`（steps.json から重い配列を完全に省略） + `--no-post-sp-diagnostics`
  - 併用推奨: SQLite永続化（`--persist-graph-sqlite ... --persist-namespace ...`）
  - HTML生成: `build_reports.py --light-steps --sqlite <db> --namespace <ns>`（必要に応じて DS から再構成）

- B) 可視化重視（最小スナップショット）
  - フラグ: `--snapshot-level minimal`（重い配列を最小限に絞る）
  - `--post_sp_diagnostics` は既定ON（必要に応じて `--no-post-sp-diagnostics` でOFF）

補足
- DS再構成: `build_reports.py` は `--sqlite` と `--present-mode`（`strict|relaxed`）で SQLite から段階的にグラフ断片を復元できます（dotted importが難しい環境でも動くようにパス読み込みに対応済み）。

### 論文（v4）向けの補助ツール

- 迷路スナップショットの明示保存
  - 実行時に `--maze-snapshot-out docs/paper/data/maze_51x51.json` を指定すると、レイアウト/開始/目標/サイズをJSONで保存します。

- 指標エクスポート（LaTeX取り込み用）
  - `tools/export_paper_maze.py` … summary/steps から主要メトリクスを抽出して JSON/CSV 出力
  - 例:
    - `python experiments/maze-query-hub-prototype/tools/export_paper_maze.py \
       --summary experiments/maze-query-hub-prototype/results/_51x51_s200_l3_summary.json \
       --steps   experiments/maze-query-hub-prototype/results/_51x51_s200_l3_steps.json \
       --out-json docs/paper/data/maze_51x51_l3_s200.json \
       --out-csv  docs/paper/data/maze_51x51_l3_s200.csv \
       --compression-base mem`

- ベースライン比較（A/B差分）
  - `baselines/compare_runs.py` … 2つの（summary, steps）を比較し、差分をJSONにまとめます。
  - 例:
    - `python experiments/maze-query-hub-prototype/baselines/compare_runs.py \
       --base-summary A_summary.json --base-steps A_steps.json \
       --var-summary  B_summary.json --var-steps  B_steps.json \
       --out docs/paper/data/ab_diff.json`

### 実行モードの違い（L3-only vs A/B）

- L3-only（最終結果向け）
  - 目的: 論文の最終値を一本化（Evaluatorを使わず、L3のmulti-hop/core SPで評価）
  - 特徴: 速度最優先（`--sp-cache-mode core`、union無し）。per-hop記録はせず hop0/最終値を出力。
  - 実行: `tools/run_paper_l3_only.py` を利用（15/25/51のサイズ別で seeds を設定）

- A/B（Evaluator vs L3 の整合確認、診断）
  - 目的: hop0/per-hop まで設計が揃っていることを確認（paper preset）。
  - 特徴: `--sp-scope union` を付けると per-hop 記録のため evaluator にフォールバック→計算が重くなる（平均 3倍程度）。
  - 実行: `tools/run_paper_comparison.py` または `tools/run_paper_grid.py`（任意）。最終報告には含めない。

- おすすめ運用
  - 最終値は L3-only（union 無し, `--sp_hop_expand=0` が基本）
  - 診断時だけ A/B を回して per-hop を確認（必要なら `--union --force-per-hop`）

### 付録: 実行スクリプト一覧

- L3-only: `tools/run_paper_l3_only.py`
- 比較（A/B）: `tools/run_paper_comparison.py`
- グリッド（複数サイズ一括）: `tools/run_paper_grid.py`
- 51×51 長時間（s=250, seeds=40）: `tools/run_51x51_s250_seeds40.sh`

#### ステップログ（steps.json）の構造（要点）
各ステップのスナップショットに加えて、差分レジャーを保持します。

- スナップショット
  - `graph_nodes`: その時点の全ノード（`[r,c,dir]`）
  - `graph_edges`: その時点の全エッジ（`[[nodeA],[nodeB]]`）
  - `timeline_edges`: 可視化専用の時系列エッジ（graphへは未挿入; 既定OFFで Q↔Q は非表示）

- 差分（このステップでの追加分のみ）
  - `committed_only_edges`: 追加エッジ一覧（計算対象）
  - `committed_only_nodes`: 追加ノード一覧
  - `committed_only_edges_meta`: `[{ nodes, forced, stage, step }]`
    - `stage`: `"base"`（0-hop暫定配線） or `"dg"`（貪欲コミット）
  - `committed_only_nodes_meta`: `[{ node, node_type, direction, source, visit_count, anchor_positions, target_position, birth_step, step }]`

備考:
- 評価は `prev_graph` を基準に行われるため、時系列エッジは評価に混入しません（`--timeline-to-graph`/`--add-next-q` を付けない限り）。

### 永続化（任意）
- 差分レジャーをSQLiteに永続化できます。
  - CLI: `--persist-graph-sqlite path/to/db.sqlite`（`--persist-namespace` で名前空間指定可）
  - スキーマ: `graph_nodes(id, namespace, node_type, attributes)` / `graph_edges(namespace, source_id, target_id, edge_type, attributes)`
  - 追加するのは「コミット差分（committed_only_*）」のみ（timelineは対象外）
- ノードIDは `"r,c,dir"` 形式の文字列で保存します。

### Strict DS（可視化）と Present 層
- Strict DS: ビューアは「DS（保存済み）」のノード/エッジのみで描画（候補やtimeline等のオーバレイ無効化）。
- present 層: DSから各ステップのグラフ断片を再構成し、HTMLが参照する JSON を整形。
  - 実装: `experiments/maze-query-hub-prototype/qhlib/present.py`
  - 関数: `load_ds_graph/ load_ds_graph_upto_step/ reconstruct_records(mode='strict'|'relaxed')`
  - Strict では timeline/candidate の混入を避け、浮遊や次Qリークを抑止。

## ビルドと生成

### 実行（例）
```
.venv/bin/python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 40 --seeds 1 \
  --lambda-weight 0.5 --max-hops 15 \
  --norm-base link --linkset-mode \
  --theta-ag 0.2 --theta-dg 0.15 --top-link 1 \
  --output experiments/maze-query-hub-prototype/results/_phase4c/summary.json \
  --step-log experiments/maze-query-hub-prototype/results/_phase4c/steps.json \
  --persist-graph-sqlite experiments/maze-query-hub-prototype/results/_phase4c/graph.sqlite \
  --persist-forced-candidates
```

### HTML 生成（例）
```
.venv/bin/python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/_phase4c/summary.json \
  --steps   experiments/maze-query-hub-prototype/results/_phase4c/steps.json \
  --sqlite  experiments/maze-query-hub-prototype/results/_phase4c/graph.sqlite \
  --present-mode strict --strict --light-steps \
  --out     experiments/maze-query-hub-prototype/results/_phase4c/interactive.html
```

### ステップ上限を変えた安全ラン（25x25→500, 51x51→1500）
論文 v4 では 250step 設定でしたが、25×25 / 51×51 で同じ設定を繰り返すと長距離経路が十分に観測できず、編集しながらの再実験でミスを起こしやすくなります。`tools/run_adjusted_step_caps.py` は迷路サイズごとの安全キャップを守りながら、L3-only 構成で再実行するためのラッパーです。

````bash
python experiments/maze-query-hub-prototype/tools/run_adjusted_step_caps.py \
  --targets 25:600:60,51:2000:40 \
  --safety-caps 25:500,51:1500 \
  --out-root experiments/maze-query-hub-prototype/results/adjusted_steps \
  --namespace-prefix adj_v4 --ultra-light
````

- `size:max_steps[:seeds]` 形式で複数ターゲットを列挙し、`--safety-caps` で最大全長を指定（既定: 25→500, 51→1500）。
- 既定で **seed ごとに分割実行**し、`_seed{n}` 付きの summary/steps/SQLite/HTML を生成します。従来どおりまとめて出力したい場合は `--aggregate-output` を付けてください。
- SP は `--sp-cache-mode cached_incr` + SP pairset DS (`sp_pairsets.sqlite`) を基盤に、**広域ノルム (`--link-radius 1 --cand-radius 1`)** と **無閾値 (`--theta-link 0 --theta-cand 0`)**、**SP貪欲無制限 (`--sp-cand-topk 0`)** を既定オンにしています。これにより遠方の未探索メモリや forced fallback も常に候補・SP評価に載ります。必要に応じて `--extra-args -- --sp-cand-topk 32` などで調整してください。
- `--build-html` を付けると各 seed の summary/steps/SQLite から `build_reports.py` を呼び出し、Strict DS の HTML を自動生成します。
- キャップを超えた値は自動で丸められるため、実行時間の暴走を防げます。明示的に超えたい場合のみ `--force-over-cap` を指定してください。
- 論文用スクリプト（`run_paper_l3_only.py`, `run_paper_comparison.py`, `run_paper_grid.py`, `.sh` ランナー）は既定で **最小スナップショット＋`--steps-ultra-light`** を採用します。フル記録が必要な場合のみ `--rich-logs` を追加してください。
- **短縮版（スループット優先）**: 長時間バッチでは以下を `--extra-args` で指定すると 1 seed あたりの時間を抑えられます。
  - `--sp-cand-topk 32` …… ΔSP の貪欲候補を最大 32 件に制限（精度とのトレードオフあり）。
  - `--candidate-cap 24 --top-m 24` …… ランキング対象を絞り、SP/IG の計算を軽量化。
  - `--link-radius 0.7 --cand-radius 0.7` …… 遠方メモリを減らして候補生成時間を短縮。
  - リスク: 遠距離ノードや low-frequency の洞察が候補から漏れる場合があるため、最終報告ではフル設定での再走を推奨。

### HTML トグル（初期値は `--strict/--relaxed` で設定可能）
- Use DS graph / Strict DS / Pre‑step (eval) graph / Cumulative（積み上げ）
- Timeline uses mh‑only minima（gminはhop>=1のみで集計）
- Show all query nodes（過去Qの表示） / Show SP anchors（アンカー表示）

## CLI（抜粋）
- geDIG/評価
  - `--max-hops H` / `--lambda-weight` / `--sp-beta` / `--sp-scope` / `--sp-boundary` / `--sp-hop-expand`
  - `--theta-ag` / `--theta-dg` / `--top-link`
  - `--norm-base {link|cand}` / `--linkset-mode`
  - `--gh-mode {greedy|radius}` / `--pre-eval/--no-pre-eval` / `--eval-all-hops`
- 候補/行動
  - `--theta-cand/--theta-link` / `--candidate-cap` / `--top-m` / `--cand-radius/--link-radius`
  - `--action-policy {argmax|softmax}` / `--action-temp τ` / `--anti-backtrack/--no-anti-backtrack`
- 永続化/スナップショット
  - `--persist-graph-sqlite PATH` / `--persist-namespace` / `--persist-forced-candidates` / `--persist-timeline-edges`
  - `--snapshot-level {minimal|standard|full}` / `--snapshot-mode {before_select|after_select}`
- ビルド（HTML）
  - `--sqlite PATH` / `--present-mode {none|strict|relaxed}` / `--strict/--relaxed`
  - `--light-steps`（steps.jsonを軽量化して埋め込み）


## 開発の進め方
1. **Skeleton 作成**: 既存 `maze-fixed-kstar` からコピーし、中心ノード関連ユーティリティを `query` 仕様に差し替えた雛形を置く。
2. **クエリノード管理**: 生成タイミング・訪問カウント同期・リンク張り替えを実装。
3. **geDIG 入力の調整**: `prev_graph` 保存タイミングとサブグラフ抽出ロジックを更新。
4. **ログ/HTML**: JSON 出力・インタラクティブビューアを新モードに対応。
5. **比較実験**: 同シード・同パラメータで現行実装と挙動を比較し、気になる分岐（T 字路など）で挙動を記録。

## 次のタスク候補
- `run_experiment.py` のクローン（例: `run_experiment_query.py`）を配置して骨組みを実装。
- クエリノード生成／破棄まわりの単体テストを追加（`tests/` 配下）。
- HTML ビューアを `graph_mode` 切り替え対応にする（`query_hub` 描画は実装済み、既存モードとの共存を仕上げる）。

## 備考
- 既存コードはそのまま残し、`maze-query-hub-prototype` で新パスを育てる。
- インタラクティブビューアのテンプレートは `query_interactive_template.html`。実験ごとにコピーし、`build_reports.py` で `const experimentData = ...` を差し込む。
- 構造が確定するまでは `README.md` に仕様更新を追記する。


## オプション: Linkset モード

`--linkset-mode` を付与すると、g₀ を S_link（確定リンク集合）ベースで評価します。
このモードでは before=S_link、after=S_link+決定アクションとして ΔGED / ΔH を算出し、行き止まりや分岐での局所的なコスト変化を把握しやすくなります。
```
poetry run python run_experiment_query.py --linkset-mode --output ...
```
デフォルトのグローバル評価に戻したい場合はフラグを付けずに実行してください。

### 追加CLI（安定化・二段ゲート）
- `--theta-ag`, `--theta-dg`: 二段ゲートの閾値（既定 0.0, 0.0）
- `--top-link`: 0-hop 暫定配線に使う S_link の本数（既定 1）
- `--commit-budget`: DG 成立時のコミット上限本数（既定 1）
- `--commit-from`: コミット候補の台（`cand`=S_cand, `link`=S_link）
- `--norm-base`: 正規化の台（`link`=S_link基準, `cand`=S_cand基準）。アブレーション用（既定 `link`）。


## 使い方（How to Use）

### 1) セットアップ（初回のみ）

Pythonの仮想環境を使う場合の一例:

```
python3 -m venv .venv
source .venv/bin/activate
pip install networkx numpy
```

### 2) 最小実行（クイック）

推奨既定（λ=1.0, γ=1.0, hops=10, linkset IG, 正規化=|S_link| 基準）で60ステップを1シード実行し、HTMLを作成:

```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 \
  --linkset-mode --norm-base link \
  --output experiments/maze-query-hub-prototype/results/quick25_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/quick25_steps.json

python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/quick25_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/quick25_steps.json \
  --out     experiments/maze-query-hub-prototype/results/quick25_interactive.html
```

### 3) 長めの実験（168/500 ステップ例）

```
# 168 steps
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 168 \
  --linkset-mode --norm-base link \
  --output experiments/maze-query-hub-prototype/results/run25_s168_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/run25_s168_steps.json

# 500 steps
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 500 \
  --linkset-mode --norm-base link \
  --output experiments/maze-query-hub-prototype/results/run25_s500_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/run25_s500_steps.json
```

### 4) L3 ライトパス（軽量・hop0のクエリ中心）

main の L3GraphReasoner を直接使い、各ステップの評価を hop0 のクエリ中心で簡易化します。
ΔSP は `cached`/`cached_incr` を使用し、ペア集合は ENV の `INSIGHTSPIKE_SP_REGISTRY` で再利用可能です。

```
INSIGHTSPIKE_SP_REGISTRY=experiments/maze-query-hub-prototype/results/pairsets.json \
INSIGHTSPIKE_SP_ENGINE=cached_incr \
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 60 \
  --sp-cache --sp-cache-mode cached_incr --sp-pair-samples 80 --commit-budget 2 \
  --use-main-l3 \
  --output experiments/maze-query-hub-prototype/results/l3lite_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/l3lite_steps.json

python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/l3lite_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/l3lite_steps.json \
  --out     experiments/maze-query-hub-prototype/results/l3lite_interactive.html
```

## 高速処理オプションと手法の解説

速度を落とさずに“意味のある時だけ”多ホップ評価を回す設計です。以下のオプションを必要に応じて組み合わせます。

### A) DG発火時に g0 で繋がる範囲を一括反映（S_link全配線）

- フラグ: `--dg-commit-all-linkset`
- 内容: DG発火（best_hop≥1 かつ g_min(mh)<θ_DG）時、hop0のTop‑L制限を外し、S_link（確定リンク）を全本コミットします。
- 手法意図: “今ステップでg0時点で繋がっている範囲”をまとめて反映（g0のBFSに等価）。次ステップで同近傍を再評価する回数が減り、バックトラック周辺での多ホップが削減されます。

### B) 行き止まり/バックトラック時は多ホップ評価をスキップ

- フラグ: `--skip-mh-on-deadend`
- 内容: `possible_moves ≤ 1`（事実上の行き止まり or 即戻り）と判定したステップは、hop0のみ（max_hops=0相当）を評価します。
- 手法意図: デッドエンドでの高コストな多ホップ探索を回避（P95レイテンシの顕著な低下）。

### C) AGの動的分位（g₀で早期判断）

- フラグ: `--ag-auto`（必要に応じて `--ag-window`, `--ag-quantile`）
- 内容: 直近の g₀ 分布の分位（例: 90%）をθ_AGとして用い、g₀が十分に小さい（改善見込みが薄い）ステップでは多ホップをスキップします。
- 手法意図: 無駄な多ホップを抑制。A/Bと併用で速度・安定の両立を図れます。

### D) SP計算のパフォーマンス（必要に応じて）

- フラグ: `--sp-cache`（`--sp-pair-samples` と併用）
- 内容: ΔSP（平均最短路）の評価を端点SSSP合成の近似/キャッシュで高速化（固定ペア/DS再利用の構成もあり）。
- 手法意図: multi-hopのSP評価コストを削減。長尺実験や大きな迷路サイズで効きます。

### おすすめプリセット

- 高速重視（P95抑制）:
  - `--dg-commit-all-linkset --skip-mh-on-deadend --ag-auto`
- 解析重視（詳細確認）:
  - 既定のまま（必要時のみ `--sp-cache`）

## 表示/解釈のヒント（UI）

- Timeline の赤線は既定で「overall g_min（hop0含む）」です。UIの「Timeline uses mh-only minima」をONにすると「g_min(mh)（hop≥1のみ）」に切り替わります。
- Stepパネルの「Per-hop Metrics」は hop_series の g(h) をそのまま、Step 情報の g_min は「hop0含む」最小を表示します。両者の不一致はこの定義差です。

## 既定と論文の整合（メモ）

- ΔIG = ΔH + γ·ΔSP（γ=`--sp-beta`、既定=1.0）
- ΔH = H_after − H_before（秩序化で負）
- GED 正規化は候補台（Cmax = c_node + |S_link|·c_edge）
