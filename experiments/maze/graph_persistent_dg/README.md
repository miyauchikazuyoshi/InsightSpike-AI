# Graph-Persistent DG

Wake1 → Sleep → Wake2 パイプラインにおいて、**グラフを捨てずに引き継ぐ**ことで
2回目の試行を改善する実験モジュール。

## 実験目標と証拠状態（2026-06 時点）

**目標**: wake で構築した記憶グラフを sleep（報酬伝播＋剪定）で再編して引き継ぐと、
次の試行が改善することを示す。これは論文の Phase 2（オフライン再配線）の最小実装であり、
将来の Sleep-RAG（チャンク粒度・エッジ重みの自律調整）の機構検証を兼ねる。

**従属変数**: 成功率は使わない（15×15 では 98% で天井に近く感度がない）。
主要指標は **平均ステップ数・最終エッジ数（圧縮）・再訪時の改善幅**。

**現状の証拠と限界**:
- v6_perseed（25×25）: 成功率 71.9%→95.3%、平均ステップ −41%、エッジ −34%
- **ただしこれは 10D ベクトル＋sleep 伝播＋カリキュラムを同時に変えたパッケージ効果**で、
  sleep 単独の寄与は未分離（baseline n=32 / extended n=64 とサンプル数も非対称）
- **次の実験（最優先）**: sleep on/off 単独アブレーション
  （`--vector-mode extended`・カリキュラムは両群で固定、sleep 伝播のみ切替、n≥30 seeds）。
  **事前登録済み（2026-07-02）**: [`docs/prereg/maze_sleep_ablation.md`](../../../docs/prereg/maze_sleep_ablation.md)
  — 予測 P1–P3・反証条件・打ち切り処理を登録。FROZEN 化（パイロット 1 シードで実行時間確認）後に実行する。
  切替フラグ `--sleep-propagate {on,off}` 実装済み（off = 生 Wake1 グラフを無変更で継承）。
  実行: `bash run_sleep_ablation.sh 0 29`（on/off を同一シードで paired 実行）

## 概要

### 問題 (従来)
```
Wake1: graph = nx.Graph()  →  探索  →  graphは捨てられる
Sleep: stepsログから辞書(sleep_plan, sleep_q, sleep_edge)を抽出
Wake2: graph = nx.Graph()  →  辞書をβで加算  →  グラフを辿っていない
```

### 解決 (本モジュール)
```
Wake1: graph構築 + 各ノードにreward記録
Sleep: graphを最適化 (報酬伝播 + ベクトル同期)
Wake2: 最適化graphを引き継ぎ、類似度計算自体が報酬情報を含む
```

## ベクトル拡張

`--vector-mode extended` で 8D → 10D に拡張。

| dim | 内容 | weight | 備考 |
|-----|------|--------|------|
| 0-1 | 位置 (row/col) | 1.0 | 既存 |
| 2-3 | 方向 (dx/dy) | 0.0 | 既存 |
| 4 | 通過可否 | 3.0 | 既存 |
| 5 | 訪問回数 | 2.0 | 既存 |
| 6-7 | success/goal | 0.0 | 既存 |
| **8** | **reward** | **2.0** | 新規: ヒューリスティック報酬 |
| **9** | **tanh(propagated)** | **3.0** | 新規: Sleep伝播後の累積値 |

query vector は dim8=1.0, dim9=1.0 を持つため、
正報酬/正伝播のノードが L2距離で「近い」= 高類似度になる。

## 報酬テーブル

実装値（`run_experiment_query.py` の報酬記録部を正とする。SPEC.md §2.1 の +0.3/−0.3 は
設計時の初期案で、§7 の調整指針に従い下記に調整済み）:

| イベント | reward | 根拠 |
|----------|--------|------|
| goal到達 | +1.0 | 目的達成 |
| 新セル通過 | +0.2 | 探索の進展 |
| 既訪問セル再訪 | -0.4 | 進展なし |
| 行き止まり | -1.0 | 構造的な罠（revisit等より優先） |
| 壁衝突 | -1.0 | 無効な行動 |

## Sleep伝播アルゴリズム

```
propagated(n) = reward(n) + γ · max(propagated(neighbor))
```

Q-learning的な反復伝播。goalの正報酬が上流に、dead-endの負報酬が上流に
それぞれγ減衰で伝播する。

## 使い方

### 基本 (extended vector + curriculum)

```bash
cd experiments/maze
export PYTHONPATH="../../src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

python3 run_experiment_query.py \
  --maze-size 25 \
  --max-steps 500 \
  --seed-start 32 --seeds 1 \
  --max-hops 15 \
  --lambda-weight 1.0 \
  --decay-factor 0.7 \
  --theta-ag -1.0 --theta-dg 0.15 \
  --sp-scope union --sp-hop-expand 3 --sp-boundary trim \
  --action-policy softmax \
  --curriculum-warmup-steps 500 \
  --sleep-guide prefer \
  --vector-mode extended \
  --sleep-propagate-gamma 0.95 \
  --sleep-propagate-iters 50 \
  --output results/graph_persistent_dg/seed32_extended.json
```

### ベースライン比較 (standard vector, カリキュラムなし)

```bash
python3 run_experiment_query.py \
  --maze-size 25 \
  --max-steps 500 \
  --seed-start 32 --seeds 1 \
  --max-hops 15 \
  --lambda-weight 1.0 \
  --decay-factor 0.7 \
  --theta-ag -1.0 --theta-dg 0.15 \
  --sp-scope union --sp-hop-expand 3 --sp-boundary trim \
  --action-policy softmax \
  --output results/graph_persistent_dg/seed32_baseline.json
```

### カリキュラム + standard vector (辞書方式との比較)

```bash
python3 run_experiment_query.py \
  --maze-size 25 \
  --max-steps 500 \
  --seed-start 32 --seeds 1 \
  --max-hops 15 \
  --lambda-weight 1.0 \
  --decay-factor 0.7 \
  --theta-ag -1.0 --theta-dg 0.15 \
  --sp-scope union --sp-hop-expand 3 --sp-boundary trim \
  --action-policy softmax \
  --curriculum-warmup-steps 500 \
  --sleep-guide prefer \
  --vector-mode standard \
  --output results/graph_persistent_dg/seed32_curriculum_std.json
```

### オプション一覧

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--vector-mode` | standard | `standard` (8D) or `extended` (10D) |
| `--propagated-alpha` | 1.0 | standardモード用のスカラーボーナス重み |
| `--sleep-propagate-gamma` | 0.95 | 伝播の割引率 (大→遠距離影響、小→近距離のみ) |
| `--sleep-propagate-iters` | 50 | 伝播の最大反復数 |
| `--curriculum-warmup-steps` | 0 | Wake1のステップ数 (0=カリキュラム無効) |

## ファイル構成

```
graph_persistent_dg/
├── README.md              ← この文書
├── SPEC.md                ← 設計仕様書
├── __init__.py
└── sleep_propagate.py     ← Sleep伝播エンジン
```

変更された既存ファイル:
- `qhlib/utils.py` — 10Dベクトル、weight_vector拡張
- `qhlib/cli.py` — CLIオプション追加
- `qhlib/models.py` — EpisodeArtifacts.graph フィールド
- `qhlib/l3_adapter.py` — 次元数対応
- `run_experiment_query.py` — 報酬記録、Sleep配線、scoring拡張
