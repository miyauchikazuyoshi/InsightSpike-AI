# グラフ構造パターン認知を Sleep に取り込む：意味空間醸成の設計メモ

**Version**: 0.1 (Draft)  
**Date**: 2026-01-29  
**Author**: Kazuyoshi Miyauchi  
**Status**: Proposal

---

## 0. 先に結論

「グラフ構造パターン認知（同型発見 / 構造類似 / モチーフ抽出）」は、**意味空間（埋め込み/索引）**と**Sleep（オフライン統合）**の両方に直結する。

- **意味空間**：パターン認知で得られる「同型・近同型」「繰り返し構造（モチーフ）」「近いが破綻する（hard negative）」が、対比学習の教師信号になる
- **Sleep**：Wakeで集めたエピソード（グラフ/ログ）から、再利用できる“語彙（chunk/macro）”と“不変量（invariants）”を抽出して整理する時間になる

この接続を作ると、「if文で解ける特化」を避けつつ、**トポロジカルなルール（構造不変量）**を “再利用可能な表現” として育てやすくなる。

---

## 1. 役割分担（Wake / Sleep）

### 1.1 Wake（オンライン）

Wake は「探索して、証拠を集めて、ログに残す」フェーズ。

- エピソード（迷路/ARC/その他）を **グラフ表現**として残す
- AG/DG を通じて採択された編集・候補を “正解ラベル” ではなく **証拠（evidence）**として保存する
- 失敗（blocked / near-miss）も資産として残す（後で hard negative になる）

参照：
- DG ledger（迷路PoC）: `experiments/maze-query-hub-prototype/run_experiment_query.py --dg-ledger-log ...`

### 1.2 Sleep（オフライン）

Sleep は「表現と索引を作り直す」フェーズ。

- 同型発見・構造類似・モチーフ抽出で、繰り返し構造をまとめる
- まとめた結果を “語彙（chunk/macro/テンプレ）” として登録し、次のWakeで初期候補生成を速くする
- hard negative（近いが破綻する）を整理し、提案器の精度を上げる

---

## 2. 「意味空間」の最小定義（このrepoの運用レベル）

読み物的な定義ではなく、実装の受け皿として最小セットに落とす。

- `episode_vector`: エピソード（グラフ/ログ）→ d次元ベクトル
- `index`: 近傍検索（例: `hnswlib` / `faiss-cpu`）
- `concept_bank`: Sleepで抽出された “構造語彙” の辞書（モチーフID、対応変換IDなど）

意味空間は「真理」ではなく、**候補集合を狭める prior** として使う（DGの監査対象）。

---

## 3. パターン認知 → 教師信号（Sleepの学習データ化）

### 3.1 Positive / Negative の作り方（構造ベース）

- **Positive（近づける）**
  - 同型（isomorphic）または低コスト変換（低 `Transform.cost`）で結べるエピソード/部分グラフ
  - DG commit 後に “再現性のある改善” を起こした編集パターン（再利用実績つき）
- **Hard negative（遠ざける）**
  - near-miss（ほぼ同型だが、1条件で破綻する）
  - DG reject / `blocked` / 壁衝突などの “構造化された失敗”

この「near-miss を hard negative 化」できるのが、グラフ構造パターン認知を入れる最大のメリット。

### 3.2 Lossの最小案（いきなりGNNしない）

最初は **埋め込み学習なし**でもよい（モチーフ頻度ベクトルで近傍検索するだけでも Sleep の価値が出る）。

埋め込みを学習する場合の最小は対比学習（InfoNCE/Triplet）：

- 入力: `episode_vector(anchor)`, `episode_vector(pos)`, `episode_vector(neg*)`
- 目的: `sim(anchor,pos)` を上げ、`sim(anchor,neg)` を下げる

補足：FEP/ELBO的な枠組みは v2 以降で合流しやすい（例：VAEの再構成 + KL に、構造パターン由来の正則化を足す）。

---

## 4. 既存コード/実験との接続ポイント

### 4.1 同型発見（構造パターン認知）

- 実験: `experiments/isomorphism_discovery/benchmark.py`
- 実験: `experiments/isomorphism_discovery/novel_analogy_discovery.py`
- 実装: `src/insightspike/algorithms/isomorphism_discovery.py`（`Transform` / `Transform.cost`）

### 4.2 DG ledger（Wakeの証拠）

- 実験: `experiments/maze-query-hub-prototype/run_experiment_query.py`
- 出力: `--dg-ledger-log ...jsonl`（`staged_edges` と `committed_edges` を含む）

---

## 5. 出力アーティファクト（Sleepの成果物）案

Sleepは「後で監査できる形」で残すのが重要（DG感）。

### 5.1 最小（v0）
- `results/sleep/motifs.jsonl`: 抽出したモチーフのIDと統計（頻度、支持エピソード、再利用実績）
- `results/sleep/episode_vectors.jsonl`: エピソードID→ベクトル（または特徴量）
- `results/sleep/index.*`: 近傍検索用のインデックス

### 5.2 追加（v1以降）
- `results/sleep/pairs.jsonl`: (anchor, positive, negatives[])（対比学習用）
- `results/sleep/transforms.jsonl`: `Transform` の集約（同型/近同型のクラスタ）

---

## 6. ロードマップ（パラメータ爆発を防ぐ）

### v0: “パターンを保存できる” を作る
- 同型発見結果（`Transform`）とDG ledgerから、pairs/モチーフ統計を作って保存する
- 近傍検索が「ちゃんとそれっぽい」ことを目視確認できる（まずはここ）

### v1: Sleepで意味空間を更新し、Wakeで使う
- Sleepで `episode_vector` と `index` を更新
- Wakeの候補生成を “近傍エピソード由来の初期案” でブーストする

### v2: FEP/ELBO（不確実性）を入れる
- エピソードを点ではなく分布として持ち、曖昧性（解釈の揺れ）を管理する
- ただし最初から入れると設計自由度が増えるので、v1が回ってから

---

## 7. ここが「DG感」になる（設計上のポイント）

Sleepで増える自由度（語彙/特徴/埋め込み）が暴走しないよう、採択をDG化する。

- “語彙追加（モチーフ/マクロ）” は仮説として記録し、改善が再現したら commit
- near-miss/hard negative を構造化して残し、次の提案器に必ず効かせる
- 取り込んだ要素の「コスト（ΔEPC）」と「利得（ΔIG）」をログに残す（後で撤回できる）

関連設計：
- `docs/design/episode_memory_autodesign.md`
- `docs/research/self_organizing_world_model.md`

