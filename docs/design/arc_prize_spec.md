# ARC Prize Solver Spec（geDIG駆動のDSLプログラム探索）

**Version**: 0.1 (Draft)  
**Date**: 2026-01-29  
**Author**: Kazuyoshi Miyauchi  
**Status**: Proposal

---

## 1. Scope / Non-goals

### Scope
- ARCタスク（2Dグリッド）を対象に、**DSL + 探索**で解を生成するソルバを実装する。
- geDIG（`F = ΔEPC_norm − λ·ΔIG_norm`）と AG/DG を用い、探索・統合・記憶を制御する。

### Non-goals（最初はやらない）
- 大規模LLMによる直接出力（ルール次第で禁止の可能性があるため、別トラック扱い）
- 画像モデルでの end-to-end（後段で検討）

---

## 2. Data Model（入出力と内部表現）

### 2.1 Grid
- `Grid`: `np.ndarray`（shape=`(H, W)`、dtype=`uint8`推奨）
- 値域: `0..9`（色）

### 2.2 Task
- `ARCExample`: `{ "input": Grid, "output": Grid }`
- `ARCTask`: `{ "train": list[ARCExample], "test": list[Grid] }`

### 2.3 I/O
- ARC JSON（一般的な形式）を読み込み、内部 `ARCTask` に変換する
- 予測は `list[Grid]`（test input の個数ぶん）

### 2.4 追加メタデータ（推奨）
- `task_id`
- `source_split`（train/dev/test）
- `time_budget_ms`（提出制約に合わせて）

---

## 3. DSL（Domain Specific Language）

### 3.1 設計方針
- **型付きDSL**（探索爆発を抑える）
- **合成可能**（小さな操作の組み合わせで大きな変換を表現）
- **コストが定義できる**（ΔEPCの基礎）
- **同値性を潰せる**（正規化/キャッシュ）

### 3.2 型（最小セット）
- `Grid -> Grid`
- `Grid -> ObjSet`（物体抽出）
- `ObjSet -> Grid`（合成）
- `Obj -> Obj` / `ObjSet -> ObjSet`（フィルタ、変換）
- `Grid -> Scalar`（色/サイズ/周期などの特徴）

### 3.3 基本オペレータ（v1候補）
例（最初は少数で良い）：
- `rotate90/180/270(grid)`
- `flip_h/flip_v(grid)`
- `crop_to_bbox(grid, color?)`
- `pad(grid, top,bottom,left,right, fill)`
- `recolor(grid, src_color, dst_color)`
- `replace_color(grid, mapping)`
- `translate(grid, dx, dy, fill)`
- `extract_objects(grid, connectivity=4/8, ignore_bg=true)`
- `filter_objects(objs, predicate)`
- `paint_objects(canvas, objs, mode=overwrite/merge)`

### 3.4 Object Model（v2で拡張）
`Obj` は以下を持つ：
- `mask`（Gridと同サイズのbool配列、または座標集合）
- `bbox`（min/max）
- `color_hist`
- `area`, `perimeter`（軽量）
- `anchor`（重心/左上など）

---

## 4. Program Representation

### 4.1 AST / Program
- `Op`（名前、入出力型、パラメータスキーマ、実行関数）
- `Node`（`Op` + 引数（入力ノード参照 or 定数））
- `Program`（DAG推奨：共通部分式を共有してキャッシュ効率を上げる）

### 4.2 Serialization
- JSONで `Program` を保存できる（再現・監査用）
- 実行ログに「候補プログラム」を必ず保存（上位K件）

---

## 5. Search Engine

### 5.1 基本アルゴリズム

最初は「型付き列挙 + ビーム」を標準とする。

- **Frontier**: 候補プログラム集合（深さd）
- **Expand**: 型が一致する `Op` を末尾に追加し、パラメータを列挙して次候補を生成
- **Score**: train ペアに対する損失（完全一致 + 近似損失）とコストからスコアリング
- **Prune**: キャッシュ/同値排除/ビーム幅/時間で枝刈り
- **Stop**: train 完全一致を達成した最良候補を返す（提出では test に適用）

### 5.2 近似損失（Partial Credit）

ARCは最終判定が「完全一致」でも、探索中は勾配の代替が必要になる。

推奨する複合損失（例）：
- `L_shape`: shape 不一致（強いペナルティ）
- `L_cell`: セル一致率（Hamming）
- `L_color`: 色集合の一致度（Jaccardなど）
- `L_obj`: 物体数/サイズ分布の一致度（v2以降）

最終的な探索用損失：
```
L = w_shape*L_shape + w_cell*L_cell + w_color*L_color + w_obj*L_obj
```

### 5.3 キャッシュと同値排除

探索の勝敗はキャッシュで決まる。

- `Grid` のハッシュ（bytes化 + shape）
- `Program` の正規形（commutative な操作、無意味なパラメータ差を潰す）
- 「同じ入出力を生む」プログラムの抑制（outputハッシュで代表を残す）
- train の各例に対する中間結果をメモ化（例ごとに結果が違う点に注意）

---

## 6. geDIG統合（ARC探索への写像）

### 6.1 ゲージ（F）の定義（ARC版）

ARC探索では「構造編集 = プログラム編集」と解釈する。

```
F = ΔEPC_norm − λ · ΔIG_norm
```

- **ΔEPC_norm**: 「複雑さの増加」
  - 例: オペレータ1つ追加、パラメータ自由度の増加、物体抽出など高コスト操作
  - 正規化: 深さdにおける上界で割る、または累積コストの `tanh` など
- **ΔIG_norm**: 「当たりに近づいた量」
  - 例: train 損失 `L` の減少、あるいは不一致セル数の減少
  - 正規化: `L` のスケールに依存しない形（0..1）へ

### 6.2 AG/DG（論文と同じ意味で使う）

- **AG（Attention Gate）**: `g0 > θ_AG` なら探索を深める（探索幅/深さ/高コストDSLを解放）
- **DG（Decision Gate）**: `min{g0, g_min} ≤ θ_DG` なら統合を確定（部分プログラム/マクロを採択）

ここで：
- `g0`: 0-hop（安価）評価（例: 簡易DSLのみ、部分例のみで評価）
- `g_min`: multi-hop（高価）評価（例: 深い合成、物体DSL、より多くの候補）

### 6.3 何を“探索”し、何を“統合”するか

ARCにおける統合対象（DGで確定して再利用するもの）：
- **部分プログラム（subprogram）**
- **マクロ（頻出テンプレ）**
- **特徴抽出器（例: 「背景色」「主要色」「周期」推定」）**
- **探索方針（このタイプのタスクではこのDSL順が効く等）**

### 6.4 閾値の校正（Quantile calibration）

論文流儀に合わせ、`θ_AG` と `θ_DG` は「固定値」ではなく、
**dev（検証）で分位校正してから test で固定**する。

---

## 7. 記憶（Macro Library）と Wake/Sleep

### 7.1 Wake（オンライン）
- DGで確定した subprogram を「短期記憶」に保存
- 保存単位は `Program`（JSON）＋適用条件（型/特徴）＋性能ログ

### 7.2 Sleep（オフライン）
- 解けたタスク群から subprogram をマイニングし、マクロ化（共通部分抽出）
- 冗長なマクロを剪定（低再利用/高コスト/低IG）
- マクロを DSL と同等の `Op` として登録（探索で使える）

### 7.3 自己生成負例（Hard Negative Mining）

ARCには明示的な負例ラベルがないため、探索過程から負例を自己生成して再利用する。

- **負例の源泉**：AGで検討対象になったが、DGで棄却された候補プログラム
- **Hard negative**：train例の大半は合うが、少数例で破綻する near-miss を優先保存
- **ログ要件**：どの train 例で破綻したか、差分（セル不一致数など）を保存して「負例の理由」を残す
- **利用先**：Sleepで提案器（次のOp/パラメータ提案）や索引（近傍タスク検索）を改善する教師信号にする

> 詳細設計: `docs/design/episode_memory_autodesign.md`

### 7.4 エピソード記憶の自己設計（メタDG）

どの粒度で subprogram を「エピソード」として固定し、どの特徴で索引するかも仮説として扱う。

- 変更案（粒度/特徴/温度/負例比率）を候補として生成
- held-out（dev）で改善が安定に出る場合のみ、メタDGで commit する

---

## 8. 実験・ログ（監査可能性）

### 8.1 タスク単位ログ（必須）
- 入力/出力（train/test）
- 探索予算（時間/ノード/ビーム幅）
- 上位K候補のプログラムとスコア
- `g0/g_min` の時系列、AG/DGの発火点
- 失敗時: 最良候補の出力と差分

### 8.2 失敗の分類（推奨）
- DSL不足
- 探索爆発（予算不足）
- オブジェクト対応失敗
- 過学習（train-fitはあるがtestで落ちる）

---

## 9. CLI / Script（最低限）

例：
- `scripts/arc/download_data.py`（データ取得、または配置チェック）
- `scripts/arc/run_eval.py --split dev --budget-ms 2000`
- `scripts/arc/solve_one.py --task-id ... --dump-trace`
- `scripts/arc/render_task.py --task-id ...`（可視化）

---

## 10. テスト戦略

- DSLオペレータの単体テスト（入出力shape、色域、同値性）
- 小さなゴールデンタスクでの回帰（「このタスクはこのプログラムで解ける」）
- 速度回帰（キャッシュが効いているか）

---

## 11. 実装マッピング（推奨構成）

```
src/insightspike/arc/
  io.py                 # JSON load/save, split
  types.py              # Grid, Task, Example
  program.py            # Op/Node/Program, cost, serialization
  dsl/
    ops_grid.py         # Grid->Grid ops
    ops_objects.py      # Grid<->ObjSet, Obj ops
    registry.py         # Op registry, param enumeration
  search/
    enumerative.py      # BFS/beam + caching
    scoring.py          # L, IG, EPC, F, AG/DG logic
  solver.py             # solve(task) -> predictions
scripts/arc/
  run_eval.py
  solve_one.py
  render_task.py
```

---

## 12. 先に決めるべき設計判断（実装前に固定）

- 提出制約（CPU/GPU、タイムアウト、外部依存）の前提
- DSL v1 の範囲（“少数で勝てる”操作をどう選ぶか）
- 近似損失 `L` の形（探索誘導の品質を左右）
- ΔEPC/ΔIG の正規化と、`λ, θ_AG, θ_DG` の校正手順
