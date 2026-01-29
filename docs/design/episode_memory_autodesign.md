# エピソード記憶の自己設計：AG/DG × Wake/Sleep × 自己生成負例

**Version**: 0.1 (Draft)  
**Date**: 2026-01-29  
**Author**: Kazuyoshi Miyauchi  
**Status**: Proposal

---

## 1. 背景：人間の解き方 = 負例生成のループ

人間の認知は、ざっくり言うと次の流れになっている。

1. 正解（デモ）を見る
2. 操作方法・UI（表現/操作の語彙）を把握する
3. ルールを抽象的に把握する（仮説を立てる）
4. 問題から回答を生成する

このとき本質的なのは、**「外れた仮説を捨てる」過程が、次の学習信号（負例）になっている**点である。

ARCの文脈では、trainの入出力例が「正解（デモ）」に相当し、テストの正解は見えない。したがって、負例生成は **train例への整合性**（および内部の整合性・簡潔性）から作られる必要がある。

---

## 2. 目的 / 非目的

### 2.1 目的
- **負例（Hard Negative）を自己生成**し、探索・記憶・表現学習に再利用する。
- **エピソード記憶の単位（chunking）**と**索引（index/embedding）**を、Wake/Sleep で自己改善する。
- 迷路PoCとARC（DSL探索）の両方に通る、実装可能な最小定義を置く。

### 2.2 非目的（最初はやらない）
- テスト正解の利用（ARC系では禁止/不可能）
- 大規模モデルによる end-to-end 生成（別トラック扱い）

---

## 3. 用語（最小）

- **Hypothesis（仮説）**：ルール、変換、部分プログラム、エピソード分割規則など「提案され、検証されるもの」。
- **Episode（エピソード）**：記憶に保存される経験単位（状態・行動・結果・コンテキスト）。
- **Positive（正例）**：DGで確定（commit）された仮説、または確定したエッジ/マクロ。
- **Negative（負例）**：AGで検討対象になったがDGで棄却、あるいは検証で破綻した仮説。
- **Hard Negative**：ほぼ正しいが、ある条件（例: 1つのtrain例）で破綻する仮説。

---

## 4. 全体ループ（geDIG + AG/DG）

この設計では、負例生成は「副産物」ではなく**第一級の成果物**として扱う。

### 4.1 Wake（オンライン）

1. **提案（Propose）**：候補仮説 `h`（候補エッジ/候補マクロ/候補プログラム/候補分割）を生成する
2. **軽量評価（0-hop）**：`g0(h)` を計算する（安価な近似）
3. **AG（Attention Gate）**：`g0 > θ_AG` なら探索を開く（候補増やす / 高コストDSL解放 / multi-hopへ）
4. **高価評価（multi-hop）**：`g_min(h)` を計算する（より厳密な検証）
5. **DG（Decision Gate）**：`min(g0, g_min) ≤ θ_DG` なら commit する
6. **ログ**：
   - commit された仮説 → **正例**
   - 棄却された仮説 → **負例（理由付き）**

> 注：AG/DGの形式は実装の `src/insightspike/algorithms/gating.py` と整合する。

### 4.2 Sleep（オフライン）

Sleep は「記憶や表現を作り直す時間」であり、負例を使って次を改善する。

- **負例マイニング**：近いが破綻した仮説（hard negatives）を優先抽出
- **表現の学習**：Episode/Task/Program の埋め込みを、対比学習で再配置（「醸成」）
- **マクロ圧縮**：DG確定の部分構造を共通化して DSL マクロ化（再利用率を上げる）
- **自己設計の更新（メタDG）**：
  - 例: 「どの粒度でエピソードを切るか」「どの特徴で索引するか」「負例の比率や温度」
  - 変更案を “仮説” として扱い、改善が一貫して出るときのみ commit する

---

## 5. 自己生成負例の種類（推奨カタログ）

負例は「単に失敗した」では弱い。**どこで破綻したか**が価値になる。

1. **即死負例（easy）**：明らかに矛盾（trainの多くで破綻）
2. **近接負例（near-miss）**：trainの大半は合うが、少数例で破綻
3. **不変量破壊（invariance-break）**：回転/平行移動など、想定不変量を崩すと破綻
4. **過剰自由度（overfit）**：train-fitはするが、記述が過度に複雑（ΔEPCが大きい）
5. **対立仮説（contradictory）**：同じデモを説明するが、他デモと両立しない

このうち 2) と 4) が、探索と汎化を強く鍛える（hard negative の主要供給源）。

---

## 6. データ設計（学習用ログの最小スキーマ案）

ログは「再現」だけでなく「学習」にも使う前提で、最初から構造化しておく。

### 6.1 Pair / Triplet（対比学習）

例：JSONL（1行1サンプル）

```json
{
  "domain": "maze|arc",
  "anchor": { "type": "state|task|partial_program", "repr": "..." },
  "positive": { "type": "episode|macro|program", "repr": "..." },
  "negatives": [
    { "repr": "...", "kind": "near_miss|overfit|invariance_break", "weight": 1.0 }
  ],
  "gate": { "g0": 0.12, "g_min": -0.03, "theta_ag": 0.5, "theta_dg": 0.0, "ag": true, "dg": false },
  "evidence": { "failed_example_ids": ["train:2"], "reason": "mismatch_cells=7" }
}
```

### 6.2 “負例生成”の禁止事項（ARC系）

- **test output は使わない**
- negative の判定根拠は **train例** と **内部コスト/整合性** からのみ作る

---

## 7. 迷路PoCへの写像（エピソード記憶）

迷路では、各ステップで「候補（観測/記憶からのリンク）→検証→行動」が起きる。

- **Anchor**：現在状態（位置、局所観測、直近の軌跡、近傍グラフ）
- **Hypothesis**：候補リンク、近道エッジ、エピソード分割（“ここから先は別チャンク”）など
- **Positive**：
  - DGで `dg_committed_edges` に入ったエッジ
  - あるいは（設計により）「その後の成功確率が上がった」エピソード分割
- **Negative**：
  - AGで探索が開いたがDGで commit されなかった候補（hard negative の核）
  - `is_dead_end=true` に繋がりやすい候補（行動面の負例）

実験ログ（例：`experiments/maze-query-hub-prototype/results/*steps*.json`）には、`g0/gmin`、`theta_ag`、`ag_fire/dg_fire`、候補集合などが含まれるため、**負例データセットを外部教師なしで作れる**。

---

## 8. ARCへの写像（ルール/プログラム探索）

ARCでは、Hypothesis は「DSLプログラム（または部分プログラム/マクロ）」になる。

- **Positive**：
  - train の全例で完全一致するプログラム
  - かつ ΔEPC が小さい（短い/単純）ものを優先（同じ train-fit でも“良い”とする）
- **Negative**：
  - 1つのtrain例だけ外す near-miss プログラム（hard negative）
  - train-fit だが極端に冗長なプログラム（overfit）
  - 想定不変量を壊すプログラム（invariance-break）

ここでの「負例生成」は、**候補を作って照合するだけ**なので、テスト正解に依存しない。

---

## 9. `self_organizing_world_model.md` との接続（“意味空間の醸成”）

`docs/research/self_organizing_world_model.md` の要点を、この設計に落とすとこうなる。

- **対比学習**：DG確定（正例）と、DG棄却（負例）で埋め込み空間を彫る
- **不確実性（VAE的）**：エピソードを点ではなく分布（μ, Σ）として持ち、曖昧性を扱う（v2以降）
- **階層（双曲空間）**：抽象ルール（上位）と具体例（下位）の関係を、自然に表現できる可能性（任意）

最初はユークリッド埋め込み + 対比学習だけで十分で、Sleep での改善が回り始めたら拡張する。

---

## 10. 実装に使うライブラリ候補

### 10.1 最小（推奨）
- `numpy`：グリッド/特徴量
- `scipy`：連結成分、距離、軽量最適化
- `networkx`：グラフ表現（設計検証と可視化）
- `pydantic`：ログ/スキーマ（壊れない記録）
- `hnswlib`（または `faiss-cpu`）：類似検索（エピソード/タスク近傍）

### 10.2 学習（任意）
- `torch`：埋め込み学習
- `pytorch-metric-learning`：InfoNCE/Triplet/Hard-mining を実装しやすい（導入するなら）

### 10.3 幾何（任意）
- `geoopt`：双曲空間などのRiemann最適化（必要になってから）

---

## 11. MVP（まず回す最小タスク）

1. 迷路ログから、`(anchor, positive, negatives)` を JSONL に書き出す
2. もっとも単純な埋め込み（例: MLP）を対比学習で学習する
3. Wakeで「近傍エピソード提案」に使う（探索の初期候補を改善）
4. “改善したか” を、Solved率/探索ノード数/P95 で比較する

---

## References

- `docs/design/arc_prize_spec.md`
- `docs/research/self_organizing_world_model.md`
- `src/insightspike/algorithms/gating.py`
