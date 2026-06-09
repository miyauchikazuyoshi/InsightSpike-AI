# JSAI 2026 ポスター発表 — 想定 Q&A と基礎概念解説

ポスター + 論文を初見で見た来場者から想定される質問を、段階別に整理。
回答は 30 秒〜2 分程度で口頭返答できる長さを目安にする。

---

## レベル 0 — 用語・記号の即答 (10 秒以内)

| 来場者の質問 | 即答 |
|---|---|
| geDIG って何の略？ | **graph edit Distance and Information Gain**。グラフ編集距離と情報利得を一つにまとめた指標。 |
| ΔEPC って？ | **Edit Path Cost**。ノード・エッジを編集するコストの正規化値。GED の局所近似と思って良い。 |
| ΔH って？ | シャノンエントロピーの正規化差分。**不確かさの変化量**。 |
| ΔSP って？ | 平均最短路長の相対短縮。「**新しい近道ができたか**」を測る。 |
| λ, γ は？ | 重み。λ = 構造コスト vs 情報利得のバランス、γ = エントロピー vs 経路短縮のバランス。**情報温度**として解釈。 |
| AG / DG は？ | **A**ttention **G**ate（0-hop で違和感検知）、**D**ecision **G**ate（multi-hop で近道確証）。 |
| β₁ は？ | グラフのトポロジー指標。**独立サイクル数**。論文の今後の展望で SP を置換する候補。 |
| FEP は？ | **F**ree **E**nergy **P**rinciple。Friston の自由エネルギー原理。脳は予測誤差を最小化するという仮説。 |
| MDL は？ | **M**inimum **D**escription **L**ength。最小記述長原理。良いモデル = 短く記述できる。 |
| PSZ は？ | **P**erfect **S**caling **Z**one。Acc≥95% かつ FMR≤2% かつ追加遅延 P50≤200ms の運用帯。 |

---

## レベル 1 — 基礎概念の素朴な質問 (30 秒以内)

### Q1. 結局、この研究は何を解いているの？

**動的に成長する知識グラフで「いつ新しい情報を受け入れるか」を判定する関数を作った**。

GraphRAG や Self-RAG は「何 (What) を取るか」を最適化してきたが、
「いつ (When) 統合するか」の規範がなかった。本研究はその規範を F として与える。

### Q2. F = ΔEPC − λ(ΔH + γΔSP) の直感は？

**「変えるコスト」と「得る情報量」のトレードオフを一本のスカラーにする**。

- ΔEPC が大きい = 構造をたくさん変える → コスト高
- ΔH+γΔSP が大きい = 情報がたくさん得られる → 利得高
- 利得 > コスト なら F < 0 → 統合採用
- 利得 < コスト なら F > 0 → 棄却

### Q3. なぜスカラー 1 個に潰すの？情報損失しない？

**判断（採用 / 棄却）は二値だから、最終的に 1 スカラーが必要**。

各成分（EPC, H, SP）は内部で別計算している。最後の判定段階で 1 にする。
診断したいときは内訳を見れば良い（ポスターの component chart）。

### Q4. AG/DG ゲートが 2 段階あるのはなぜ？

**計算コストの段階分担**。

- AG (0-hop): 仮配線直後の F だけ見る。安い。「明らかに棄却 / 明らかに採用」をここで切る。
- DG (multi-hop): 数歩先まで辿って「本当に近道？」を確認。高い。AG で迷ったケースだけ走らせる。

これにより**大多数のクエリは 0-hop で決着**、F 評価の総コストを抑えられる。

### Q5. β₁ って何？なぜ SP の代わりに使いたいの？

β₁ は **グラフ理論の位相不変量**で、独立サイクル数を数える。

- SP: 計算量 O(N²)、グラフのサイズに依存する量。
- β₁: 計算量 O(V+E)、トポロジカルな性質で正規化不要。

ポスターの 25×25 maze 60-seed では、β₁ と SP の成功率は p=0.69 で有意差なし、
**メモリは 88% 削減**。論文の今後の展望に書いた。

---

## レベル 2 — 比較・差別化質問 (1 分以内)

### Q6. GraphRAG と何が違う？

| | GraphRAG | geDIG |
|---|---|---|
| 主眼 | What (何を検索) | When (いつ統合) |
| グラフ | 静的スナップショット | 動的更新 |
| 拒否基準 | なし or ヒューリスティック | 構造的に F < θ_DG |

GraphRAG はコミュニティ検出が強み。geDIG は「採用 / 棄却 / 保留」を原理的に判断する層を足す。
**直交関係で、組み合わせ可能**。

### Q7. FEP / Active Inference と何が違う？

FEP は **変分自由エネルギー**（連続的な確率分布で定義）を最小化する。
geDIG はそれを **離散グラフ操作に翻訳した operational なバージョン**。

- FEP の「複雑さ」 ↔ ΔEPC（構造編集コスト）
- FEP の「精度」 ↔ ΔH（不確かさの減少）
- FEP の「ベイズ更新」 ↔ DG ゲート確定

**仮定 (B1-B4) の下で F ∝ ΔMDL + O(1/N) という operational な対応を主張**。
厳密な数学的同型ではなく、設計の理論的バックボーン。

### Q8. MDL と何が違う？

MDL = L(M) + L(D|M) を最小化。
geDIG は：
- ΔEPC ↔ ΔL(M)（モデルが複雑になる）
- ΔH + γΔSP ↔ −ΔL(D|M)（データが短く書ける）

つまり F = ΔEPC − λ(ΔH+γΔSP) ∝ Δ[L(M)+L(D|M)] = ΔMDL（仮定下）。

### Q9. 何が新しいの？単なる組み合わせでは？

**3 つの異なる数学構造を 1 スカラーに統一した点**：

- ΔEPC: メトリック（距離空間）
- ΔH: 測度（確率空間）
- Δβ₁ / ΔSP: トポロジー（位相空間）

それぞれは既存研究で扱われている（Hewitt&Manning, Entropy-Lens, Oyama et al）。
**3 者を独立な軸として組み合わせる点と、operational な FEP/MDL ブリッジを与える点が新規**。

### Q10. SOTA 比較は？

ポスターでは SOTA 比較ではなく **F の原理検証** が目的。

- maze で F が「迷ったら探索、近道見たら統合」を駆動できることを示す
- Transformer の attention で F が層ごとに相転移するパターンを示す
- F 正則化で精度が悪化しない（むしろ改善傾向）

SOTA 比較は v7 論文（共同研究歓迎）で。

---

## レベル 3 — 技術的深堀り質問 (1-2 分)

### Q11. ΔEPC の具体的な計算式は？

```
ΔEPC = Σ (operation_cost) / C_max
```

- 各編集操作（ノード追加 = node_cost, エッジ追加 = edge_cost）の合計
- 正規化定数 C_max は候補集合のサイズで決まる
- 詳細は論文 §3.5、コードは `experiments/maze/qhlib/evaluator.py`

### Q12. ΔSP は重そう。実用上どうしてる？

3 つの工夫：

1. **固定ペアサンプリング**: 全ノードペアではなく一部だけ
2. **AG ゲート前置**: 0-hop で棄却したら multi-hop に行かない
3. **β₁ への置換** (今後): O(V+E) で済む

ポスターの 25×25 60-seed では、β₁ で **メモリ 4.3GB → 500MB** に削減。

### Q13. Transformer 実験で「Layer 0-1 vs 10-11 で p<10⁻⁸⁰」って何を比較してる？

BERT-base 200 sample で各層の F 値を測定し、
**初層 (L0-L1) と最終層 (L10-L11) で F の分布が違うか** を対応 t 検定。
Cohen's d = 2.31 で効果量も大きい。

要するに「層を進むと F が系統的に上がる」= **相転移パターン**を統計的に示している。

### Q14. F 正則化の +0.33pp って小さくない？

小さい。**統計的有意 (p=0.038, n=3 seeds) だが効果量も控えめ**。

ポスターの主張は「F 正則化が精度を悪化させない」のレベル。
強い主張ではなく「F が attention の何かを捉えている」の傍証。

### Q15. 迷路 98% の具体的なベースラインは？

| 手法 | 成功率 | ステップ数 |
|---|---|---|
| Random Walk | 45% | 210 |
| Greedy DFS | 92% | 85 |
| **geDIG** | **98%** | **69** |
| Oracle BFS (上限) | 100% | 40 |

geDIG は **AG/DG 判定だけで Oracle の 1.7 倍まで来ている**。

### Q16. 「Sleep フェーズ」って論文にあったけど何？

論文の §4.5 と今後の展望。
**Wake (探索/評価) と Sleep (グラフ整理) を交互に回す設計**。

- Wake: 環境からのエピソード注入 → F 評価 → AG/DG → 統合
- Sleep: 入力遮断、過去のグラフを最適化（β₁ 圧縮、冗長エッジ削除）
- 海馬リプレイ / Tononi SHY 仮説とのアナロジー（operational metaphor）

JSAI 後の実装課題。

---

## レベル 4 — 妥当性・批判への応答 (重要)

### Q17. 内的整合性が高いだけで、本当に正しい保証は？

**正しい指摘**。内的整合性は必要条件であって十分条件ではない。

本研究は以下のスタンス：
- 形式化と内的整合性は提示した
- 反証可能な実験は迷路と Transformer の 2 つ
- **広くレビューと反証検証を求めている**（これがポスター投稿の主目的）

### Q18. 仮定 B1-B4 が成立しない場面では F は意味ある？

成立しない場面では理論的保証はない。

- B1: 局所有界性が崩れる → 評価値が発散する可能性
- B3: エントロピー推定の分散が大きい → ΔH が信頼できない
- 大規模・スパースグラフでの実証は今後の課題

論文では **適用範囲を明示**して「PSZ 外では性能保証なし」と書いている。

### Q19. 「相転移」って言うけど物理の相転移と等価？

**比喩**。物理的相転移とは数学的に等価ではない。

- 物理: 自由エネルギーの 2 階微分が不連続
- geDIG: F 軌跡が層によって qualitatively 異なるパターンを示す

「相転移と読める」レベルの metaphor として位置付けている。
論文 §8 で operational analogy であると明記。

### Q20. 結局個人プロジェクトでスケール限界では？

正直に：その通り。

- 検証規模: BERT-base (110M)、GPT-2 medium (355M) まで
- 70B+ は GPU リソース不足で未検証
- **共同研究を歓迎**（ポスターと論文末尾に明記）

### Q21. ポスターと論文で言ってる「Unified Gauge」って強すぎでは？

**自家中毒のリスクは認識している**。

- 同じ F の式が異なるドメインで動作した、というレベル
- 「すべての知識グラフを統一する究極理論」とは主張していない
- 詳細は research メモ `docs/research/geDIG_transformer_discussion_20260416.md` の §9

### Q22. F 正則化が学習で機能する理由を Negative の方が良い、というのは反直感的

反直感だが Exp4 で再現：

```
Baseline (CE only):           88.07%
Positive (CE + F最小化):       87.16% ← 悪化
Negative (CE + F最大化):       89.45% ← 改善
```

仮説：F 最大化 = DG（位相的多様性）を保持する正則化として機能。
attention の「冗長サイクル」を壊さない方が学習が良い。
**ただし強度の探索 (β sweep) と再現実験が必要**。

---

## レベル 5 — 実装・次の一歩

### Q23. コードはどこ？

- GitHub: <https://github.com/miyauchikazuyoshi/InsightSpike-AI>
- Apache 2.0 ライセンス
- DOI: 10.5281/zenodo.19454110
- ポスター隣でデモも動かしてる（Transformer F-trajectory）

### Q24. 自分のグラフで試したい

`networkx` グラフがあれば走る：

```python
from insightspike.algorithms.gedig_core import GeDIGCore

core = GeDIGCore(lambda_weight=1.0)
result = core.calculate(g_before, g_after)
print(result.f_value)
```

`spaces/transformer-f-trajectory/` のデモもラッパー例として参照。

### Q25. β₁ の実装はいつ完成？

すでに maze で実装済（`experiments/maze/qhlib/evaluator.py` の `--sp-mode betti1`）。
60-seed 比較で SP と有意差なし、メモリ 88% 削減を確認。
論文への正式組み込みは v7 で。

### Q26. JSAI の後はどう進める？

3 軸：

1. **理論精緻化**: β₁ への完全置換、Sleep フェーズの実装
2. **スケール検証**: 大規模モデル（GPT-2 large、Llama 7B）での F 軌跡
3. **応用**: AGHT (Analytical Heterogeneous Graph Transformer) を BRIGHT 等で SOTA に近づける

**共同研究の窓口は GitHub Issues か mail**。

---

## ポスターを見せる流れ (来場者対応のテンプレ)

### 最初の 10 秒
> 「動的グラフで『いつ統合するか』を一つのスカラー F で判定する枠組みです。
> 同じ F が迷路と Transformer の両方で動くのを示しました。」

→ ポスター上段の **F 式** と **Maze/Transformer の数値** を指差す

### 興味を持ったら 30 秒
> 「F = ΔEPC − λ(ΔH+γΔSP)。
> 構造を変えるコスト vs 情報利得のバランス。
> AG が違和感を検知、DG が近道を確証する 2 段ゲートで効率化。」

→ ポスター §2 (F式) と §4 (AG/DG フロー) を指差す

### 突っ込みが入ったら
- 「FEP との対応は operational な橋渡しで、厳密な同型は主張していません」
- 「効果量は小さいですが、F が機能している方向性が見えたという段階です」
- 「共同研究歓迎です、ぜひコンタクトを」

### デモを見せたい
ポスター隣の PC で `streamlit run app.py` 起動。
プリセット文を切り替えて F 軌跡の相転移パターンを見せる。
1 分で「同じ F が層ごとに動いている」を視覚化できる。

---

## デモ説明スクリプト

ポスター隣で動かす 2 つのデモの口頭説明。30 秒・1 分・2 分の 3 段構え。

### Demo A — Transformer F-Trajectory (Streamlit)

**起動**: `cd spaces/transformer-f-trajectory && streamlit run app.py` → http://localhost:8501/
**URL ハンドアウト用**: <https://github.com/miyauchikazuyoshi/InsightSpike-AI/tree/main/spaces/transformer-f-trajectory>

**何を見せるデモか**:
論文 §3 (検証 II: 意味的 KG としての Transformer) で採用された **Pattern C — Attention-graph F per (layer × head)** の再現。Real attention vs Random attention の対比、深層方向の相転移、ヘッド多様性を visualise する。

論文の核心数値 (Real F ≈ -0.43, Random F ≈ -0.52, ΔF ≈ +0.08, win rate 90.5%, paired t(199)=32.6, p<10⁻⁸⁰) が全プリセットで再現される。

#### 30 秒バージョン (来場者がチラ見した時)

> 「これは **学習済み BERT の Attention 行列をグラフ化** して、**F = ΔEPC − λγΔSP − λΔH** を **層 × ヘッド** ごとに計算したものです。
> **橙線が実 Attention、灰線がランダムベースライン**。実 Attention が常に上(less negative)、深層で 0 に近づく ＝ 論文の **相転移パターン** です。」

→ `simple_1` を開いて、左の「Real vs Random」折れ線で橙が常に灰の上、深層で接近を指差す。

#### 1 分バージョン (関心を示した時)

> 「**ポスターの式と同じ F**を、BERT の attention 行列で計算しています。
>
> 左の折れ線: **Real F (橙) vs Random F (灰、破線)**。Random は -0.50 付近で平坦、Real は -0.47→-0.40 と深層で 0 に近づきます。これが論文の **Layer 0-1 vs 10-11 で p<10⁻⁸⁰** で示された相転移。
>
> 右のヒートマップ: **(層 × ヘッド) 144 セルの F**。同じ層でもヘッドによって F が違う(明=構造的、暗=ランダム的)。論文 §3.4 の **ヘッド多様性** です。」

→ Real vs Random チャートで深層の橙が灰に接近する様子、ヒートマップで明暗のパターンを指差す。

#### 2 分バージョン (技術者が深堀りしてきた時)

> 「**プリセットは事前計算 (λ=0.5, γ=0.5, 上位 10% パーセンタイル)** で、論文 Phase 1 score_full.json の数値域を全 12 文で再現確認しています。
>
> 数値カードに『**paper Phase 1: -0.43**』『**paper: 90.5%**』と論文値を併記しているので、来場者は再現性を一目で確認できます。
>
> **Custom input タブ** で来場者の文をその場で BERT に通せます。1-2 秒。Sidebar で λ, γ, percentile を可変。
>
> なお、論文採用は **このパターン C** だけです。観察実験としては他にも 2 種 (hidden state 軌跡、Pythia 学習動態) を試しましたが判定基準未達で送りました。詳しくは下の Q&A で」

→ Sidebar スライダー操作 + Compare タブで 2 文を重ね表示し、文ごとの違いを示す。

#### 見せる順番のテンプレ

| ステップ | 操作 | 何を強調するか |
|---|---|---|
| 1 | `simple_1` "The cat sat on the mat." を開く | **Real (橙) > Random (灰)、深層で接近** = 相転移の最小例 |
| 2 | カードの「Real F = -0.42」を指差す | 「論文 Phase 1: -0.43」横並びで再現性 |
| 3 | Heatmap を指差す | (層×ヘッド) 144 セル、**ヘッド多様性** |
| 4 | `complex_1` 長文に切替 | 文長が違っても **同じ Real > Random パターン** |
| 5 | `garden_path_1` に切替 | ΔF が他より小さい仮説的観察 |
| 6 | Compare タブで simple_1 vs garden_path_1 | **同じ式 F が文で違うカーブ** = F の感受性 |
| 7 | Custom input に来場者の文 | 「自分の文でも 1 秒で動きます」 |

#### よく出る質問への返答 (Demo A 固有)

| 質問 | 即答 |
|---|---|
| F は小さい方が良いのでは？ | 迷路 / RAG (採択判断) では小さい F が良い。本デモは論文 **§3 (推論観察)** の再現で、F は 0 に近づくほど「構造化されている」と読む。`F が負値で測定され、Real > Random = 構造的優位` の論文構成。 |
| 論文の数値と合っている？ | **はい**。Phase 1 score_full.json の数値域 (Real F ≈ -0.43, Random F ≈ -0.52, ΔF ≈ +0.08, win 90.5%) を全プリセットで再現確認済み。 |
| なぜ深層で F が 0 に近づく？ | Attention 分布が集中化 → ΔH (エントロピー) 減少。エッジ密度 ΔEPC も下がるが ΔH の効果が支配的。論文の「相転移」解釈。 |
| ΔF が文によって違うのは？ | Real attention の構造化度合いが文の内容で変わるため。Garden-path のように構文困難な文では ΔF が小さい傾向 (= 構造化の利得が少ない) 仮説。 |
| 他のモデルでは？ | DistilBERT (6 層) も切替可。GPT-2 も研究コード extract_and_score.py で実証済み (paper Table 1)。 |
| 推論時間は？ | CPU で 1-2 秒、Apple Silicon mps で 0.3-0.7 秒。 |
| 学習との関係は？ | 論文 §4 (F 正則化) では α=0.001 で +0.33pp の精度向上。デモは推論時 (観察) の話で別。後続の exp4 で **F 最大化が +1.4pp** という再現結果が出ています (詳細は次の Q&A)。 |

---

#### 後続実験を聞かれた時 (重要 — ポスターには載っていない情報)

ポスターに載っているのは **Pattern C (観察)** と **Phase 5 (F 最小化正則化)** の 2 つだけ。
実際には **観察 3 パターン + 介入 3 段階** をやっている。来場者から「他の実験は?」と聞かれた時の対応:

##### 観察実験 3 パターン

| Pattern | やったこと | 結果 | 論文採用 |
|---|---|---|---|
| **A** (`inference_f_trajectory`) | 言語注入時の **hidden state** で層ごとに F を観測 (8 モデル) | U字 (bathtub)、`delta_r2_struct > 0` 未達 → "initial signal" 止まり | ❌ |
| **B** (`pythia_checkpoints`) | **学習過程** の attention で F を観測 (Pythia step 0→143k) | H -54%, EPC +30%, \|ΔF\| 単調収束、step 64-512 で **相転移**発見 | ❌ (Causal LM の SP 制約) |
| **C** (`extract_and_score.py`) | **学習済モデル** の attention で F を観測 (BERT, GPT-2 計 2304 (層×頭×文)) | p<10⁻⁸⁰, d=2.31, 深層方向の相転移 | ✅ (§3) |

##### 介入実験 3 段階

| 段階 | やったこと | 結果 | 論文採用 |
|---|---|---|---|
| **Phase 3** (`intervene_eval.py`) | 推論時に attention を改変 (noise, sparsify) | どんな介入でも 97%→47% 一律崩壊 = **F 特有性示せず** | ❌ |
| **Phase 5** (`train_f_regularized.py`) | 学習時に F **最小化** 正則化 (CE + βF) | α=0.001 で +0.33pp, p=0.038、ただし Phase 4 で random 正則化に負ける | ✅ (§4) |
| **exp4** (`thermodynamic_gedig.py`) | 学習時に F **最大化** 正則化 (CE − βF) | baseline 88.1% → **89.4% (+1.4pp)**, SP/β₁/統一公式 3 回再現 | ❌ (v7 へ) |

##### Q&A 想定

| 質問 | 即答 |
|---|---|
| **Hidden state で同じことやったら?** | Pattern A としてやりました。8 モデル全てで `delta_r2_struct > 0` の判定基準を満たせず "初期シグナル" 段階で停止。深層方向の相転移を見たいなら attention graph (Pattern C) の方が clean に出ます。 |
| **学習過程は観察しないの?** | Pattern B として Pythia の checkpoint 7 点を観察しました。step 64-512 で相転移、H が 54% 減を確認。ただし **Causal LM (GPT 系) では SP 定義が機能しない** (CLS なし) ため、F の 3 項が揃わず論文未掲載。代替 SP の実装が次の課題です。 |
| **Pattern A/B はなぜ落とした?** | 統計的判定基準 (Pattern A は `delta_r2_struct > 0`, Pattern B は 3 項揃った F の単調性) を満たせなかったため。Pattern C は p<10⁻⁸⁰, d=2.31 と clean に出たので採用。**負の結果は誠実に切り分けています**。 |
| **介入実験は?** | やりました 3 段階。Phase 3 (推論時介入) は 50% 崩壊で F 特有性示せず。Phase 5 (F 最小化学習) は +0.33pp の小さい効果。後続の **exp4 で方向を逆転** (F 最大化) させたら **+1.4pp** が SP/β₁/統一公式の 3 度の独立再現で得られました。これが AGHT 設計に直結し、HotpotQA で EM 40→48.9% まで来ています。 |
| **論文採用と未採用の境界は?** | **再現性 + 統計的有意性**。Pattern C は 2304 サンプル、p<10⁻⁸⁰。Phase 5 は 3 シードで p=0.038。Pattern A/B/Phase 3 は判定基準未達のため "今後の課題" 扱い。 |
| **exp4 を v7 に入れる予定?** | はい。SP 版・β₁ 版・統一公式版で 3 度再現できた "negative_better" は AGHT の理論基盤になっています。v7 §7 (Transformer F-regularization) で正式に位置付けます。 |
| **次の Transformer 単体実験は?** | **Grokking 相転移時の β₁ + curl 同時観測** (H_grokking-curl) を計画中。Power et al. 2022 の grokking 現象と Özönder 2025 (Ising/BKT) と並走する位置付けです。 |

---

### Demo B — Maze Interactive Playback (HTML 単体)

**起動**: ブラウザで以下のローカルファイルを開く
```
file:///Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/results/paper_25x25_s500_allpairs_exact_interactive.html
```

22 MB の自己完結 HTML。ネット接続不要。`Maze Report v-SPfix-2` というタイトル。

**画面構成** (上から順に):
- 上段の指標カード: Success Rate / Average Steps / Mean g₀ / Mean g_min / k★ Mean / k★ ≥ 1 Ratio / Multihop Usage
- 左側 Controls: Seed セレクタ / Edges モード / Step スライダー
- 右側: Maze Playback & Graph Integration、Graph Snapshot、Temporal Metrics
- 下部: SP Debug、Candidate Snapshot、Per-hop Metrics

#### 30 秒バージョン

> 「これは **25×25 迷路を 500 ステップ走らせた実験**を、シードごとに再生できるインタラクティブビューワです。
> 上のカードが集計指標。**Success Rate が 98%、Average Steps が約 69** — これがポスターの迷路結果と同じ数字です。
>
> 下のスライダーで **エージェントが各ステップで何を見て F を計算したか** をステップ・バイ・ステップで遡れます。」

→ Step スライダーを左右に動かして迷路上のエージェント軌跡が動くのを見せる。

#### 1 分バージョン

> 「シードを切り替えると別の迷路インスタンスが見えます。
>
> 重要な点は 3 つ：
>
> **(1) g₀ / g_min の時系列**: 各ステップで AG (0-hop) と DG (multi-hop) の F 値を出している。グラフの折れ線で、g_min が g₀ より十分下がった瞬間が **DG 発火 = 統合の確信**。
>
> **(2) k★ ≥ 1 Ratio**: マルチホップ評価が実際に発火した割合。**大半は 0-hop で決着し、必要なときだけ multi-hop に降りる** ことが分かります。これがポスターでいう『計算量の段階分担』の実証。
>
> **(3) Per-hop Metrics**: 各ホップでの ΔEPC, IG, H, ΔSP の内訳。F が単一スカラーで判定しているのが見えます。」

→ Temporal Metrics チャートと Per-hop Metrics テーブルを順に指差す。

#### 2 分バージョン

> 「Edges モードを切り替えると、エージェントの内部グラフが時間とともにどう成長していくかが見えます。**Wake → Sleep → Wake** のサイクル設計の Wake 部分にあたります。
>
> Candidate Snapshot で、各ステップで **AG エリア内のどのエッジを候補として吟味したか** を可視化しています。多くは棄却される。**98% の候補棄却率と 98% の到達成功率を両立しているのが geDIG の特徴**です。
>
> SP Debug は **δSP が最大の候補トップ**を表示。ループ短絡を作る候補が **どの瞬間に検知され DG 発火に繋がったか** をデバッグできます。
>
> このビューワは生データを直接埋め込んだ自己完結 HTML なので、論文の Figure を **追試・反証可能な形** で公開する手段でもあります。査読してもらえる方には URL ごとお渡しします。」

→ 各 Debug セクションを順に指差し、データの埋め込みであることを強調。

#### 見せる順番のテンプレ

| ステップ | 操作 | 何を強調するか |
|---|---|---|
| 1 | 起動直後の Default seed を表示 | 上段カードで「ポスターと同じ数字」 |
| 2 | Step スライダーを最初から最後へ動かす | エージェントの軌跡 = AG/DG の動作シーケンス |
| 3 | Temporal Metrics チャートを指差す | g₀ と g_min の差が大きい瞬間 = DG 発火 |
| 4 | Per-hop Metrics に降りる | F の内訳。単一スカラーがどう構成されるか |
| 5 | Seed を変えて別インスタンス | 一貫性の demonstration |

#### よく出る質問への返答 (Demo B 固有)

| 質問 | 即答 |
|---|---|
| 学習なし？ | **学習なし**。F だけで AG/DG 判定して 98%。Greedy DFS (92%) を上回る。 |
| 何 seeds で取った？ | 25×25 で 100 episodes (500 steps cap)。SP 計算は allpairs exact。 |
| Random Walk との差は？ | Random 45%、Greedy DFS 92%、geDIG 98%、Oracle BFS 100%。 |
| 大きい迷路では？ | 50×50 も実証済 (ポスター記載)。15×15 から 50×50 までスケールする。 |
| この HTML どうやって作ってる？ | 実験コードが各 episode の生データを JSON 埋め込みで吐く。view は plain JS。 |

---

### 両デモ共通の運用 Tips

- **WiFi を切ってもデモ動く**: Transformer demo は cache 済み presets と local BERT、Maze demo は self-contained HTML。
- **Laptop を 2 台用意**: 1 台で交互に動かすと切替コストが大きい。**Transformer は Streamlit on Mac A、Maze は HTML on Mac B** がベター。1 台しかない場合は ブラウザの **タブ 2 枚** で切替。
- **デモ前にプリセット計算済みか確認**: `presets.json` の更新日が古いと不安。前夜に `python compute_presets.py --device mps` を回しておく。
- **来場者の文を Custom input に入れる時**: Personal Information (本名・所属) を入れないよう一言添える。BERT に入っても問題ないが、配慮として。
- **質問が深堀りに入ったら QR/URL を渡して終わる**: Landing page (https://miyauchikazuyoshi.github.io/InsightSpike-AI/) に paper PDF と GitHub があるので、そこから自分のペースで読んでもらう。

---

## 注意：自分への戒め

- **論破モードに入らない**。来場者の批判は学びの種。
- **個人プロジェクト感を売りにする**。「在野で AI と協働して詰めた」は強みでもある。
- **共同研究の窓口を開いておく**。これがポスター投稿の主目的の一つ。
- **強すぎる主張をしない**。「示唆する」「対応する」「示唆的に」を多用。
