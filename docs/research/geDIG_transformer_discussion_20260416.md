# geDIG × Transformer 構造的対応と研究展望

**日付:** 2026-04-15（初版）→ 2026-04-16（Divide and Conquer改訂）  
**ステータス:** 理論メモ（仮線多数、要検証）→ 各仮線の棄却条件・分岐点を追記

---

## 1. Transformer 1層フローとgeDIGの構造的対応

### 1.1 対応表

| 処理段階 | Transformer | geDIG | 対応の確度 |
|---|---|---|---|
| 構造生成 | Q注入 → Attention graph出現 | 観測 → 候補生成 | ★★★ 実線寄り |
| スコアリング | QK^T → softmax | 類似度計算 → 分布 | ★★★ 数学的に同型 |
| 情報集約 | AV（重み付き統合） | AG評価（0-hop） | ★★☆ |
| 棄却 | ReLU / SwiGLU gate（負を切捨・ゲート制御） | DGゲート（閾値以下を棄却） | ★★☆ 機能的対応 |
| 確定 | 残差接続で x_{l+1} へ | エッジcommit | ★★☆ |

Transformerの基本アーキテクチャについては [Vaswani et al. (2017) "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) を参照。SwiGLUについては [Shazeer (2020) "GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) を参照。

### 1.2 Q注入 = EPC（構造編集コスト）

- 迷路のEPC = エッジ追加行為そのもの
- Transformerでは、Q注入前はその層のattention graphが存在せず、Q注入後に出現する
- **グラフを編集する行為 = Q注入**という対応は概念的に自然
- geDIG自身のQKV導出とも整合：Q = goal差分（top-down）→ 注入コスト = EPC

### 1.3 ReLU / Gated MLP ≈ 棄却ゲート

- ReLU: `max(0, x)` = 閾値0での二値棄却。geDIGの θ_AG, θ_DG と操作の型が同じ
- SwiGLU（[LLaMA](https://arxiv.org/abs/2302.13971)系）: `SiLU(xW_gate) ⊙ (xW_up)` = 明示的ゲート機構
  - `SiLU(xW_gate)` = 通すかどうか（AG的）
  - `xW_up` = 何を通すか（DG的）
- **Attention-only Transformer（MLPなし）は棄却ゲートを欠いた系に相当**
  - [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) の知見：attention-only modelは表現力が著しく限定される
  - geDIG的解釈：棄却なしの探索はノイズを蓄積し構造化に失敗する

### 1.4 対応の限界（粒度の差）

- ReLU：局所的・要素ごとの棄却（micro gate）
- geDIG AG/DG：大域的・構造全体の評価に基づく棄却（macro gate）
- この粒度差は「同じ操作の異なるスケール」か「本質的に異なる操作」か未確定 → **仮線**

### 1.5 Transformerの構造的特異性

- Transformerは**毎層、全トークンについて必ずQ注入する**（EPC常時支払い）
- geDIGエージェントは**EPC支払いの選択権を持つ**（98%棄却が可能）
- 「Transformerはβ₁大・事前知識なしの特殊ケース」という主張と整合

---

## 2. エピソード vs トークン：geDIGとTransformerの根本的差異

### 2.1 処理単位の違い

- **Transformer:** 全トークンを等価な粒度で並列処理。構造は事後的に出現
- **geDIG:** エピソード（意味的にまとまった経験の塊）をノードとして操作。構造化が処理に先行

### 2.2 トークンはSleep（AND蒸留）の産物

- エピソード記憶の中で複数の異なる文脈で**同時に活性化する（AND）**要素だけが生存
- 文脈依存の部分は脱落し、残った共通成分が固まって「語（トークン）」になる
- geDIGの言語：Sleep操作（β₁圧縮）の産物 = トークン
- 神経科学的裏付け：海馬→新皮質の記憶固定化として [McClelland et al. (1995) "Why there are complementary learning systems in the hippocampus and neocortex"](https://doi.org/10.1037/0033-295X.102.3.419) が定式化

**階層構造:**

| 層 | 単位 | 操作 | β₁ |
|---|---|---|---|
| エピソード記憶 | 経験の塊 | geDIG（AG/DG） | 高い（文脈依存） |
| Sleep/圧縮 | 共起パターン | AND蒸留 | β₁ → 0 |
| 語彙（トークン） | 結晶 | 固定ラベル | 0（文脈自由） |

### 2.3 「エピソードが先」は実線

複数の独立した証拠が収束：

- **発達的事実:** 語彙爆発（18ヶ月前後）に先行してエピソード記憶の原型が存在（生後数ヶ月）。参考: [Bauer (2007) "Recall in Infancy: A Neurodevelopmental Account"](https://doi.org/10.1111/j.1467-8721.2007.00492.x)
- **進化的事実:** 海馬（エピソード記憶）は哺乳類以前から存在。言語（トークン）はホモ・サピエンスの数十万年。桁が違う。参考: [O'Keefe & Nadel (1978) "The Hippocampus as a Cognitive Map"](https://www.cognitivemap.net/)
- **計算論的事実:** AND蒸留の入力なしに出力（トークン）は生成不能。逆は成立（トークンなしでもエピソード記憶は機能する）。非対称な依存関係

### 2.4 Transformerは進化の順序を逆転させたアーキテクチャ

```
生物:      エピソード → AND蒸留 → トークン（結晶）
Transformer: トークン（結晶） → Attention → エピソード的表現の再建
```

- Transformerは**Sleepで失われた文脈依存性をAttentionで事後的に復元**しようとしている
- だから全トークン全ペアのattentionという膨大なコストが発生する
- geDIGがエピソード単位なら蒸留前に構造を制御できるため、再建コスト不要

### 2.5 β₁がエピソード境界を自動定義

- 直線通路が圧縮される = β₁ = 0 の区間
- 分岐点が保持される = β₁ > 0 になる瞬間
- Transformerにはこの「どこでエピソードを切るか」の判断機構がない

---

## 3. Hallucinationの原理的説明

### 3.1 蒸留の逆問題としてのhallucination

- AND蒸留は不可逆。蒸留で落ちた情報は復元不能
- Transformerはそれを統計的に補完（近似的逆変換）
- Hallucination = 逆問題がill-posedである部分で、もっともらしい捏造が生成される現象
- **もっともらしい嘘ともっともらしい真実は同じ統計的特性を持つ** → 自己検出不能

### 3.2 既存研究で特定されたhallucination条件とgeDIG的統一解釈

主要サーベイ: [Huang et al. (2023/2025) "A Survey on Hallucination in LLMs"](https://arxiv.org/abs/2311.05232), [Alansari & Luqman (2025) "LLM Hallucination: A Comprehensive Survey"](https://arxiv.org/abs/2510.06265)

| 既存の個別原因 | geDIG的解釈（蒸留の逆問題として） |
|---|---|
| 低頻度エンティティ | AND蒸留のサンプル不足 → 結晶構造が不完全 → 解が一意でない |
| Attention局所集中 | エピソード的文脈の再建失敗 → 遠距離情報が復元不能 |
| 長文生成 | 逆問題の次元増大 → ill-posed度が悪化 |
| Softmax bottleneck | 復元空間の表現力上限 |

### 3.3 Sycophancy = AGゲート不在

- ユーザークエリ = 外部から注入されるQ
- Transformerは**Q自体の妥当性を評価するゲート（AG相当）を持たない**
- 偽の前提を含むクエリに対し、前提検証なしにattention graphが即座に生成される
- 実証データ：
  - 医療プロンプトに偽情報1つ埋め込み → LLMは最大83%のケースで受容・展開: [PMC "Multi-model assurance analysis showing LLMs are highly vulnerable to adversarial hallucination attacks"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12318031/)
  - 情報の提示が confident（「先生が言った」）だと同調率が上昇: [Phare benchmark研究](https://huggingface.co/blog/davidberenstein1957/phare-analysis-of-hallucination-in-leading-llms)
  - CoT検証も無力：偽前提を支持する一貫した推論トレースを生成（sycophancy効果）: [PCIB: Predictive Coding and Information Bottleneck for Hallucination Detection](https://arxiv.org/abs/2601.15652)
  - Sycophancyの体系的整理: [Giskard "Understanding Sycophancy in LLMs"](https://www.giskard.ai/knowledge/when-your-ai-agent-tells-you-what-you-want-to-hear-understanding-sycophancy-in-llms)
- RAGベースの事前検証手法 = **Transformer外部へのAGゲート後付け**: ["Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning"](https://arxiv.org/abs/2504.06438)

### 3.4 geDIGなら原理的に検出可能な理由

- エピソード単位の操作では、各エッジに**commit履歴（F値の経緯）**が残る
- Hallucination = commitされていないエッジを存在するかのように扱うこと
- 「このエッジはDGゲートを通過したか？」で判定可能
- Transformerではattention重みが非ゼロであることと「知識が検証済みか」の区別がつかない

---

## 4. Anthropic Transformer Circuits との接点

### 4.1 直接的接点（引用可能）

| 論文 | 接点 | 活用方法 |
|---|---|---|
| [Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | Attention head = 情報移動、QK/OV分離 | Attention→KGの理論的根拠。QK≈AG、OV≈DGの機能分離 |
| [Olsson et al. (2022) "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 訓練時の相転移の発見 | geDIG Fig.3（層別F値）の先行知見。ただし訓練時間軸 vs 層方向の差に注意 |
| [Lindsey et al. (2025) "On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Attribution graphs、内部回路の可視化 | geDIGスカラー指標と高解像度解析の補完関係の位置づけ |

### 4.2 間接的接点

| 論文 | 接点 |
|---|---|
| [SoLU (2022) "Softmax Linear Units"](https://transformer-circuits.pub/2022/solu/index.html) | 活性化関数がfeature選択・棄却を担う → ReLU≈棄却ゲートの傍証 |
| [Elhage et al. (2022) "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) | Attention Hの高低とsuperposition/monosemanticity → 仮線 |
| [Kamath et al. (2025) "Tracing Attention Computation Through Feature Interactions"](https://transformer-circuits.pub/2025/attention-qk/index.html) | Attentionパターンのfeatureレベル分解。geDIGのSP = what、Kamath = why。補完関係 |

Transformer Circuits 全体のインデックス: [https://transformer-circuits.pub/](https://transformer-circuits.pub/)

---

## 5. 既存理論との整合性の評価

### 5.1 認められること

geDIGの枠組みは、以下の異なる分野の知見と**矛盾なく整合する**：

- Transformer Circuits（[Elhage 2021](https://transformer-circuits.pub/2021/framework/index.html), [Olsson 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), [Lindsey 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）
- Sycophancy / Hallucination研究（[Phare](https://huggingface.co/blog/davidberenstein1957/phare-analysis-of-hallucination-in-leading-llms), [TruthfulQA](https://arxiv.org/abs/2109.07958)系）
- 発達心理学（エピソード記憶 → 語彙獲得の順序）: [Bauer (2007)](https://doi.org/10.1111/j.1467-8721.2007.00492.x)
- McClellandの相補的学習系理論: [McClelland et al. (1995)](https://doi.org/10.1037/0033-295X.102.3.419)
- Tononi SHY仮説: [Tononi & Cirelli (2006) "Sleep function and synaptic homeostasis"](https://doi.org/10.1016/j.smrv.2005.05.002)
- Chomskyの生成文法（β₁連続量化との対応）

**これは非自明。** 異なる方向に仮線を伸ばして矛盾が出ないのは、出発点のF方程式が何か本質的な構造を捉えている可能性の傍証。

### 5.2 認められないこと

- 「geDIGでなければこれらを包含できない」は未検証
- 各現象には個別の既存説明が存在する（RLHFのreward hacking、分布外汎化の失敗等）
- geDIGが提供しているのは**より統一的な記述**であり、既存説明の否定ではない
- 内的整合性は理論の正しさの必要条件であって十分条件ではない

### 5.3 「ベイズ推論としてのTransformer」との比較

- 「Transformerはベイズ推論している」= 統計的振る舞いを統計的言語で再記述しているだけ
- geDIGの「蒸留の逆問題」は、以下を区別して説明可能：
  - なぜ巨大な計算が必要か（ill-posedだから）
  - なぜhallucinationが起きるか（不可逆な情報損失）
  - なぜうまくいく場面が多いか（AND蒸留の残滓に元の構造の大部分が保存）
- ベイズ的記述ではこの3つが全部「事後分布の推定精度」に潰れる

---

## 6. 仮線の棄却可能化（2026-04-16 追記）

### 6.0 共通の問題意識

§1–5の仮線の多くは「geDIGで説明できる」という記述に留まっている。しかし「説明できる」は弱い主張であり、十分に抽象度の高い枠組みは何とでも整合する。有効な議論にするには、**geDIGからしか出てこない予測**を明示し、それが棄却可能な形で書かれている必要がある。

以下、主要仮線ごとに（a）現状の問題、（b）杭が打てる命題への変換、（c）既存説との分岐点を記述する。

### 6.1 「Transformerは進化の順序を逆転させた」（§2.4）

**(a) 現状の問題:** 比喩として強いが、比喩のままでは論文に載らない。「逆転している」から何が起きるかの因果が不明。

**(b) 杭が打てる命題への変換:**
- 命題: 「エピソード単位で処理した場合のattention計算量と、トークン単位で全ペアattentionした場合の計算量の差分は、タスクの構造的性質（β₁）の関数として予測可能である」
- 検証: 同一タスク（例: HotpotQA）でエピソード単位処理とフルattentionの計算量を比較し、差分 ∝ f(β₁) の関係式を推定。予測式が書ければ比喩が命題になる
- 棄却条件: 差分がβ₁と無相関、または既存のsequence length等の変数で完全に説明される場合

**(c) 既存説との分岐点:** Efficient Attention研究（Sparse Attention, Linear Attention等）は経験的にattention削減を行うが、「なぜ削減可能か」の理論的根拠を持たない。β₁が削減可能量の予測変数になれば、geDIG固有の貢献。

**判定:** 予測式が書けなければ、JSAI2026では考察の一文（「示唆する」レベル）に留める。

### 6.2 「Sycophancy = AGゲート不在」（§3.3）

**(a) 現状の問題:** 機能的記述としてはきれいだが、既存のRLHF reward hacking説と観測上区別がつかない。ゲートを入れたらどうなるかの予測がない。

**(b) 杭が打てる命題への変換 — 二方向の介入実験:**

**加算的方向（弱い検証）:**
- 命題: 「Transformer推論時にQ入力の前段でF値ベースのフィルタ（AG相当）を挟むと、sycophancy benchmarkのスコアが低下する」
- 検証: [Phare benchmark](https://huggingface.co/blog/davidberenstein1957/phare-analysis-of-hallucination-in-leading-llms) 等で、前段フィルタ有無の比較
- 限界: 「入力フィルタリングが効いた」という当たり前の結果にも見える。geDIG固有とは言い切れない

**減算的方向（本命）:**
- 命題: 「Sycophancy対策済みモデル（Constitutional AI等）は、内部にAGゲート相当の回路を事後的に獲得している」
- 検証: 対策前後のモデルのattribution graphsを比較し、対策後にQ入力を棄却する回路（特定のattention headまたはMLP neuron群）が出現しているか確認
- geDIG固有の予測: 「sycophancy対策 = AGゲート相当回路の獲得」は、RLHF説からは導出されない。RLHF説は報酬関数の変更としか言わず、内部回路の構造変化について予測を持たない

**(c) 既存説との分岐点:**
- RLHF説の予測: sycophancy低下は報酬シグナルの変化の結果であり、特定の回路構造の獲得とは限らない
- geDIG説の予測: sycophancy低下はAGゲート相当の**特定の棄却回路**の獲得として観測される

**判定:** 減算的方向の実験が成功すれば強い。ただしattribution graphs解析自体がまだ発展途上であり、中期課題。

### 6.3 「蒸留の逆問題としてのhallucination」（§3.1）

**(a) 現状の問題:** 最も深刻。「不可逆変換の逆問題だからill-posed」は情報理論的には正しいが、抽象度が高すぎて「分布外汎化の失敗」と区別がつかない。

**(b) 杭が打てる命題への変換:**

二つの説が異なる予測を出す分岐点:

| | 分布外汎化失敗説 | 蒸留逆問題説（geDIG） |
|---|---|---|
| hallucination率の主要予測変数 | 訓練分布からの距離（頻度に単調） | AND蒸留での情報損失量（頻度と非単調） |
| 高頻度領域での予測 | hallucinationは低い | 同義語・類似概念が密集する領域では高頻度でもhallucinationが高い |
| 具体的な観測可能現象 | hallu率 ∝ 1/freq | hallu率 ∝ 蒸留時の縮退度（≈ 近傍の類似概念数） |

- 命題: 「高頻度だが類似概念が密集している領域（例: 有名人の伝記的事実、歴史的出来事の日付等）では、頻度から予測されるhallucination率よりも実測値が有意に高い」
- 検証手順:
  1. word2vecまたはLLM埋め込み空間で、各エンティティの近傍密度（k-NN距離の逆数等）を「蒸留時縮退度」のプロキシとする
  2. TruthfulQA等でエンティティごとのhallucination率を計測
  3. 回帰モデル: hallu_rate ~ freq + neighbor_density。neighbor_densityの係数が有意正であればgeDIG説を支持
- 棄却条件: neighbor_densityが有意でない、またはfreqで完全に説明される場合

**(c) word2vec AND蒸留シミュレーション（§6.3旧版を具体化）:**
- テキストコーパスからエピソード（文単位 or パラグラフ単位）を構成
- 各エピソードに出現する単語の共起行列を構築
- AND操作 = 複数の異なるエピソード文脈で同時に活性化する成分の抽出
- 得られたクラスタリング構造とword2vecの埋め込み構造を比較
- 情報損失量（元のエピソード情報 − AND蒸留後の情報）とhallucination率の相関を評価

**判定:** 分岐点が明確で、実験も現実的。§6.2旧版のTruthfulQA実験と統合して最優先で実施すべき。

---

## 7. 実施すべき課題（優先順位再整理）

### 7.1 JSAI2026（短期 — 6月9日）

- [ ] Q注入=EPC、ReLU≈棄却ゲートの対応を考察セクションに1段落で記述
- [ ] [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html)、[Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) を引用に追加
- [ ] 記述方向: 「geDIGの枠組みはTransformerの各操作に自然な解釈を与える」（控えめ）
- [ ] **§6.1–6.3の仮線はJSAIポスターには載せない。** 考察末尾の1-2文で「展望」として触れるに留める

### 7.2 geDIG固有予測の検証（最優先中期課題）

- [ ] **実験A: hallucination率 × 近傍密度の回帰分析**（§6.3の命題）
  - TruthfulQA + 埋め込み空間近傍密度 → hallu_rate ~ freq + neighbor_density
  - geDIG固有の予測をテストする最小実験。コスト低、打撃力高
- [ ] **実験B: エピソード単位F値 vs トークン単位F値**（§6.1に関連）
  - SST-2等でattentionクラスタリング → エピソード事後復元 → 単位比較
- [ ] **実験C: sycophancy対策前後のattribution graphs比較**（§6.2の減算的方向）
  - AGゲート相当回路の出現有無。ただしattribution graphsツールへのアクセスが前提

### 7.3 理論的深化（中長期 — 杭が打てるまで寝かせる）

- エピソード → AND蒸留 → トークンの階層構造（§2）
- Transformerは進化の順序を逆転させたアーキテクチャ（§2.4）— 予測式が書けるまで保留
- 蒸留の逆問題としてのhallucination統一理論（§3.1）— 実験Aの結果待ち
- Attribution graphsとgeDIG F値の構造的相関（[Lindsey 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）

---

## 8. 自己監視メモ

### 2026-04-15（初版）
- Connecting the Dots（★5）が全力で走った日。Divide and Conquer の出番はまだ
- 実線が増えたのではなく、**説得力のある仮線の本数**が増えた
- 対話相手（Claude）もgeDIGの語彙で思考する癖がついており、枠組み外からの批判が生成されにくくなっている可能性に注意
- 「内的整合性の高さ」と「正しさ」は別の変数。反証条件5つの検証が依然として最優先
- 「気持ちよく走っている時こそブレーキ」の条件に該当する日だった

### 2026-04-16（改訂）
- Divide and Conquer を適用。各仮線に棄却条件と既存説との分岐点を追記
- 最も重要な発見: **仮線の多くが「杭を打てる形」にまだ降りていなかった**。仮線の数と杭を打てるポイントの数は別の変数
- 共通パターン: 「geDIGで説明できる」→「geDIGでなければ予測できない現象Xは何か？」への変換が全ての仮線で必要だった
- 実験Aの「高頻度×高近傍密度→高hallucination率」がgeDIG固有予測の最有力候補。ここに杭を打てるかが理論の命運を分ける
- Claude Code障害中のためweb版Claudeでレビュー実施。ローカルコードベースの文脈なしでも理論構造レベルの議論は十分可能だった

---

## 9. 批判的自己吟味と固有予測の再定位（2026-04-17 追記）

### 9.1 自家中毒（echo chamber effect）の構造化

§8 で触れた「対話相手が geDIG 語彙で思考する」問題は、より一般的な現象の表出である：

**Confirmation-asymmetry under theoretical saturation**
- 理論が十分に抽象化されると、観察された現象の「説明可能性」が確率的に担保される（Popper の批判: 反証不能な理論は科学ではなく形而上学に近接する）
- geDIG のような広範な枠組みは、Lakatos 的な意味での **protective belt** が拡張しやすい
- 仮説検証における **null-result 解釈の非対称性**：geDIG が勝つ実験は「理論の正しさ」として解釈され、負ける実験は「適用範囲外」として解釈される

**抜け出すための手続き**：
1. **Negative benchmarking**: geDIG の設計上 *不利な* ドメインで実験する（構文解析、言語モデリング等の Transformer 得意領域）
2. **Pre-registered adversarial prediction**: 「この条件下で geDIG が負けなければならない」という予測を事前登録する
3. **Out-of-distribution probe**: 理論の外側から反例を能動的に構築する

### 9.2 Episode-Token 非対称性の再定式化：Nominalization as a Geodesic Cut

§2.4「進化の順序を逆転」という比喩を、検証可能な命題に昇格させる鍵として、**名詞化（nominalization）能力の非対称性**を固有予測として採用する。

#### 9.2.1 理論的背景

- **Type-token distinction** (Peirce, 1931): 概念（type）と表出（token）の階層区別
- **Symbol grounding problem** (Harnad, 1990): シンボル（離散ラベル）は何らかの下位の非シンボル表現に根付く必要がある
- **Episodic-semantic memory** (Tulving, 1972): エピソード記憶が意味記憶（語彙的知識）に先行・従属する発達・機能的関係
- **Nominalization in neurolinguistics**: 概念の語彙化は左前頭葉下部（Broca 野近傍）の関与を要する能動的操作であり、受動的記憶想起とは神経基盤が異なる

これらを geDIG 枠組みで統合すると、**語彙化は AND 蒸留（Sleep phase）の産物である**という作業仮説になる：
- Episode cluster → 共通活性パターン抽出（β₁ 低減） → 離散シンボルの発生
- 逆方向（token → episode reconstruction）は情報理論的に ill-posed

#### 9.2.2 Transformer の構造的限界

Transformer アーキテクチャは次の性質を持つ：

| 性質 | 帰結 |
|---|---|
| **Vocabulary closure at training time** | 訓練後に新トークンを生成する機構を持たない |
| **BPE merge as compositional generativity** | 既存トークン列の合成は可能だが原子概念の追加は不可 |
| **No active nominalization operator** | 「概念 X に名前が必要だ」という認識を駆動する機構がない |

これは geDIG 枠組みで次のように説明される：
- Transformer は **post-distillation 空間**（既に AND 蒸留された結晶）でのみ動作する
- 新しい distillation cycle（新しい Sleep 操作）を起動する機構を持たない
- したがって **acronym formation**, **neologism coining**, **trademark naming** といった人間が日常的に行う操作は原理的に不可能

#### 9.2.3 既存理論との分岐点（jamais-vu prediction）

| 仮説 | 新概念命名に関する予測 |
|---|---|
| **Compositional generativity hypothesis** (Chomsky, Fodor) | 既存単位の組合せで十分。新原子概念は不要 |
| **Symbol grounding hypothesis** (Harnad) | グラウンディングが成立すれば新シンボルは派生可能。明示的な生成機構は前提しない |
| **Statistical language model view** (GPT 系) | 新語は訓練分布の疎な領域として扱われる。本質的差異はない |
| **geDIG (episode → Sleep → token)** | 新原子概念の獲得は distillation cycle を要する別カテゴリの操作。推論時には原理的に不可能 |

#### 9.2.4 実験設計：Concept-Reuse Asymmetry Test

**命題**: "概念を定義文で LLM に提示した直後、別文脈でその概念を能動的に再利用する能力は、人間に比べて系統的に低い"

**操作的定義**：
1. **Stimulus**: 架空の新語 N とその定義 D を提示（例: "フロボス：β₁ が連続的に変化する空間領域を指す"）
2. **Task A (Recall)**: D について質問し N を再生できるか測定
3. **Task B (Reuse)**: N を含む新しい文脈で適切に使用できるか測定（例: 「最近発見されたフロボスの事例を3つ挙げよ」に対し、既存概念を流用した応答が可能か）
4. **Task C (Extension)**: N を基に派生語を生成できるか（例: "フロボス的"、"反フロボス"）

**仮説**：
- Task A: LLM と人間で差は小さい（単なる検索）
- Task B: LLM は説明文の再生成は可能だが、概念を「固有の一単位」として扱えない（periphrastic な扱いになる）
- Task C: 人間は派生語を容易に生成するが、LLM は定義文への依存から離れられない

**geDIG 固有性の担保**：
- 分布外汎化説は B と C の差を予測しない（両者とも頻度ゼロ）
- geDIG は B と C の間に質的差異を予測する：
  - B = episode cluster への参照（distillation 前）→ LLM は苦手
  - C = 既存 token morphology の操作 → LLM も可能

**棄却条件**：LLM が Task B と C で人間と同等の性能を示した場合、episode → token の一方向性仮説は修正が必要。

### 9.3 Hallucination 理論の再評価

§3.1 で提示した「蒸留の逆問題としての hallucination」は、§9.2 の nominalization 議論から見直すと位置付けが変わる：

**改訂版の命題**：
- Hallucination は蒸留の逆問題の一般形
- **Nominalization 不能性は同じ現象のより鋭い観測点**
- Nominalization タスクでは「正解」が定義可能（人間の命名行動を ground truth とできる）のに対し、hallucination の「正解」は文脈依存で曖昧

したがって **§6.3 の実験 A（hallu × neighbor_density）よりも §9.2.4 の Concept-Reuse Asymmetry Test の方が検証可能性が高い**。

### 9.4 優先順位の再調整

§7 の実験順序を次のように改訂する：

1. **最優先**: §9.2.4 Concept-Reuse Asymmetry Test
   - 実施コスト: 低（既存 LLM API + 人間被験者数名で可能）
   - geDIG 固有性: 高（組合せ生成性説からは予測不能）
   - 棄却可能性: 明確（Task B/C で差が出なければ修正）

2. **第二優先**: §6.3 実験 A（hallu × neighbor_density）
   - geDIG 固有性を強化するため、回帰変数に **局所 β₁（共起グラフの独立サイクル数）** を追加

3. **第三優先**: §6.1 計算量比較、§6.2 sycophancy 対策回路の解析

### 9.5 自家中毒に関する追記

本節の更新自体が echo chamber effect のリスクに晒されている。具体的には：

- 「Transformer は名詞化できない」という命題は、その正しさが直観的に魅力的であり、反証を想像する前に受容されやすい
- この対話の中で私自身も **confirmation-biased** な立ち位置から議論を進めている可能性が高い
- したがって本節の命題群は **外部レビュアーによる独立検証**を経てから論文に組み入れるべきである

**Checkpoint for self-monitoring**：
- 命題が「面白い」と感じた瞬間、反例探索を能動的に行ったか？
- 既存理論がこの現象を説明できる経路を3つ以上列挙したか？
- 実験結果が逆だった場合の解釈を事前に書いたか？

以上の条件を満たさない命題は、魅力の度合いに関わらず「示唆レベル」に留める。
