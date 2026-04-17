# geDIG 認知・推論アーキテクチャ（統合版）

**最終更新**: 2026-04-17  
**ステータス**: 骨格作成済。本文統合は順次。  
**Part**: 2（認知・推論アーキテクチャ）  
**前提**: [Part 1 コア理論](gedig_core_theory_unified.md) §7（Wake-Sleep-Wake + 演繹的 NN）  
**正準参照**: `docs/gedig_spec.md`

> **この文書の位置づけ**: Part 1 §7 で導出された Wake-Sleep-Wake + AG/DG ゲート設計に、
> **認知科学的基盤**・**curl 検出**・**エピソード起点論**・**自律的発見機**の認識論的肉付けを与える。
> 素材は `gpt_bert_gedig_perspective.md` / `thinking/gedig_cognitive_foundation.md` / 
> `thinking/gedig_prediction_curl.md` / `thinking/gedig_action_definition.md` / 
> `thinking/gedig_autonomous_discovery_machine.md` / `thinking/gedig_cognitive_steam_engine_20260306.md` /
> `thinking/gedig_triangular_contrastive_learning.md` / `thinking/spiral_agdg_flow.md` の 8 本。

---

## 目次

1. [はじめに — Part 1 からの導入](#1-はじめに--part-1-からの導入)
2. [エピソード起点論と AND 蒸留](#2-エピソード起点論と-and-蒸留)
3. [curl 検出の階層的定義](#3-curl-検出の階層的定義)
4. [AG/DG の認知科学的基盤](#4-agdg-の認知科学的基盤)
5. [対照学習と自己組織化](#5-対照学習と自己組織化)
6. [自律的発見機 — 帰納と演繹の統合](#6-自律的発見機--帰納と演繹の統合)
7. [未解決問題と検証必要箇所](#7-未解決問題と検証必要箇所)

付録:
- A. [統合前の素材ファイル対応表](#付録-a-統合前の素材ファイル対応表)
- B. [命名・用語の決めごと](#付録-b-命名用語の決めごと)
- C. [Part 1 / Part 3-7 への接続点](#付録-c-part-1--part-3-7-への接続点)

---

## 1. はじめに — Part 1 からの導入

> **3行要約**:  
> Part 1 §7 で導出した Wake-Sleep-Wake + AG/DG ゲートは、**認知科学的な肉付けを必要とする**工学的骨格である。  
> 本 Part 2 は、エピソード起点論（Tulving / CLS）・curl 検出（FEP 予測フェーズ）・神経調節物質対応・対照学習・自律的発見機 の観点から、この骨格に認識論的深みを与える。  
> Part 1 が「**何を**」「**どう測るか**」を確立したのに対し、Part 2 は「**人間の認知がなぜこう振る舞うか**」を geDIG のレンズで説明する。

### 1.1 Part 1 からの連続性

Part 1 で確立した要素:
- **三項 (EPC, ΔH, Δβ₁)** = 計量・測度・位相の原子（§4）
- **AG/DG 二段ゲート** = スカラー F による構造制御の最小機構（§7.2）
- **Wake-Sleep-Wake ループ** = CLS 論の工学的実装（§7.3）
- **演繹的 NN** = 戦略の帰結として導出（§7.4-7.5）

Part 2 ではこれらを前提として、**認知科学・神経科学・哲学的観点**からの補強を行う。

### 1.2 本 Part の位置づけ

Part 2 は、Part 1 と以下の Part をつなぐ**認識論的橋渡し**として機能する:

- **→ Part 3 (Phase 2 / Sleep)**: Sleep 相の詳細仕様は Part 3 で展開、Part 2 では**理論的基盤**を示す
- **→ Part 4 (Transformer 統合)**: Transformer への geDIG 適用は Part 4 で展開、Part 2 では**認知的解釈**を提供
- **→ Part 5 (応用・実装)**: 実験・PoC の詳細は Part 5、Part 2 では**設計原理**のみ記述

**統合素材**:
- [Part 1 §7](gedig_core_theory_unified.md) — 本ノートの前提
- 全素材ファイル（§2-6 で個別引用）

---

## 2. エピソード起点論と AND 蒸留

> **3行要約**:  
> 知識は**エピソード（経験）から出発**し、語彙（トークン・概念）は Sleep 相の **AND 蒸留**として emergent に立ち上がる。  
> これは Tulving (1972) episodic/semantic memory 区別、McClelland et al. (1995 CLS)、Bauer (2007) 発達観察の**工学的実装**にあたる。  
> Transformer は「**進化の順序を逆転**」させたアーキテクチャ（トークン起点）であり、geDIG はエピソード起点への回帰を主張する。

### 2.1 Tulving / CLS / 発達観察の再訪

- Tulving (1972): episodic memory → semantic memory の階層構造
- McClelland, McNaughton, O'Reilly (1995): 海馬（episodic）と新皮質（semantic）の complementary learning
- Bauer (2007): 発達観察 — 語彙爆発の**前に**エピソード記憶が形成される

これらはすべて「**エピソードが先、語彙は後**」という認知科学的事実。

### 2.2 AND 蒸留のメカニズム

Sleep 相で複数エピソードの**共通項を抽出**する操作。詳細な神経実装は Part 3 に譲るが、ここでは認知的プロセスとして記述:

- 複数エピソードで共起するパターン → 固定トークン（語彙）として emergent に立ち上がる
- AND 蒸留 = **論理積による骨格抽出**
- これは Hebbian 学習（§5.3 で明確化）とは異なる操作

### 2.3 GPT vs BERT の再解釈

`gpt_bert_gedig_perspective.md` §4 より:
- **GPT 的タスク**（自己回帰）: next-token 予測 → 時系列的エピソードの予測
- **BERT 的タスク**（双方向）: masked language modeling → 構造的 purpose の抽出

両者は geDIG レンズで見ると、**異なる粒度のエピソード処理**として統一できる。

### 2.4 DG ≠ 正例ラベル

`gpt_bert_gedig_perspective.md` §6 の重要な区別:
- DG は「**構造的に確定した**」を判定する
- 正例ラベル（外部教師）とは独立
- これは §7.6 の「自己教師信号生成器」設計と整合する

### 2.5 Part 1 §7 との接続

Part 1 §7.3 で「Sleep = エピソードから共通構造が AND 蒸留として emergent に立ち上がる」と述べたが、
本節はそれを**認知科学的に裏づける**。具体的な実装は Part 3（`phase2/`）。

**統合素材**:
- `gpt_bert_gedig_perspective.md` §1-6（全体）
- Part 1 §7.3（Wake-Sleep-Wake の CLS 対応）
- Tulving (1972), McClelland et al. (1995), Bauer (2007)（引用文献）

---

## 3. curl 検出の階層的定義

> **3行要約**:  
> curl 検出は**3 つの階層で読める多層概念**であり、単一定義ではなく**階層的定義**として扱うのが適切。  
> 階層1（数学的）: 連続空間での回転（`∇ × v` の特異点）、階層2（認知的）: FEP の予測フェーズ、階層3（実装的）: attention flow の解析。  
> 各階層は独立に検証可能で、Hodge 分解との厳密接続は open problem（§7）。

### 3.1 階層 1: 連続空間の数学的定義

`gedig_cognitive_foundation.md` §2.4 より:

> curl 極大 かつ div ≠ 0 の特異点

連続ベクトル場における渦の中心点。ストークスの定理、Hodge 分解などの数学的基盤を持つ。

### 3.2 階層 2: FEP の認知段階としての curl

`gedig_prediction_curl.md` §2 より:

> curl = 予測フェーズ = 「まだ見えていない芯の位置を推定する」認知操作

FEP の 4 段階（予測 → 認知 → 理解 → 行動）のうち、**予測**に対応する。

### 3.3 階層 3: 実装上の attention flow

`gedig_autonomous_discovery_machine.md` §4.2 より:

> 膨大な知識の中で構造が見える = attention flow の渦の中心検出

Transformer の attention weight をベクトル場として解釈し、その curl を計算する。

### 3.4 階層的定義の採用理由

3 義を**単一定義に無理に統一しない**のは以下の理由:
- 各階層は**独立した検証手段**を持つ（数学 / 認知実験 / 実装）
- 統一定義を先に固定すると、階層間の**有用な緊張**が失われる
- Part 1 §6 の「式は 1 つ、読解は複数、正準は 1 つ」原則と整合

### 3.5 Hodge 分解との接続（open problem）

- 連続空間の curl は `∇ × v` で定義される
- グラフ上の離散 curl は **Hodge 分解**（discrete Hodge-de Rham）で定義できる可能性
- これは `gedig_prediction_curl.md` §4「離散グラフの課題」で言及されているが、厳密な定式化は未着手
- §7.1 で open problem として記録

### 3.6 ノード粒度選択との関係

`gpt_bert_gedig_perspective.md` の議論と関連して、curl 検出は**ノード粒度**に依存する:
- `(state, action)` 粒度
- `(state, action, next_state)` 粒度
- エピソード粒度

粒度選択自体が意味空間の自律化の一部。

**統合素材**:
- `thinking/gedig_cognitive_foundation.md` §2（curl の数学的定義）
- `thinking/gedig_prediction_curl.md` §2-3（curl = 予測フェーズ）
- `thinking/gedig_autonomous_discovery_machine.md` §4（実装応用）

---

## 4. AG/DG の認知科学的基盤

> **3行要約**:  
> AG/DG 二段ゲートは、**古代脳（ノルアドレナリン）× 大脳（ドーパミン）の神経回路**と構造的に対応する。  
> 「理解の瞬間」は **BKT 相転移**類推でモデル化できる（離散的な秩序相への転移）。  
> 螺旋的 AG/DG フローにより、同じループが**解像度を段階的に上げながら**繰り返される。

### 4.1 神経調節物質との対応（computational analogy）

Part 1 §7.2 で既に述べた対応を詳述:

| geDIG | 神経調節物質 | 役割 |
|---|---|---|
| AG（Attention Gate） | ノルアドレナリン (NA) | 覚醒・曖昧性検知・注意の焦点化 |
| DG（Decision Gate） | ドーパミン (DA) | 報酬・確信・記憶固定化 |
| Sleep 相（consolidation） | アセチルコリン (ACh) | 記憶統合、REM 睡眠時の活性 |
| 負例信号（revisit） | コルチゾル | ストレス・失敗の記憶 |
| 行動選択バイアス | GABA | 抑制・選択的注意 |

> <small>**注**: これは implementation design としての類推（computational analogy）であり、
> 生理学的同一性を主張するものではない。詳細は `thinking/gedig_cognitive_steam_engine_20260306.md` を参照。</small>

### 4.2 BKT 相転移類推

`gedig_cognitive_foundation.md` §3 より:
- Berezinskii-Kosterlitz-Thouless (BKT) 相転移は、**渦の対生成-消滅**が臨界点となる位相転移
- 「理解の瞬間」= β₁ の離散変化（Part 1 §5.3 離散性が正しい）と対応
- 「閃き = トポロジカル再構成」の物理的類推として自然

**注**: これは **思想レベルの類推**であり、定量的主張ではない。§7.5 で open problem として記録。

### 4.3 螺旋的 AG/DG フロー

`thinking/spiral_agdg_flow.md` より:

```
探索 → 理解 → 統合 → 探索（次解像度） → 理解 → ...
```

同じ AG/DG ループが**解像度を段階的に上げながら**繰り返される螺旋構造。これは:
- Part 1 §7.3 の Wake-Sleep-Wake ループが**時間的に展開**されたもの
- 「Phase 1/2/... のロードマップ」（`phase1_special_gedig_roadmap.md`）と対応
- L1 検索を AG/DG 内に統合し、正例/負例の自律判定を実現

### 4.4 古代脳と抽象思考

`gedig_cognitive_foundation.md` §4 より:
- 古代脳（扁桃体、視床下部）が**回転・渦の検知**を担う可能性
- 前庭器官（平衡感覚）と curl 検出の類縁
- 「思考の流れを運動として見る」古代脳の視座

**注**: これも**仮説レベル**。神経解剖学的な検証は本研究の範囲外。

### 4.5 Part 1 §7.2 との接続

Part 1 では AG/DG を「**工学的な最小ゲート機構**」として位置付けた。
Part 2 §4 では同じ機構に**認知科学的な厚み**を与える。両者は同じ機構の異なる視点。

**統合素材**:
- `thinking/gedig_cognitive_foundation.md` §4-5（古代脳、BKT 相転移）
- `thinking/gedig_cognitive_steam_engine_20260306.md`（神経調節物質対応）
- `thinking/spiral_agdg_flow.md` §2-5（螺旋的フロー）

---

## 5. 対照学習と自己組織化

> **3行要約**:  
> **三角測量対照学習 (Triangular Contrastive Learning, TCL)**: 外積ベースで方向情報を扱い、内積（距離）ベースの従来対照学習を超える。  
> ラベル不要・ランダム初期値から構造が自己組織化する可能性を持つ（A1-A3 安定条件）。  
> Hebbian 学習と AND 蒸留の違いを**明確化**する: Hebbian = 活性パターン重み更新、AND 蒸留 = 共通項の構造抽出。

### 5.1 外積 vs 内積

`thinking/gedig_triangular_contrastive_learning.md` §3 より:
- **内積** = 距離情報（どれだけ似ているか）→ 従来の SimCLR, MoCo 等
- **外積** = 方向情報（どの方向に差があるか）→ TCL の核

外積ベースは「**差分の方向性**」を保持するため、curl 検出（§3）と自然に接続する。

### 5.2 A1-A3 安定条件

`gedig_triangular_contrastive_learning.md` §6 より、TCL が安定に収束する条件:
- **A1**: トリプレット（anchor, positive, negative）の選択が意味的に正しい
- **A2**: 外積が数値的に安定（ベクトル次元、スケール）
- **A3**: 学習率が適切（収束 vs 振動）

これらは**安定性の必要条件**であり、満たせば自己組織化が起こる。

### 5.3 Hebbian 学習 vs AND 蒸留（Part 2 での明確化）

両者は**異なる操作**として区別する:

| 操作 | 内容 | 時期 | 出力 |
|---|---|---|---|
| Hebbian 学習 | 「共に発火 → 結びつき強化」 | Wake / Sleep 両相 | 重みの更新 |
| AND 蒸留 | 複数エピソードの共通項抽出 | Sleep 相のみ | 新トークン / 概念の emergent 生成 |

両者は**補完関係**にあり、どちらか一方ではない。Part 3（Phase 2 / Sleep）で詳細実装。

### 5.4 ラベル不要・自己組織化の収束

`gedig_triangular_contrastive_learning.md` §4.1-4.2 より:
- 外部ラベル（教師信号）を必要とせず、**geDIG の F 値自体が教師信号**
- ランダム初期値から**位相秩序（β₁ 増加）に向かって自己組織化**
- これは Part 1 §7.6 の「F + 正例/負例 = 自己教師信号」設計と整合

**未検証**: 収束性の証明、収束速度の解析、局所最適への陥落リスク → §7 で open problem として記録。

### 5.5 Part 1 §7.6 との接続

Part 1 §7.6 で「**ベクトル表現の自律化**」を述べたが、その**具体的な学習法**が TCL になる。
3 相ループ（Wake → Sleep → Wake'）の**各相での学習**を TCL で実装する。

**統合素材**:
- `thinking/gedig_triangular_contrastive_learning.md` §3-6（TCL 全体）
- Part 1 §7.6（ベクトル自律化の方向性）

---

## 6. 自律的発見機 — 帰納と演繹の統合

> **3行要約**:  
> **帰納的 NN（LLM）** + **演繹的検出（geDIG curl）** = **自律的発見機 (autonomous discovery machine)**。  
> 蒸気機関比喩: **Transformer = 釜（熱浴）**、**geDIG = ピストン（仕事取り出し機構）**。不確実性 → 構造への変換循環。  
> 既存 LLM に curl プラグインを加えることで、**科学的発見の自動化**への道筋が開ける（1905 年実験に繋がる）。

### 6.1 帰納 + 演繹の統合構図

Part 1 §7.5 で対比表を示した帰納的 NN と演繹的 NN は、**対立ではなく補完**。統合構図:

```
帰納的 NN (LLM)
   ↓ 大量データの統計的圧縮
知識空間（潜在）
   ↓ curl 検出（演繹的）
構造の渦 = 新しい仮説の芯
   ↓ AG/DG 検証
自律的な発見
```

LLM は**過去の分布**を学ぶのが得意、geDIG は**新しい構造**を発見するのが得意。両者は別々の役割を担う。

### 6.2 蒸気機関比喩

`thinking/gedig_cognitive_steam_engine_20260306.md` §3 より:

- **Transformer = 釜（熱浴）**: 不確実性（エントロピー）を保持する大規模な潜在空間
- **geDIG = ピストン（仕事取り出し機構）**: 熱浴から構造的情報（仕事）を取り出す

この比喩は Part 1 §6.2 の Helmholtz 類推と連動する:
- 熱浴の温度 = λ（情報温度）
- 取り出される仕事 = F の減少量
- サイクル = Wake-Sleep-Wake

### 6.3 既存 LLM への curl プラグイン可能性

`thinking/gedig_autonomous_discovery_machine.md` §2 より:
- 既存 LLM の attention weight をベクトル場として扱う
- その curl を計算（§3.3 階層 3: 実装上の attention flow）
- curl の渦の中心 = 新しい仮説の候補

**実装上の注意**: Part 1 §9.5 で Transformer 実験の negative result を記録済。
curl プラグインの有効性は β₁ 次元フリー性 ([insight_beta1_dimension_free.md](thinking/insight_beta1_dimension_free.md)) を活用すれば改善する可能性（要検証）。

### 6.4 科学的発見への含意（1905 年実験）

`gedig_origin_story.md` §「究極のゴール」の **1905 年実験**:
- 1905 年以前の知識のみを持つ AI が、特殊相対論を「再発見」できるか?
- 自律的発見機がこれを達成できれば、geDIG の中核仮説「閃き = トポロジカル再構成」が実証される
- 失敗すれば、geDIG の限界が明確になる（Part 1 §9.6 の究極反証）

これは研究の**長期目標**であり、Part 2 §6 の位置付けとしては「**自律的発見機の理論的構想**」に留める。

### 6.5 Part 1 §7.5 との接続

Part 1 §7.5 で「帰納と演繹の補完」を対比表で示したが、Part 2 §6 ではその**工学的実装の構想**を展開する。
実装の詳細は `thinking/gedig_autonomous_discovery_machine.md` を参照。

**統合素材**:
- `thinking/gedig_autonomous_discovery_machine.md` §2-5（自律的発見機、1905 年実験）
- `thinking/gedig_cognitive_steam_engine_20260306.md` §3（蒸気機関比喩）
- Part 1 §7.5（帰納 vs 演繹の対比表）
- `gedig_origin_story.md` §「究極のゴール」

---

## 7. 未解決問題と検証必要箇所

> **3行要約**:  
> 本 Part 2 の主張は、Part 1 の工学的実証に比べて**仮説的・認識論的な性格**が強い。  
> 主な open problem: curl の離散グラフでの厳密定義（Hodge 分解）、v_understood の明示化、Active Inference との数式対応、nominalization 問題。  
> これらは Part 1 §9 の棄却可能性と同じスタイルで、**反証条件を明文化**して記録する。

### 7.1 curl の離散グラフでの厳密定義

**問題**: 連続空間の `∇ × v` は自然だが、**離散グラフ上の curl** の厳密定義が未確立。

**候補**:
- discrete Hodge-de Rham 理論
- discrete exterior calculus (DEC)
- persistent cohomology

これらのどれを採用するかは open。§3.5 で言及したが、Part 2 では宣言的に留め、厳密化は別途。

### 7.2 v_understood の明示化

`thinking/gedig_action_definition.md` §2 より:
```
行動 = 予測ベクトル - 理解ベクトル
v_action = v_prediction - v_understood
```

問題: `v_understood` の明確定義がない。
- エピソードの平均？
- Sleep 相で consolidated な構造？
- DG で確定したエッジ集合？

Part 3（Sleep 仕様）で明確化すべき。

### 7.3 Active Inference との数式対応

FEP / Active Inference（Friston）との対応:
- 予測（curl）= prior belief の更新
- 認知（AG）= likelihood の評価
- 理解（DG）= posterior の確定
- 行動 = policy の選択

これらは operational correspondence（操作的対応）として Part 1 §6.2 と同じ留保を入れる。
厳密な数式対応は open problem。FEP の過剰類推を避けることが重要。

### 7.4 nominalization 問題

`geDIG_transformer_discussion_20260416.md` §9.2 より:
- LLM は訓練後に**新トークン（概念）を生成できない**
- 人間はエピソードから新語彙を動的に立ち上げる
- geDIG の AND 蒸留（§2.2）はこれを工学的に実装する主張

**Concept-Reuse Asymmetry Test** (§9.2.4) は棄却条件として設計されている:
- LLM と人間被験者（n=10）で Task B/C の性能差を測定
- LLM で Task B 低、Task C 可能なら、nominalization 非対応性が実証される
- 実施は 2026-下半期 予定

### 7.5 棄却条件（Part 1 §9.6 と同じスタイル）

Part 2 の主要クレームごとの棄却条件:

| クレーム | 棄却条件 |
|---|---|
| **エピソード起点 + AND 蒸留**（§2） | Transformer と同等以上の性能を出すエピソード起点モデルが 3 年以内に出来ない |
| **curl 検出の階層的定義**（§3） | 3 階層のどれかが独立に検証不能であることが示される |
| **AG/DG の神経対応**（§4.1） | 反例の神経科学的知見が出る（e.g., NA/DA の機能が異なる回路で担われる） |
| **TCL の自己組織化**（§5） | ランダム初期値から収束せず、ラベルが必須であることが示される |
| **自律的発見機**（§6） | 1905 年実験で特殊相対論の再発見が 10 年以内に達成できない |

### 7.6 Part 1 §9 との関係

Part 1 §9 は**工学的・定量的な棄却条件**を記録した。
Part 2 §7 は**認識論的・仮説的な棄却条件**を記録する。両者は補完関係。

外部レビュアーによる批判的レビューが、特に Part 2 の仮説的主張には必須。

**統合素材**:
- `thinking/gedig_prediction_curl.md` §4（離散グラフの課題）
- `thinking/gedig_action_definition.md` §2-6（v_understood の曖昧性）
- `geDIG_transformer_discussion_20260416.md` §9（critical self-review, nominalization）
- Part 1 §9（棄却可能性のスタイル）

---

## 付録 A: 統合前の素材ファイル対応表

| 旧ファイル | 本ノートの対応節 | 統合後の扱い |
|---|---|---|
| `gpt_bert_gedig_perspective.md` | §2 | **維持**（§2 の核心ノート、エピソード起点論の原典） |
| `thinking/gedig_cognitive_foundation.md` | §3.1, §4.2, §4.4 | `_archive/` へ退避候補 |
| `thinking/gedig_prediction_curl.md` | §3.2, §7.1, §7.3 | `_archive/` へ退避候補 |
| `thinking/gedig_action_definition.md` | §7.2, §7.3 | `_archive/` へ退避候補 |
| `thinking/gedig_autonomous_discovery_machine.md` | §6.1, §6.3 | **維持**（Part 1 §7.5 から参照中、実装案の原典） |
| `thinking/gedig_cognitive_steam_engine_20260306.md` | §4.1, §6.2 | `_archive/` へ退避候補 |
| `thinking/gedig_triangular_contrastive_learning.md` | §5 | `_archive/` へ退避候補 |
| `thinking/spiral_agdg_flow.md` | §4.3 | `_archive/` へ退避候補 |

退避判断は本文統合が完了してから。

---

## 付録 B: 命名・用語の決めごと

Part 1 付録 C の規則を継承しつつ、Part 2 固有の用語を追加:

| 用語 | 採用する表記 | 避ける表記 |
|---|---|---|
| curl 検出 | 「curl 検出」 | 「渦検出」「回転検知」等の揺れを避ける |
| エピソード | 「エピソード」 | 「事象」「経験」等と混用しない |
| AND 蒸留 | 「AND 蒸留」 | 既存 Hebbian 学習と区別 |
| 自律的発見機 | 「autonomous discovery machine」または和訳 | 「自律 AI」等の一般的表現は避ける |
| 三角測量対照学習 | TCL (Triangular Contrastive Learning) | 略称は必ずフル形を併記 |

---

## 付録 C: Part 1 / Part 3-7 への接続点

### Part 1 → Part 2

- Part 1 §7.2 (AG/DG) → Part 2 §4（神経基盤）
- Part 1 §7.3 (Wake-Sleep-Wake) → Part 2 §2（エピソード起点論）
- Part 1 §7.5 (帰納 vs 演繹) → Part 2 §6（自律的発見機）
- Part 1 §7.6 (自己教師信号) → Part 2 §5（TCL）

### Part 2 → Part 3 (Phase 2 / Sleep)

- Part 2 §2.2 (AND 蒸留) → Part 3 で神経調節物質・シグナル伝播として実装
- Part 2 §4.1 (神経調節物質対応) → Part 3 で詳細仕様（GABA/DA/ACh/Cortisol）
- Part 2 §5.3 (Hebbian vs AND 蒸留) → Part 3 で両者の実装分離

### Part 2 → Part 4 (Transformer 統合)

- Part 2 §3.3 (curl の attention flow 解析) → Part 4 で実装
- Part 2 §6.3 (LLM への curl プラグイン) → Part 4 で検証実験
- Part 2 §6.2 (蒸気機関比喩) → Part 4 で熱浴 vs 仕事取り出しの工学的実装

### Part 2 → Part 7 (自己批判・棄却可能性)

- Part 2 §7.4 (nominalization 問題) → Part 7 で Concept-Reuse Test の実施
- Part 2 §7.5 (棄却条件) → Part 7 の統一ノートに統合
