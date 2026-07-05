# Research Notes — INDEX

**最終更新**: 2026-04-17  
**ステータス**: 統合作業中（散在していた研究メモを 0軸 + 6パート構成に再編中）

---

## 0. このディレクトリは何か

`docs/research/` は、geDIG フレームワーク（`F = ΔEPC - λ(ΔH + γ·Δβ₁)`）の
理論・設計・考察を蓄積した研究ノート群です。

論文化前の議論・試行錯誤・未解決問題が混在しているため、
本 INDEX は **「どこに何が書いてあるか」** を一望するための目次として機能します。

---

## 🎯 クイックアクセス

**1 ページで研究を理解したい方**: [overview.md](overview.md) — Figure 1・三本柱・実証状況・読み進め方を 1 ページに凝縮。外向け資料としてそのまま使える。

---

## ★ 0 軸: 出発点（人間の直感）

すべての起点は **[gedig_origin_story.md](gedig_origin_story.md)** にあります。

> 「アインシュタインのような AI を作るにはどうすればいいか？」
>
> アインシュタインと湯川秀樹の閃きを観察して得た直感:
> **閃き = 既存の知識（記憶）をトポロジカルに再構成すること**
>
> この直感は**人間（著者）の観察と直観から生まれた**もの。
> 数式・実装・精緻化は、ここから **AI との対話を通じて**展開された。

この 0 軸を Part 1 以降のすべてのノートが支えています。
ノートを読む時は、常に origin story に立ち返ることで意図が明確になります。

---

## 推しポイント（先に一言で）

**geDIG は、構造コストと構造利得を確率分布に押し込めずスカラー量として直接扱う工学フレームワークである。**

既存手法（FEP, VAE, GNN, Information Bottleneck, KL ダイバージェンス等）は
構造情報を確率表現に押し込めて処理するが、本研究は**構造量を構造量のまま扱う**。

### 核心視覚化

**[thinking/matchstick_figure_v2.html](thinking/matchstick_figure_v2.html)** — 研究の決定打。

同一の編集コスト `EPC = 1` に対して、位相 `Δβ₁` は +1 / 0 / −1 に分岐する:

| Case | 操作 | EPC | Δβ₁ | ΔH | 判定 |
|---|---|---|---|---|---|
| A | 三角形完成 | 1 | **+1** | +0.4 | **Aha! 洞察** |
| B | 棒の延長 | 1 | 0 | +0.3 | 力仕事 |
| C | 四角形崩壊 | 1 | **−1** | −0.2 | 構造崩壊 |

**KL ダイバージェンスは ΔH しか測れないため、Case A（洞察）と Case B（力仕事）を区別できない**。
geDIG は Δβ₁ によってこの区別を可能にする。これが既存手法への具体的な反例であり、
「スカラー直接扱い」の工学的必然性の証明。

### 三本柱

1. **三項の独立性**: `EPC` (計量) / `ΔH` (測度) / `Δβ₁` (位相) は数学の3基本概念に対応、必要十分な組
2. **AG/DG 二段ゲート**: スカラー F による工学的な構造制御
3. **Wake-Sleep-Wake**: 構造を動的に育てるアーキテクチャ（帰結として**演繹的 NN** が導かれる）

### エピソード起点（第二の柱）

語彙（トークン）は Sleep 相での **AND 蒸留**の産物として emergent に立ち上がる。
これは Tulving (1972) / McClelland et al. (1995 CLS) / Bauer (2007) の工学的実装。
詳細は [gpt_bert_gedig_perspective.md](gpt_bert_gedig_perspective.md)。

### 留保（誠実さの明示）

「構造 ≡ 確率」の数学的厳密化は本研究の範囲外とする。
これは情報幾何・Kolmogorov 複雑性・MaxEnt 等の専門家に任せる open problem。
本研究の貢献は、**スカラー F による構造制御が複数ドメインで実用的に機能することの工学的実証**。

実証ドメイン: 迷路（15×15〜51×51、スケール不変性確認）/
OCR（[vector-based-cnn-ocr](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr): 18K params で 73.53%）/
RAG (HotpotQA) / Transformer 層別解析。

---

## 1. 読み方ガイド（読者タイプ別）

| あなたは… | 推奨ルート |
|---|---|
| 初見で全体像を掴みたい | [Part 0](#part-0-哲学歴史) → [Part 1](#part-1-コア理論) § 1-2 |
| 査読者・批評者 | [Part 1](#part-1-コア理論) → [Part 7 自己批判](#part-7-自己批判棄却可能性) |
| 実装者 | [Part 1](#part-1-コア理論) § 4 (β₁ 実装) → [Part 5 応用・実装](#part-5-応用実装) |
| 認知科学・神経科学側の読者 | [Part 2](#part-2-認知推論アーキテクチャ) → [Part 3 Sleep](#part-3-phase-2sleep) |
| Transformer/LLM 研究者 | [Part 4 Transformer統合](#part-4-transformer統合) → [Part 2 エピソード起点](#part-2-認知推論アーキテクチャ) |
| 社会実装・ガバナンス | [Part 6 ガバナンス](#part-6-社会ガバナンス) |

---

## 2. 6 パート構成

### Part 0: 哲学・歴史（0 軸 + 補助ノート）

研究の出発点と直感的動機。1905年実験構想、洞察 vs 理解の区別、ロードマップ。
**Part 0 の中でも `gedig_origin_story.md` は 0 軸として特別な位置**を占める
（人間の直感の源、以降すべてのノートの基点）。

| ファイル | 内容 | ステータス |
|---|---|---|
| **★ [gedig_origin_story.md](gedig_origin_story.md)** | **0 軸: 1905実験、トポロジカル再構成の直感** | **基点・現役** |
| [insight_vs_understanding.md](insight_vs_understanding.md) | エッジ操作(理解) vs ノード創発(閃き) | 現役 |
| [phase1_special_gedig_roadmap.md](phase1_special_gedig_roadmap.md) | Phase 1-5 の進化ロードマップ | 現役 |

### Part 1: コア理論

スカラー直接扱い戦略、F 分解式の正準定義、三項の独立性（Figure 1）、β₁ 採用根拠、
**演繹的 NN は戦略の帰結**として導出。

| ファイル | 内容 | ステータス |
|---|---|---|
| **[gedig_core_theory_unified.md](gedig_core_theory_unified.md)** | **統合版（§1-9 本文化完了、付録 A-D 完備）** | **統合完了** |
| ★ [references/agent_memory_landscape_2026.md](references/agent_memory_landscape_2026.md) | **応用軸(agent memory)の外部地形図** — 2026 のベンチ・手法・オープン課題と geDIG の空き地(忘却/insight 評価)。精読者の読み方註釈付き | **リファレンス・現役**（応用軸はこれを参照） |
| ★ [thinking/strategy_memory_insight_roadmap_20260705.md](thinking/strategy_memory_insight_roadmap_20260705.md) | **戦略羅針盤** — RAG→agent memory 転回、閃きの工学 4 段階、二軸戦略 | **羅針盤・現役**（2026-07-05） |
| **★ [thinking/matchstick_figure_v2.html](thinking/matchstick_figure_v2.html)** | **核心視覚化: Figure 1（KLの盲点）/ Figure 2（剪定パラドックス）** | **核心図・現役** |
| [thinking/betti1_engineering_spec.md](thinking/betti1_engineering_spec.md) | β₁ 実装仕様（詳細なコード変更仕様、現役参照） | **実装仕様・現役** |
| [thinking/gedig_formula_three_readings_20260306.md](thinking/gedig_formula_three_readings_20260306.md) | F 式の3つの読み方（§4.3-4.4 で Helmholtz 詳細 + λ 温度制御） | **現役**（core §6 から参照中） |
| [thinking/gedig_as_discrete_fep_schrodinger_analogy_20260227.md](thinking/gedig_as_discrete_fep_schrodinger_analogy_20260227.md) | 離散 FEP / Schrödinger 類推 | 現役（付録 D 関連） |
| ~~deductive_optimal_nn.md~~ | ~~演繹的NN設計、迷路PoC、OCR検証~~ | **[_archive/](\_archive/) 退避済**（統合先: core §1, §7, §8） |
| ~~thinking/betti_number_adoption_memo.md~~ | ~~β₁ 採用の理論的根拠~~ | **[_archive/](\_archive/) 退避済**（統合先: core §3, §4, §5, §9） |
| ☀ [thinking/insight_bourbaki_three_structures.md](thinking/insight_bourbaki_three_structures.md) | 三項（EPC/ΔH/Δβ₁）と**現代数学の3つの基本空間**（計量/測度/位相）の対応 | **気づきメモ（2026-04-17）** |
| ☀ [thinking/insight_beta1_dimension_free.md](thinking/insight_beta1_dimension_free.md) | β₁ の**次元フリー性**と curse of dimensionality の回避 — Part 4 Transformer 統合の理論的正当化 | **気づきメモ（2026-04-17）** |
| ☀ [thinking/insight_three_terms_orthogonality.md](thinking/insight_three_terms_orthogonality.md) | 三項独立性の厳密化は**open problem**（例示的独立 → 統計的独立 → 情報幾何的直交の3段階） | **気づきメモ（2026-04-17）** |
| ☀ [thinking/insight_continuous_probabilistic_paradigm_critique.md](thinking/insight_continuous_probabilistic_paradigm_critique.md) | **Part 1 §1 戦略宣言の思想的根拠**: 空間軸（KL 盲点）と時間軸（FEP 粒度）の双対批判、「微分と確率が便利すぎた」の射程、離散・位相・組合せ論の系譜（Euler→Poincaré→Kolmogorov→Kitaev→geDIG） | **気づきメモ（2026-04-17）** |
| ☀ [thinking/insight_transformer_phase_transition_landscape.md](thinking/insight_transformer_phase_transition_landscape.md) | **Transformer × 相転移 × 位相**の先行研究ランドスケープ（Özönder 2025, Sun-Haghighat 2025, T3former）+ 既存 BKT 言及 3 箇所の統合。検証計画 H_ising-bkt 付き。**Part 4 §8.6 negative result の具体的修正方針** | **気づきメモ（2026-04-17）** |
| 🧪 [thinking/experiment_grokking_curl.md](thinking/experiment_grokking_curl.md) | **最優先実験プロトコル**: Grokking 相転移で β₁ + curl(attention) + F を測定。Nanda 2023 で 10 週間。既存 TAG-DS 2025 の β₁ proxy を **curl 追加で拡張**、既存 curl TODO ([cognitive_foundation §8](thinking/gedig_cognitive_foundation.md)) を一気に検証 | **実験プロトコル（2026-04-17）** |
| 💭 [thinking/insight_morphogenetic_generality.md](thinking/insight_morphogenetic_generality.md) | **妄想メモ（作者自身が疑いながら書いている）**: 樹木・脳細胞・粘菌の形態形成と geDIG の同型可能性。射程が広すぎてスケールの幅が見えない、工学的実証範囲外 | **妄想メモ（2026-04-17）** |

> **注記**: Helmholtz 対応（気づき 3 相当）は独立メモ化せず、[thinking/gedig_formula_three_readings_20260306.md](thinking/gedig_formula_three_readings_20260306.md) §4.3-4.4 に統合（`(EPC - B) - H` の形が Helmholtz 対応として正しい）。  
> **注記2**: マーク凡例:  
> - ☀ = 工学的実証に接続する気づき  
> - 🧪 = 実行可能な実験プロトコル（action 候補、GPU 利用時に着手）  
> - 💭 = 射程が広すぎて作者も疑っている妄想メモ（長期構想の種、論文では主張しない）

### Part 2: 認知・推論アーキテクチャ

Tulving / CLS / 発達心理、エピソード起点、AG/DG 二段ゲートの神経基盤、curl 検出、自律的発見機。

| ファイル | 内容 | ステータス |
|---|---|---|
| **[gedig_cognitive_architecture.md](gedig_cognitive_architecture.md)** | **統合版（骨格作成済、§1-7 + 付録 A-C、本文化予定）** | **新規・統合先** |
| [gpt_bert_gedig_perspective.md](gpt_bert_gedig_perspective.md) | エピソード先行、AND蒸留としてのトークン化 | **核心ノート（維持）** |
| [thinking/gedig_autonomous_discovery_machine.md](thinking/gedig_autonomous_discovery_machine.md) | curl 検出+LLM = 自律的発見機 | **核心ノート（維持、Part 1 §7.5 から参照中）** |
| [thinking/gedig_cognitive_foundation.md](thinking/gedig_cognitive_foundation.md) | curl 検出、BKT 相転移類推 | 統合対象 |
| [thinking/gedig_prediction_curl.md](thinking/gedig_prediction_curl.md) | curl = 予測フェーズ、FEP 対応 | 統合対象 |
| [thinking/gedig_action_definition.md](thinking/gedig_action_definition.md) | 行動 = 予測 - 理解、FEP 4段階対応 | 統合対象 |
| [thinking/gedig_cognitive_steam_engine_20260306.md](thinking/gedig_cognitive_steam_engine_20260306.md) | 神経調節物質と AG/DG の対応 | 統合対象 |
| [thinking/gedig_triangular_contrastive_learning.md](thinking/gedig_triangular_contrastive_learning.md) | 外積ベース対照学習 (TCL) | 統合対象 |
| [thinking/spiral_agdg_flow.md](thinking/spiral_agdg_flow.md) | AG/DG の螺旋的流れ | 統合対象 |

### Part 3: Phase 2 (Sleep / Offline Optimization)

NREM/REM 二サイクル、Hebbian 学習、シグナル伝播、神経調節物質メタファ。

| ファイル | 内容 | ステータス |
|---|---|---|
| [phase2/draft_specification.md](phase2/draft_specification.md) | Sleep 仕様、シグナル伝播、ヘッブ学習 | 現役 |
| [phase2/entropy_temperature_spec.md](phase2/entropy_temperature_spec.md) | Softmax Shannon / Boltzmann | 現役 |
| [phase2/phase2_offline_appendix_ja_en.md](phase2/phase2_offline_appendix_ja_en.md) | GABA/DA/Ach/Cortisol 対応 | 現役 |

### Part 4: Transformer 統合

層フロー同型、where/what 双対、動的学習 Transformer 設計、attention graph 解析、Part 1 §8.6 negative result への対処。

| ファイル | 内容 | ステータス |
|---|---|---|
| **[gedig_transformer_architecture.md](gedig_transformer_architecture.md)** | **統合版（骨格作成済、§1-8 + 付録 A-C、本文化予定）** | **新規・統合先** |
| ☀ [thinking/insight_transformer_phase_transition_landscape.md](thinking/insight_transformer_phase_transition_landscape.md) | **Transformer × 相転移 × 位相** 先行研究（Özönder/Sun-Haghighat/T3former）+ 既存 BKT 3 箇所統合 + 検証計画 H_ising-bkt / H_grokking-curl | **気づきメモ（2026-04-17）** |
| 🧪 [thinking/experiment_grokking_curl.md](thinking/experiment_grokking_curl.md) | **最優先実験**: Grokking で β₁ + curl(attention) + F を測定。Nanda 2023 再現 + curl 初観測 | **実験プロトコル（2026-04-17）** |
| [geDIG_transformer_discussion_20260416.md](geDIG_transformer_discussion_20260416.md) | 層フロー同型、§9 自己批判、nominalization 問題 | **核心ノート（維持、§9 が long-term 参照）** |
| [dynamic_transformer_spec.md](dynamic_transformer_spec.md) | 層の動的再構成、Phase 1/2 サイクル | 統合対象 |
| [splatting_attention_duality_for_gedig.md](splatting_attention_duality_for_gedig.md) | Splatting(where) / Attention(what) 双対 | 統合対象 |
| [insightspike_as_gnn_transformer.md](insightspike_as_gnn_transformer.md) | GNN-Transformer 統合、スパース接続 | 統合対象 |

### Part 5: 応用・実装

世界モデル、迷路経路、メモリ検索。

| ファイル | 内容 | ステータス |
|---|---|---|
| [self_organizing_world_model.md](self_organizing_world_model.md) | VAE + 対照学習 + 双曲幾何 | 現役 |
| [thinking/beta1_navigation_routing_20260208.md](thinking/beta1_navigation_routing_20260208.md) | 迷路経路探索での β₁ 活用 | 現役 |
| [thinking/memory_search_implementation_20260208.md](thinking/memory_search_implementation_20260208.md) | FAISS 実装メモ | 現役 |

### Part 6: 社会・ガバナンス

帰納/演繹統合の社会的含意、所有権・安全性インフラ。

| ファイル | 内容 | ステータス |
|---|---|---|
| [maker_sovereignty_full_proposal_en.md](maker_sovereignty_full_proposal_en.md) | Maker sovereignty 提案 | 現役（分割候補） |

### Part 7: 自己批判・棄却可能性

§9 critical review、nominalization、confirmation bias への自覚。

| 場所 | 内容 | ステータス |
|---|---|---|
| [geDIG_transformer_discussion_20260416.md](geDIG_transformer_discussion_20260416.md) §9 | Critical self-review、Concept-Reuse Asymmetry Test | 現役 |
| （新規作成予定） | 棄却可能性の統一ノート | **未着手** |

### その他（メタ・補助）

| ファイル | 内容 | ステータス |
|---|---|---|
| [README.md](README.md) | 補遺メモの案内 | 現役 |
| [call-for-reviewers.md](call-for-reviewers.md) | レビュアー募集 | 現役 |
| [outreach_guide.md](outreach_guide.md) | 概念説明・用語整理 | 現役 |
| [promotion_drafts.md](promotion_drafts.md) | プロモーション文案 | 現役 |

---

## 3. 統合作業の進捗

### Part 1 コア理論: 完了 ✅
- [x] `_archive/` ディレクトリ作成
- [x] INDEX.md（0 軸 + スカラー直接扱い戦略版）
- [x] `gedig_core_theory_unified.md` §1-9 **本文化完了**、付録 A-D 完備
  - §1 スカラー直接扱い宣言
  - §2 三項の独立性 + Figure 1 + 既存手法対比表
  - §3 F 分解式の定義（正準形）+ ASP → β₁ 移行史
  - §4 三項の数学的意味: 計量 / 測度 / 位相
  - §5 β₁ 採用の 5 理由 + 応用 4 ドメイン + 計算パラダイム
  - §6 3 つの読み方（Helmholtz = `(EPC-B)-H` を明記）+ λ 情報温度
  - §7 Wake-Sleep-Wake + 演繹的 NN の導出（帰納との対比表含む）
  - §8 実験的裏づけ（迷路・OCR・RAG・Transformer、negative result も正直に）
  - §9 未解決問題と棄却可能性（B1-B4 仮定、β₁ 非微分性、棄却条件表）
  - 付録 D 「構造 ≡ 確率の等価性を open problem として記録」
- [x] **気づきメモ 3 本作成**（Bourbaki 三大構造、β₁ 次元フリー、三項直交性）と相互リンク設置
- [x] Helmholtz 気づきは独立メモ化せず [three_readings §4.3-4.4](thinking/gedig_formula_three_readings_20260306.md) に統合
- [x] **旧ファイル `_archive/` へ退避**（2026-04-17）:
  - `deductive_optimal_nn.md` → 統合先: core §1, §7, §8
  - `thinking/betti_number_adoption_memo.md` → 統合先: core §3, §4, §5, §9
  - `thinking/insight_asymmetry_cost_gain.md` → Helmholtz は three_readings §4 に統合、本メモは退避

### Part 2 / 4 / 外向け資料: 骨格完了 ✅
- [x] **[gedig_cognitive_architecture.md](gedig_cognitive_architecture.md)** — Part 2 骨格（§1-7 + 付録 A-C）
  - エピソード起点、curl 検出の階層的定義、AG/DG 神経基盤、TCL、自律的発見機
- [x] **[gedig_transformer_architecture.md](gedig_transformer_architecture.md)** — Part 4 骨格（§1-8 + 付録 A-C）
  - Transformer を geDIG レンズで解釈、動的学習 Transformer 6 設計候補、β₁ 次元フリー性活用
  - Part 1 §8.6 negative result への対処方針
- [x] **[overview.md](overview.md)** — 1 ページ外向け資料
  - Figure 1・三本柱・実証状況・読み進め方を凝縮
  - 査読者募集・コラボレーション募集

### 次の段階
- [ ] Part 2 本文統合（§1-7 の本文化、素材 8 本から）
- [ ] Part 4 本文統合（§1-8 の本文化、素材 4 本から）
- [ ] 棄却可能性統一ノート（Part 7、core §9 + cog §7 + trans §8 を統合）
- [ ] 英語版 overview（必要性次第）
- [ ] 実験データ揃ってから論文 §1 Introduction 書き直し

### 未着手
- [ ] `gedig_cognitive_architecture.md`（Part 2 統合）
- [ ] `gedig_transformer_architecture.md`（Part 4 統合）
- [ ] 棄却可能性統一ノート（Part 7）
- [ ] 推しポイント外向け資料（スカラー直接扱い vs 既存手法 の対比）
- [ ] 統合完了した旧ファイルを `_archive/` へ退避

---

## 4. 未解決の重複・矛盾（統合時に解消すべき）

1. **「演繹」の定義ドリフト**: 
   - `deductive_optimal_nn.md` = 操作的演繹（F + 制約による構造選択）
   - `gedig_origin_story.md` = トポロジカル再構成
   - `betti_number_adoption_memo.md` = 位相不変量演繹
   - → Part 1 統合ノートで単一定義に収束

2. **AND 蒸留 vs Hebbian 学習**: 同じ操作か別操作か
   - → Part 2 / Part 3 で明確化

3. **curl 検出の3義**: cognitive_foundation / prediction_curl / autonomous_discovery_machine
   - → Part 2 統合ノートで統一

4. **棄却可能性の穴**: 演繹を謳うが falsifiability テストが未記述
   - → Part 7 新規ノートで明文化

5. **β₁ の高次元スケーリング**: 低次元実証のみ、NN 適用の理論的根拠が弱い
   - → Part 1 § 4 で未解決問題として明記

---

## 5. 命名規則

- `*_unified.md`: 複数の素材を統合した正準ノート（本文を持つ）
- `*_spec.md`: 実装仕様
- `*_discussion_YYYYMMDD.md`: 特定日の議論ログ
- `thinking/`: 探索的メモ（正準化されていない）
- `phase2/`: Phase 2 特有の仕様
- `_archive/`: 統合済み・陳腐化した旧ノート（履歴保持のため削除しない）
