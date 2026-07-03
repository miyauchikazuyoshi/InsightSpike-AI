# このリポジトリの歩き方(MAP)

> **対象読者**: このリポジトリに初めて入る AI エージェント・共同研究者。
> **目的**: 探索なしで「どこに何があるか」「どの主張が生きているか」「用語の罠」に答える。
> **最終更新**: 2026-07-03(古くなっていたら疑って、README の Project Status を正とする)

## 1. 60 秒サマリ

geDIG は「知識グラフをいつ再構築すべきか」を単一ゲージ **F = ΔEPC − λ(ΔH + γΔβ₁)** で判定する
研究プロジェクト。maze / RAG / Transformer の 3 実験ラインが `src/gedig/` の統一 F-eval を共有する。
個人研究・AI 支援実装・**事前登録と敗北記録の文化**(誠実性が最優先の設計値)。

- **いま生きている主張の一覧** → ルート [README.md](../README.md) の Project Status 表(誠実版に維持されている)
- **実験の勝敗台帳** → [docs/prereg/README.md](prereg/README.md)(事前登録の一覧と結果)
- **現在の作業状態**(セッション間の引き継ぎ)→ リポジトリ外(著者のエージェントメモリ)。
  リポジトリ内の手がかりは prereg の「実行中」行と直近の git log

## 2. ディレクトリ地図

| 場所 | 中身 | 状態メモ |
|---|---|---|
| `src/insightspike/` | 本体パッケージ(~74k 行) | **geDIG 実装が 2 つある**(§5) |
| `src/gedig/` | 統一 F-eval コア(71 テスト) | 3 実験ラインの共有コア |
| `experiments/maze/` | 迷路ライン(主力)。`run_experiment_query.py` + `qhlib/` + `graph_persistent_dg/`(sleep) | 実行定型は [experiments/maze/README.md](../experiments/maze/README.md) |
| `experiments/hotpotqa_v2/` | RAG/ルーティングライン(BRIGHT、MuSiQue、dual-process) | Stage B は凍結中(DECISION 待ち) |
| `experiments/transformer/` | Transformer F-trajectory / F-regularization | F 符号問題は未確定(監査参照) |
| `experiments/refactor_*/` | 2026-02 リファクタ時の旧実装保存 | 読む必要は通常ない |
| `experiments/_archive_before_20260201_refactor/` | 35GB ローカルアーカイブ | **git 外・触らない** |
| `docs/gedig_spec.md` | ゲージの正準仕様 | 式の正典はここ |
| `docs/paper/` | 論文 v5(歴史)/ v6(arXiv)/ **v6.1(現行)**/ v7(策定中、Tier 1 ドラフトあり)/ jsai2026(発表済み) | |
| `docs/prereg/` | **事前登録と結果**(SOP は同ディレクトリ README) | 実験前に必読 |
| `docs/audits/` | 監査(F 符号、PER、oracle ceiling、sleep 設計監査) | 失敗の一次資料 |
| `docs/research/` | 理論ノート(~46 ファイル、`gedig_core_theory_unified.md` が統合版) | `thinking/` は思想メモ |
| `docs/research/gedig_origin_story.md` | 研究の動機と骨子(閃き=トポロジカル再構成、睡眠相の定義) | 骨子確認はここへ |
| `experiments/EXPERIMENT_GUIDELINES.md`, `OUTPUT_CONVENTION.md` | 実験の規約 | |
| `results/`, `experiments/**/results/` | 実験出力(gitignore) | ローカルのみ。`_exploratory_*/` は探索ラン隔離用 |

## 3. 用語の交通整理(エージェント最大の罠)

### 「Phase」は文脈で 4 つの意味がある

| 用法 | 出典 | 内容 |
|---|---|---|
| **論文 Phase 1/2/3** | v5/v6 論文、`docs/research/phase2/` | 1=オンライン制御(実証済み)、2=オフライン最適化 sleep(構想+最小実装)、3=Transformer 統合 |
| **SPEC 実装 Phase 0/1/2** | `graph_persistent_dg/SPEC.md` §6 | sleep 最小実装の実装段階(0=報酬記録、1=伝播、2=統合・アブレーション) |
| **v7 plan Phase 0-3** | `docs/paper/v7/plan.md` | 論文 v7 の実験ロードマップ(0=β₁統一、1=Exp4 再現、…) |
| **maze stage-1/stage-2** | README、maze 配下 | stage が正式(1=単発 PoC、2=Wake-Sleep-Wake)。Phase と混用しない |

### 頻出用語

- **F(ゲージ)**: ΔEPC − λ(ΔH + γΔβ₁)。F < 0 = 利得がコスト超過 → commit。正準は `docs/gedig_spec.md`
- **AG / DG**: 二段ゲート(AG=0-hop の曖昧さ検出、DG=multi-hop のショートカット確認)
- **sleep**: 現行実装は「**F 非依存の値再処理**」(軌跡 Q backup の転写+孤立除去+dim9 同期)。
  論文 Phase 2 が構想する F 駆動オフライン再配線の**前駆体**であり、同一視しないこと
  (prereg v2 §1.2 の用語注が正)
- **replay / on / off**(`--sleep-propagate`): replay=軌跡 Q 転写(現行推奨)、on=旧無向 max 伝播
  (飽和バグあり、v6_perseed 比較のためだけに残置)、off=生グラフ継承(対照)
- **10D / extended**: ベクトル拡張(dim8=生報酬、dim9=tanh(伝播値))
- **v6_perseed**: 2026-02 の stage-2 パッケージ実験。**その +23.4pt は sleep ではなく Wake1 効果**
  (設計監査で確定)— この数字を sleep の根拠に引用しないこと

## 4. よくある質問 → どこを読む

| 質問 | 一次情報源 |
|---|---|
| 理論の式の正典は? | `docs/gedig_spec.md`(README の式は概要) |
| 何が実証済み/撤回済み? | README Project Status 表 + `docs/prereg/README.md` 台帳 |
| なぜこの研究をやっている? | `docs/research/gedig_origin_story.md` |
| 実験を再現するには? | `experiments/maze/README.md`(定型・フラグ)+ 各 `run_*.sh` |
| 過去の失敗と教訓は? | `docs/audits/` + `docs/prereg/`(敗北記録は §8) |
| 新しい実験を始めるには? | `docs/prereg/README.md`(SOP)。/prereg スキルで雛形生成 |
| 論文の最新版は? | `docs/paper/v6.1/`(現行)。v7 は `docs/paper/v7/`(Tier 1 ドラフト済み) |
| JSAI2026 は? | 発表済み(2026-06-09、Session 2Yin-B-50)。`docs/paper/jsai2026/` |

## 5. 既知の負債(混乱ポイント先回り)

- **geDIG 実装が 2 系統**: `src/insightspike/gedig/`(Flash 版、Transformer attention 向け)と
  `src/insightspike/algorithms/gedig/`(フル版、maze/RAG 向け)。統一は将来課題
- 古い文書には `experiments/maze-query-hub-prototype/` という旧パスが残る(現在は `experiments/maze/`)
- `experiments/maze/ABLATION_PLAN.md` は 2026-02 の GED 定義切り分け計画(sleep アブレーションとは別物)
- SPEC.md の報酬値(+0.3/−0.3)は初期案。**実装値は novel +0.2 / revisit −0.4**(SPEC 冒頭の注記参照)
- Lite Mode 環境変数が 3 種ある(`INSIGHTSPIKE_LITE_MODE` / `INSIGHTSPIKE_MIN_IMPORT` / 旧 `INSIGHT_SPIKE_LIGHT_MODE`)
- `.venv` 必須(システム python に networkx なし)。実験は `.venv/bin/python3` で
- **docs/architecture/ は 2026-07-03 に鮮度監査済み**(コア 8 文書は実装と同期、⚠/❓ マーカーは
  architecture/README.md 参照)。**threelayer search と QHub ノード体系は docs/ 側に設計文書がない**
  (一次情報源は experiments/maze/README.md と qhlib/ 実装)。diagrams/ は 2026-02 以前の図が混在
  (新しい機構 — replay sleep、threelayer — の図は存在しない)
