# geDIG Paper Directory

**最終更新**: 2026-04-17  
**用途**: geDIG 論文の各バージョン・会議投稿・共有リソース集約

---

## ディレクトリ構造

```
docs/paper/
├── README.md                    このファイル（全体案内）
│
├── v5/                          geDIG v5 (legacy、arxiv 公開済)
│   ├── arxiv_en/               英語版 arxiv 原稿（full / short）
│   ├── arxiv_ja/               日本語版 arxiv 原稿（short）
│   ├── geDIG_onegauge_improved_v5.{tex,pdf,bbl,blg}
│   └── review_v5.md            v5 に対する外部レビュー記録
│
├── v6/                          geDIG v6 (現行 arxiv 版)
│   ├── arxiv_en/               英語版 arxiv 原稿
│   └── geDIG_onegauge_improved_v6.{tex,pdf,bbl,blg}
│
├── v7/                          geDIG v7 (策定中、β₁ ベースへの大幅改訂)
│   ├── plan.md                 v7 論文構成計画（12 章 + Phase 0-3 実行順）
│   ├── tier1_action_plan.md    今日までの研究整理を v7 に反映する Tier 1 計画
│   └── restructured_draft.tex  再構成草稿（2025-09、v7 の種として参考）
│
├── jsai2026/                    人工知能学会 2026 向け投稿原稿（採択: v3）
│   ├── README.md
│   ├── v3/                     **採択版** (SP + entropy_sign + 介入実験)
│   ├── draft_a/, draft_b/      Option A/B（草稿、v3 以前）
│   ├── draft_hotpotqa/         HotpotQA 向け
│   ├── option_ab_merged/       A+B 統合版
│   ├── option_ab_merged_v2/    A+B 統合版 v2
│   └── (poster/)               ポスター（今後作成）
│
├── logs/                        プロセスメモ（作業履歴、完成版には反映済）
│   ├── STRUCTURE_IMPROVEMENT_LOG.md
│   ├── EXPECTED_BEHAVIOR_ADDED.md
│   ├── MAZE_EXPERIMENT_DESIGN_ADDED.md
│   └── FINAL_STRUCTURE.md
│
├── shared/                      共有リソース（バージョン横断で使用）
│   ├── figures/                    論文図（40+）
│   ├── fig_scripts/                図作成スクリプト
│   ├── maze_25x25_panels/          迷路実験の図版
│   ├── maze_25x25_snapshots/       迷路実験のスクリーンショット
│   ├── fig10_ablation_study.{pdf,png}
│   ├── fig11_component_analysis.{pdf,png}
│   ├── figures.png
│   ├── appendix_fep_mdl_bridge_ja.tex   FEP-MDL 付録（日本語）
│   ├── appendix_smallworld.tex          スモールワールド付録
│   ├── sections/                   セクション断片（日英）
│   ├── templates/                  表・図のテンプレート
│   ├── references.bib              BibTeX 共通
│   ├── figures_and_tables.md       図表一覧
│   ├── story_ged_as_insight.md     theoretical narrative
│   └── README_figs.md              figures/ の案内
│
└── data/                        実験結果 JSON/CSV (60+)
                                 ※ experiments/ 配下のコード 31+ 箇所から参照されているため
                                   shared/ には移動せず、root に残す
```

### 構造の原則

- **v5 / v6**: 履歴参照用。再ビルドしない前提（PDF 保存済）
- **v7 / jsai2026**: アクティブな作業場所
- **shared/**: バージョン横断で参照する共有リソース
- **data/**: 論文 + 実験コード両方から参照、root 維持
- **logs/**: 過去の作業履歴、通常参照不要

---

## 各ディレクトリの使い分け

### v5/ — 履歴参照用（基本的に touch しない）

arXiv 公開済のバージョン。変更しない。
新しい作業は v6 または v7 で行う。

### v6/ — 現行 arxiv 版（既存引用の安定性を維持）

v6 は Zenodo DOI `10.5281/zenodo.19454110` と結びついている。
**小規模な誤植修正以外は v6 を触らない**。
大規模改訂は v7 で行う。

### v7/ — 次期版の作業場（**アクティブな作業はここ**）

- `plan.md`: v7 の全体計画（12 章構成、Phase 0-3 実行順、各実験の検証 ID）
- `tier1_action_plan.md`: **今日までの研究整理**（連続・確率パラダイム批判、Figure 1、Helmholtz、略称統一等）を v7 に反映する Tier 分け計画

次に着手するのは Tier 1（実験データ不要、1-2 日完了）。

### jsai2026/ — 人工知能学会 2026 向け（国内、締切 2026-02-18）

短編投稿（2-4 ページ）。v7 の本体とは独立して進行。
Option A（迷路）/ Option B（Transformer）/ Option AB 統合の 3 案が既に策定済。

### logs/ — 履歴メモ

構造改訂や実験追加の過程メモ。完成版には反映済なので**通常は参照不要**。

### 共有リソース（直下）

`figures/`, `data/`, `references.bib` 等はバージョン横断で使用。
v5/v6/v7 の tex ファイルから相対パスで参照している。

**注意**: ビルドパスを保つため、直下の shared リソースは**移動しない**。

---

## クイックリファレンス

| やりたいこと | 見るファイル |
|---|---|
| 現行論文を見る | [v6/geDIG_onegauge_improved_v6.pdf](v6/geDIG_onegauge_improved_v6.pdf) |
| 次期改訂の全体計画 | [v7/plan.md](v7/plan.md) |
| 今日の整理を v7 にどう反映するか | [v7/tier1_action_plan.md](v7/tier1_action_plan.md) |
| JSAI 投稿の戦略 | [jsai2026/README.md](jsai2026/README.md) |
| 論文全体の思想的根拠 | [../research/thinking/insight_continuous_probabilistic_paradigm_critique.md](../research/thinking/insight_continuous_probabilistic_paradigm_critique.md) |
| Figure 1 matchstick の元データ | [../research/thinking/matchstick_figure_v2.html](../research/thinking/matchstick_figure_v2.html) |
| Part 1 コア理論統合ノート | [../research/gedig_core_theory_unified.md](../research/gedig_core_theory_unified.md) |

---

## バージョン履歴

- **v5** (〜2025 前半): 初期 arxiv 版、SP ベース
- **v6** (2025-後半〜2026-04): 現行 arxiv 版、SP ベース、v5 の改訂
- **v7** (策定中): β₁ ベース、AGHT 追加、Transformer F-regularization 追加、**思想的根拠層を追加**（連続・確率パラダイム批判）
