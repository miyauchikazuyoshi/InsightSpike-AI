# PER=167.7 の集計経路監査（P0-2）

> **日付**: 2026-06-10
> **トリガ**: 外部レビュー「診断書と処方箋」(2026-06-10) P0-2
> **結論（先出し）**: 集計バグではなく**指標の取り違え**。167.7 の正体は旧プロトタイプ（rag-dynamic-db-v3, 2025-09）の**プロンプト長比率（enrichment ratio）**であり、PER（Perfect Episode Rate）とは無関係。さらに「n=168 クエリ」も誤りで、**168 は KB アイテム数**（実測クエリは5件程度）。

## 1. 167.7 の出所

- 起点: `archive/experiments/rag-dynamic-db-v3/README.md:9-27` —「167.7% Prompt Enrichment with Multi-Hop geDIG」「Knowledge Base Scale: **168 items** across 20 domains」
- 計算式: `archive/experiments/rag-dynamic-db-v3/src/run_scaled_experiments.py:330-337`
  ```python
  enrichment_ratio = len(prompt_2hop) / len(prompt_1hop)   # プロンプト文字長の比率
  ```
  = **multi-hop でプロンプトが何倍に膨らんだか**。100%超で当然の量。
- 実測の実体: `results/rag_prompt_impact/prompt_impact_results_20250910_005531.json` は **5クエリ×3設定**の小規模測定。168 はクエリ数ではない。

## 2. 誤伝播の経路

```
rag-dynamic-db-v3 README「167.7% enrichment」(2025-09)
  → docs/paper/shared/fig_scripts/generate_figures.py:30-35（enrichment=[100,112,128,167.7] ハードコード）
  → docs/paper/shared/figures_and_tables.md（キャプションは enrichment で正しい）
  → v5/v6 論文の表群（ホップ別・クエリタイプ別の「平均PER(%)」列として誤記）
  → v6 EN tab:rag_dynamic_result「PER 167.7*」+ 事後的な脚注 "normalized score"（最悪形）
```

PER の正しい定義（v6 EN L579）: 「Recall=1.0 かつ Contamination=0.0 のエピソード率」。lite スイートの実測は **Static 0.172 / geDIG 0.421**。167.7 が PER であることは定義上あり得ない。

## 3. 判定

| 仮説 | 判定 |
|------|------|
| (a) 集計バグ | No（enrichment_ratio の計算は正しい） |
| **(b) 指標の取り違え（enrichment→PER）** | **Yes（最有力・確定）** |
| (c) 手書き転記ミス | 部分的（複数texへ一貫してコピーされた） |

複合誤り: ①プロンプト強化率→PER の改名、②KBアイテム数168→クエリ数168 の取り違え。

## 4. 処置状況

| 対象 | 状態 |
|------|------|
| v6.1 EN `tab:rag_dynamic_result` | ✅ 修正済（2026-06-10、実測 PER 0.421 に差し替え） |
| v6.1 EN クエリ種別表（PER 152-185%） | ✅ 削除済（実測と不整合のため） |
| v6.1 JA `tab:multihop_analysis`「平均PER(%)」列 | ✅ 本監査と同時に「平均強化率(%)」へ改名 |
| v6.1 JA `tab:hop_sensitivity`「RAG PER(%)」列 | ✅ 同上 |
| v6.1 JA 「n=168 クエリ」表記 | ✅ 「KB 168項目・少数クエリ」へ修正 |
| **v6（提出時点版）** | 凍結（バージョン方針により非修正。誤りは本監査が記録） |
| v5 / v7 / archive 内の同種表記 | 未修正（v7 を次に触る際に本監査を参照のこと） |
| `docs/paper/shared/figures_and_tables.md` / `generate_figures.py` | キャプションは enrichment で正しい。図を再利用する際は「PER」と名乗らせないこと |

## 5. 再発防止

- 論文の表の数値は**結果ファイル（JSON/CSV）から自動生成**する（exp23 の `export_tables_tex` 方式をすべての表に拡大）。手書き転記が今回の根本原因。
- 指標名の辞書（PER / enrichment / Acc / FMR の定義と値域）を `docs/paper/shared/METRICS.md` として一元化する（TODO）。
