# HotPotQA dev 差分分析（geDIG vs baselines）

このノートは `gedig_20260122_100418.jsonl` を基準に、各ベースラインとの F1 差分を集計したもの。


## 使用ファイル

- gedig: `experiments/hotpotqa-benchmark/results/gedig_20260122_100418.jsonl`
- static_graphrag: `experiments/hotpotqa-benchmark/results/static_graphrag_20260115_231836.jsonl`
- bm25: `experiments/hotpotqa-benchmark/results/bm25_20260115_115855.jsonl`
- contriever: `experiments/hotpotqa-benchmark/results/contriever_20260116_075521.jsonl`
- closed_book: `experiments/hotpotqa-benchmark/results/closed_book_20260113_220125.jsonl`

## 概要（F1差分）

| baseline | count | mean ΔF1 | median ΔF1 | mean ΔEM | mean ΔSF-F1 | win/tie/loss |
| --- | --- | --- | --- | --- | --- | --- |
| static_graphrag | 7405 | -0.0055 | 0.0000 | -0.0053 | -0.0761 | 1153/5061/1191 |
| bm25 | 7405 | 0.0310 | 0.0000 | 0.0178 | -0.0761 | 974/5760/671 |
| contriever | 7405 | 0.1323 | 0.0000 | 0.1094 | 0.1081 | 2490/3782/1133 |
| closed_book | 7405 | 0.2028 | 0.0000 | 0.1637 | 0.2736 | 3036/3346/1023 |

## HotPotQA type別（geDIG vs static_graphrag）

| type | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| bridge | 5918 | -0.0065 | 0.0000 | 936/4028/954 |
| comparison | 1487 | -0.0012 | 0.0000 | 217/1033/237 |

## HotPotQA type別（geDIG vs bm25）

| type | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| bridge | 5918 | 0.0449 | 0.0000 | 833/4621/464 |
| comparison | 1487 | -0.0242 | 0.0000 | 141/1139/207 |

## Difficulty別（geDIG vs static_graphrag）

| level | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| hard | 7405 | -0.0055 | 0.0000 | 1153/5061/1191 |

## Difficulty別（geDIG vs bm25）

| level | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| hard | 7405 | 0.0310 | 0.0000 | 974/5760/671 |

## 質問形式別（geDIG vs static_graphrag）

| form | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| how | 31 | -0.0215 | 0.0000 | 12/10/9 |
| how long | 11 | 0.0253 | 0.0000 | 3/5/3 |
| how many | 118 | -0.0589 | 0.0000 | 17/70/31 |
| how much | 4 | -0.0471 | -0.0032 | 0/2/2 |
| how old | 2 | -0.1786 | -0.1786 | 0/1/1 |
| other | 2797 | -0.0163 | 0.0000 | 369/2015/413 |
| what | 1792 | 0.0091 | 0.0000 | 308/1218/266 |
| when | 187 | -0.0225 | 0.0000 | 44/94/49 |
| where | 178 | -0.0212 | 0.0000 | 36/93/49 |
| which | 1111 | 0.0083 | 0.0000 | 170/786/155 |
| who | 642 | 0.0061 | 0.0000 | 119/410/113 |
| why | 10 | -0.0896 | 0.0000 | 1/6/3 |
| yes/no | 522 | -0.0148 | 0.0000 | 74/351/97 |

## 質問形式別（geDIG vs bm25）

| form | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| how | 31 | -0.0065 | 0.0000 | 8/14/9 |
| how long | 11 | 0.0393 | 0.0000 | 2/9/0 |
| how many | 118 | 0.0410 | 0.0000 | 20/87/11 |
| how much | 4 | -0.2889 | -0.1746 | 0/2/2 |
| how old | 2 | -0.1786 | -0.1786 | 0/1/1 |
| other | 2797 | 0.0308 | 0.0000 | 317/2292/188 |
| what | 1792 | 0.0475 | 0.0000 | 273/1355/164 |
| when | 187 | 0.0173 | 0.0000 | 31/134/22 |
| where | 178 | 0.0428 | 0.0000 | 30/128/20 |
| which | 1111 | 0.0354 | 0.0000 | 141/869/101 |
| who | 642 | 0.0567 | 0.0000 | 102/481/59 |
| why | 10 | -0.0456 | 0.0000 | 2/6/2 |
| yes/no | 522 | -0.0602 | 0.0000 | 48/382/92 |

## Observations (static_graphrag 差分)

- 最も落ちやすい type: bridge (-0.0065), comparison (-0.0012)
- 最も落ちやすい level: hard (-0.0055)
- 最も落ちやすい 質問形式: how old (-0.1786, n=2), why (-0.0896), how many (-0.0589)

## 現状から言えること（メモ）

- ΔF1 の中央値が 0 のため、差分は一部サブセットに集中している
- static_graphrag との平均差は -0.0055 と僅差で、bridge が微マイナス
- bm25 に対しては全体でプラスだが、comparison でマイナスに振れる
- SF-F1 が static_graphrag/bm25 に対して -0.076 付近と弱い

## 改善仮説（要検証）

- bridge 型の負けは「必要な supporting facts が残らない」ケースが主因の可能性（SF-F1 低下と整合）
- comparison 型は AG/DG の発火が抑制され、比較根拠が不足している可能性
- F1=0 vs 1.0 の負けは「取得失敗」か「生成失敗」かを分離して確認が必要

## 次の分析アクション

- static_graphrag に負けた上位ケースで、取得文書と supporting facts の再現率を確認
- 各ケースで `initial_ag_fired` / `dg_fired` / `graph_edges` を抽出して失敗モードをラベル付け
- bridge / comparison 別に「候補数の減衰」と「誤統合（FMR）」の関係を確認

## 代表的な負け筋（F1差分が大きい例）

### static_graphrag に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5a87c13f5542996e4f30890c | -1.000 | 0.000 | 1.000 | In what city did the "Prince of tenors" star in a film based on an opera by Giacomo Puccini? |
| 5a85fb085542994775f606de | -1.000 | 0.000 | 1.000 | What is the name of the executive producer of the film that has a score composed by Jerry Goldsmith? |
| 5ab2a186554299295394677b | -1.000 | 0.000 | 1.000 | When did the English local newspaper, featuring the sculpture and war memorial in the Forbury gardens, change names? |
| 5abcfc365542993a06baf9d6 | -1.000 | 0.000 | 1.000 | What year did the series on CBS, starring the actor who known for his role in "Rebel Without a Cause," air? |
| 5a7d19d85542995ed0d165e8 | -1.000 | 0.000 | 1.000 | The Tennessee Volunteers football team plays as a member for a conference in what city? |
| 5a86769c5542994775f60776 | -1.000 | 0.000 | 1.000 | Armageddon in Retrospect was written by the author who was best known for what 1969 satire novel? |

### bm25 に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5ae058e855429945ae959331 | -1.000 | 0.000 | 1.000 | Are  Chrysalis and Look both women's magazines? |
| 5a7a46605542994f819ef1ad | -1.000 | 0.000 | 1.000 | What year did Roy Rogers and his third wife star in a film directed by Frank McDonald? |
| 5a86769c5542994775f60776 | -1.000 | 0.000 | 1.000 | Armageddon in Retrospect was written by the author who was best known for what 1969 satire novel? |
| 5a8d93ad554299653c1aa13d | -1.000 | 0.000 | 1.000 | Who was also an actor, Serri or John Fogerty? |
| 5a7f5c9f55429969796c1a0f | -1.000 | 0.000 | 1.000 | The city that contains the Yunnan Provincial Museum is also known by what nickname? |
| 5ae1f5a15542997f29b3c1b5 | -1.000 | 0.000 | 1.000 | The actor who appeared on Charlotte's Shorts and Weeds was born in what year? |
