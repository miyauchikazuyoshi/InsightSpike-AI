# HotPotQA dev 差分分析（geDIG vs baselines）

このノートは `gedig_20260114_233727.jsonl` を基準に、各ベースラインとの F1 差分を集計したもの。


## 使用ファイル

- gedig: `experiments/hotpotqa-benchmark/results/gedig_20260114_233727.jsonl`
- static_graphrag: `experiments/hotpotqa-benchmark/results/static_graphrag_20260115_231836.jsonl`
- bm25: `experiments/hotpotqa-benchmark/results/bm25_20260115_115855.jsonl`
- contriever: `experiments/hotpotqa-benchmark/results/contriever_20260116_075521.jsonl`
- closed_book: `experiments/hotpotqa-benchmark/results/closed_book_20260113_220125.jsonl`

## 概要（F1差分）

| baseline | count | mean ΔF1 | median ΔF1 | mean ΔEM | mean ΔSF-F1 | win/tie/loss |
| --- | --- | --- | --- | --- | --- | --- |
| static_graphrag | 7405 | -0.0218 | 0.0000 | -0.0144 | -0.0454 | 1047/5130/1228 |
| bm25 | 7405 | 0.0146 | 0.0000 | 0.0086 | -0.0454 | 748/6075/582 |
| contriever | 7405 | 0.1159 | 0.0000 | 0.1002 | 0.1387 | 2331/3898/1176 |
| closed_book | 7405 | 0.1864 | 0.0000 | 0.1545 | 0.3043 | 2876/3505/1024 |

## 代表的な負け筋（F1差分が大きい例）

### static_graphrag に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5ae33fa95542990afbd1e0f2 | -1.000 | 0.000 | 1.000 | Did Minersville School District v. Gobitis and Gravel v. United States occur ... |
| 5a8c493e554299653c1aa020 | -1.000 | 0.000 | 1.000 | John ruskin named his album due to a removal of what? |
| 5ab8420555429919ba4e2290 | -1.000 | 0.000 | 1.000 | The Ranters were a loose collection of radical so-called heretics whose main ... |
| 5ac1d4ad5542994d76dccef7 | -1.000 | 0.000 | 1.000 | Which of the seven Disney characters who had been given a series of their own... |
| 5a87c13f5542996e4f30890c | -1.000 | 0.000 | 1.000 | In what city did the "Prince of tenors" star in a film based on an opera by G... |
| 5a9068f85542990a98493623 | -1.000 | 0.000 | 1.000 | What was the nickname of the 76er who was drafted 3rd in the 2014 draft?  |

### bm25 に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5ab8420555429919ba4e2290 | -1.000 | 0.000 | 1.000 | The Ranters were a loose collection of radical so-called heretics whose main ... |
| 5ab626d555429953192ad279 | -1.000 | 0.000 | 1.000 | Anthony Avent played basketball fo a High School that is located in a city ap... |
| 5a712af15542994082a3e614 | -1.000 | 0.000 | 1.000 | Which show that premiered in May 2011 is hosted by someone with the birth nam... |
| 5ae4988655429970de88d9c1 | -1.000 | 0.000 | 1.000 | Which is a post-punk band, Boy Hits Car or The The? |
| 5a7a69365542996c55b2dd9b | -1.000 | 0.000 | 1.000 | Which film was Shannyn Sossamon in that was directed by Michael Lehmann? |
| 5ae22eae554299495565da30 | -1.000 | 0.000 | 1.000 | What was the 2010 population of the town where Black Crescent Mountain was lo... |
