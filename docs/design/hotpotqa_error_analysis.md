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

## HotPotQA type別（geDIG vs static_graphrag）

| type | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| bridge | 5918 | -0.0304 | 0.0000 | 824/4078/1016 |
| comparison | 1487 | 0.0123 | 0.0000 | 223/1052/212 |

## HotPotQA type別（geDIG vs bm25）

| type | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| bridge | 5918 | 0.0210 | 0.0000 | 623/4870/425 |
| comparison | 1487 | -0.0108 | 0.0000 | 125/1205/157 |

## Difficulty別（geDIG vs static_graphrag）

| level | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| hard | 7405 | -0.0218 | 0.0000 | 1047/5130/1228 |

## Difficulty別（geDIG vs bm25）

| level | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| hard | 7405 | 0.0146 | 0.0000 | 748/6075/582 |

## 質問形式別（geDIG vs static_graphrag）

| form | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| how | 31 | -0.0544 | 0.0000 | 11/11/9 |
| how long | 11 | -0.0268 | 0.0000 | 1/7/3 |
| how many | 118 | -0.0729 | 0.0000 | 16/75/27 |
| how much | 4 | 0.0748 | 0.0659 | 3/0/1 |
| how old | 2 | 0.0000 | 0.0000 | 0/2/0 |
| other | 2825 | -0.0367 | 0.0000 | 317/2072/436 |
| what | 1781 | -0.0156 | 0.0000 | 288/1193/300 |
| when | 187 | -0.0265 | 0.0000 | 40/101/46 |
| where | 177 | -0.0476 | 0.0000 | 31/100/46 |
| which | 1111 | -0.0035 | 0.0000 | 157/798/156 |
| who | 626 | -0.0164 | 0.0000 | 95/417/114 |
| why | 10 | -0.0925 | 0.0000 | 1/5/4 |
| yes/no | 522 | 0.0167 | 0.0000 | 87/349/86 |

## 質問形式別（geDIG vs bm25）

| form | count | mean ΔF1 | median ΔF1 | win/tie/loss |
| --- | --- | --- | --- | --- |
| how | 31 | -0.0394 | 0.0000 | 5/18/8 |
| how long | 11 | -0.0129 | 0.0000 | 0/10/1 |
| how many | 118 | 0.0269 | 0.0000 | 14/93/11 |
| how much | 4 | -0.1670 | 0.0000 | 1/2/1 |
| how old | 2 | 0.0000 | 0.0000 | 0/2/0 |
| other | 2825 | 0.0110 | 0.0000 | 227/2423/175 |
| what | 1781 | 0.0228 | 0.0000 | 214/1414/153 |
| when | 187 | 0.0134 | 0.0000 | 27/146/14 |
| where | 177 | 0.0123 | 0.0000 | 23/131/23 |
| which | 1111 | 0.0236 | 0.0000 | 118/912/81 |
| who | 626 | 0.0324 | 0.0000 | 74/500/52 |
| why | 10 | -0.0484 | 0.0000 | 1/7/2 |
| yes/no | 522 | -0.0287 | 0.0000 | 44/417/61 |

## Observations (static_graphrag 差分)

- 最も落ちやすい type: bridge (-0.030), comparison (0.012)
- 最も落ちやすい level: hard (-0.022)
- 最も落ちやすい 質問形式: why (-0.092), how many (-0.073), how (-0.054)

## 代表的な負け筋（F1差分が大きい例）

### static_graphrag に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5ac42a64554299076e296d7b | -1.000 | 0.000 | 1.000 | Clipper Cargo was a subsidiary cargo airline of  the principal and largest in... |
| 5abea85f5542990832d3a070 | -1.000 | 0.000 | 1.000 | Timothy J. Sloan was the CEO of Wells Fargo who succeeded the CEO who took th... |
| 5a7b820c5542997c3ec971d8 | -1.000 | 0.000 | 1.000 | The William Ulmer Brewery is located in a neighborhood that is policed by who? |
| 5ac240ef55429951e9e684ef | -1.000 | 0.000 | 1.000 | What year was a joint venture between RLJ Companies and this film studio foun... |
| 5a7c25895542997c3ec972e6 | -1.000 | 0.000 | 1.000 | Arena Bowl XII included what team coached by current head coach of the Washin... |
| 5a7a8d2355429941d65f26a1 | -1.000 | 0.000 | 1.000 | Zacarías Ferreira is the uncle of a professional basketball player who played... |

### bm25 に負けた例 (Top 6)

| id | ΔF1 | geDIG F1 | baseline F1 | question |
| --- | --- | --- | --- | --- |
| 5ac42a64554299076e296d7b | -1.000 | 0.000 | 1.000 | Clipper Cargo was a subsidiary cargo airline of  the principal and largest in... |
| 5abea85f5542990832d3a070 | -1.000 | 0.000 | 1.000 | Timothy J. Sloan was the CEO of Wells Fargo who succeeded the CEO who took th... |
| 5a7b820c5542997c3ec971d8 | -1.000 | 0.000 | 1.000 | The William Ulmer Brewery is located in a neighborhood that is policed by who? |
| 5a7a46605542994f819ef1ad | -1.000 | 0.000 | 1.000 | What year did Roy Rogers and his third wife star in a film directed by Frank ... |
| 5a81fa59554299676cceb1b0 | -1.000 | 0.000 | 1.000 | What type of profession does Jonah Meyerson and Alison Pill have in common? |
| 5ae1b1445542997283cd223d | -1.000 | 0.000 | 1.000 | Who starred in her final film role in the 1964 film directed by the man who a... |
