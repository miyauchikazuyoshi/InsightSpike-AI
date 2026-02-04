# JSAI2026 v3

v2からの更新版。geDIG v2実験結果（SP定義改善、entropy_sign、介入実験）を反映。

## v2からの主な変更点

### 1. SP (Shortcut Purity) 定義の改善

**v2**: 全ペア到達性（希釈問題あり）
**v3**: アンカー（CLS）への経路集中度

```python
SP = top_k_attention_to_CLS / total_attention_to_CLS
```

### 2. entropy_sign パラメータの導入

| フェーズ | entropy_sign | 解釈 |
|----------|--------------|------|
| 構造構築（迷路、事前学習） | +1 | 延伸が利得 |
| 構造特化（Fine-tuning） | -1 | 集中が利得 |

### 3. Transformer実験の更新

**v2の実験**:
- 層別F値の観察
- F正則化のα sweep

**v3の実験（置き換え）**:
- Positive/Negative介入実験
- 結果: Positive > Negative（4.6%差）
- geDIG Fが学習方向を予測できることを実証

## 実験データソース

- `experiments/transformer/results/gedig_v2/intervention_alpha_0.1_entropy_-1.json`
- `experiments/transformer/results/gedig_v2/microscopic_entropy_sign_1.json`

## ファイル構成

```
v3/
├── README.md                      # このファイル
├── PAPER_OUTLINE.md               # 論文構成案
├── TRANSFORMER_SECTION_DRAFT.md   # Transformer実験セクションのドラフト
└── main.tex                       # (未作成) LaTeXソース
```

## 主張の軸

> 「geDIG FはTransformerの学習効率を示す有効な指標であり、
> SP定義の改善とentropy_signパラメータにより、
> 迷路とTransformerで統一的に使用できる」

## 次のステップ

1. [ ] TRANSFORMER_SECTION_DRAFT.md を LaTeX に変換
2. [ ] 図2（SP定義の概念図）を作成
3. [ ] v2の main.tex をベースに v3 を作成
4. [ ] 全体の紙面調整

## 関連ドキュメント

- `experiments/transformer/docs/GEDIG_V2_SPEC.md` - geDIG v2 仕様
- `experiments/transformer/results/RESULTS_SUMMARY.md` - 実験結果サマリー
- `option_ab_merged_v2/` - v2のソース
