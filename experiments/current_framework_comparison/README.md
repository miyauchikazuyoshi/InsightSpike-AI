# Current Framework Comparison Experiment

## 概要
english_insight_experimentと同じ質問・知識ベースを使用して、現在のInsightSpikeフレームワークで再実験を行います。

## 実験目的
- 前回の実験（カスタム実装）と現在のフレームワーク実装の性能比較
- Layer4プロンプトビルダー、メモリマネージャー、geDIGアルゴリズムの効果検証
- 洞察検出と回答品質の改善度測定

## 実験条件
- **質問**: english_insight_experimentと同じ6つの質問
- **知識ベース**: 同じ50エピソード（5フェーズ構成）
- **LLMモデル**: DistilGPT2（比較のため同一モデル使用）

## 期待される改善
1. プロンプトビルダーによる構造化された回答
2. geDIGによる本質的な洞察検出
3. メモリマネージャーによる効果的な知識統合
4. エージェントループによる反復的改善

## 実行方法
```bash
cd experiments/current_framework_comparison
python src/run_comparison_experiment.py
```

## 結果
- `/results/outputs/`: 実験結果JSON
- `/results/visualizations/`: 比較グラフ、洞察検出の可視化
- `/results/metrics/`: 評価指標の詳細