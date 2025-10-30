# InsightSpike-AI 実験設計インデックス

## 実験設計の分類

### 1. 完了済み実験（Validated Experiments）
これらの実験は完了し、InsightSpikeの有効性を実証しています。

#### 基礎検証実験
- [**Quick Validation**](quick_validation_design.md) - 最小構成での概念実証
- [**geDIG Validation**](gedig_validation_design.md) - 理論的基盤の数学的検証

#### 応用実験
- [**English Insight Experiment**](english_insight_experiment_design.md) - 段階的知識統合によるインサイト創発
- [**DistilGPT2 RAT Experiments**](distilgpt2_rat_experiments_design.md) - 創造的問題解決能力の検証

#### 比較研究
- [**Comparative Study**](comparative_study_design.md) - 3つのアプローチの包括的比較
- [**Current Framework Comparison**](current_framework_comparison_design.md) - アーキテクチャ改善の効果測定

### 2. 提案実験（Proposed Experiments）
今後実施予定の実験設計：

- [**Insight Task Benchmarks**](01_insight_task_benchmarks.md) - 標準化されたインサイト評価基準
- [**Real World Case Studies**](02_real_world_case_studies.md) - 実世界での応用検証
- [**Human Evaluation Studies**](03_human_evaluation_studies.md) - 人間による品質評価
- [**Comparative Analysis**](04_comparative_analysis.md) - 他手法との詳細比較
- [**Scalability Testing**](05_scalability_testing.md) - 大規模データでの性能検証
- [**Continual Learning**](06_continual_learning.md) - 継続学習能力の評価

## 実験の進化系譜

```
Quick Validation (概念実証)
    ↓
geDIG Validation (理論的基盤)
    ↓
English Insight (実装検証)
    ↓
DistilGPT2 RAT (創造性検証)
    ↓
Comparative Study (包括的評価)
    ↓
Current Framework (最新改善)
```

## 主要な知見

1. **モデル非依存性**: 82Mパラメータの小規模モデルでもインサイト創発可能
2. **RAGの限界**: 従来のRAGでは創造的思考が不可能
3. **グラフ構造の重要性**: 知識グラフの構造変化がインサイト品質と相関
4. **理論的裏付け**: geDIG公式（𝓕 = w₁ ΔGED - kT ΔIG）が数学的基盤を提供

## 実験実施ガイドライン

新しい実験を実施する際は、[EXPERIMENT_GUIDELINES.md](../../experiments/EXPERIMENT_GUIDELINES.md)を参照してください。

## リファレンス

- [実験概要](experiments_overview.md) - 全実験の統合的サマリー
- [実験ディレクトリ](../../experiments/) - 実際の実験コードとデータ