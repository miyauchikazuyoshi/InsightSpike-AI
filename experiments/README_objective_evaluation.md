# 🧠 InsightSpike-AI 評価実験フレームワーク

**学術的研究基準に基づく包括的評価システム**

## 📖 概要

このフレームワークは、InsightSpike-AIの信頼性と外的妥当性を確保するための評価実験を提供します。Google Colab環境での実行に対応しており、再現可能で科学的にリゴラスな評価を実現します。

## 🎯 実験の特徴

### 📊 標準ベンチマーク評価
- **SQuAD v2.0**: 質問応答システムの標準ベンチマーク
- **ARC Challenge**: AI2推論課題（常識では解けない難問集）
- **論理パズル**: Monty Hall問題、ベルトランのパラドックス等

### ⚖️ 厳密なベースライン比較
- **Simple LLM**: 標準的な大規模言語モデル
- **Retrieval+LLM**: 検索拡張生成システム
- **Rule-based**: ルールベース推論システム
- **InsightSpike-AI**: 提案手法（洞察検出機能付き）

### 📈 統計的信頼性確保
- **クロスバリデーション**: 複数分割での検証
- **有意性検定**: t検定による統計的有意性確認
- **効果量分析**: Cohen's dによる実用的意義の評価

### 🔧 コンポーネント分析
- **アブレーション実験**: 各機能の寄与度分析
- **閾値感度分析**: パラメータのロバスト性検証
- **スケーラビリティ評価**: 大規模データでの性能評価

## 🚀 クイックスタート（Google Colab）

### 1. 基本セットアップ

```python
# Colabノートブックで実行
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI

# 必要なライブラリのインストール
!pip install -q nest-asyncio pandas matplotlib seaborn scikit-learn

# 評価フレームワークのインポート
import sys
sys.path.insert(0, 'experiments')
from colab_evaluation_interface import quick_demo, comprehensive_demo
```

### 2. クイック評価実験

```python
# 2-3分で完了する基本実験
result = quick_demo(sample_size=20)
```

### 3. 包括的評価実験

```python
# より詳細な分析（5-8分）
result = comprehensive_demo(
    datasets=["logic_puzzles", "squad_v2"], 
    sample_size=30
)
```

## 📊 利用可能な評価指標

### 基本指標
- **精度 (Accuracy)**: 正解率
- **応答時間 (Response Time)**: 平均処理時間
- **洞察検出率 (Insight Detection Rate)**: 洞察問題の検出精度

### 統計指標
- **p値**: 統計的有意性
- **効果量 (Cohen's d)**: 実用的意義
- **信頼区間**: 結果の信頼性範囲

### 安定性指標
- **クロスバリデーション**: 複数分割での性能安定性
- **標準偏差**: 結果のばらつき
- **再現性**: 固定シードでの結果一致性

## 🔬 実験の詳細設定

### ExperimentConfig

```python
from objective_evaluation_framework import ExperimentConfig

config = ExperimentConfig(
    name="custom_evaluation",
    description="カスタム評価実験",
    datasets=["squad_v2", "arc_challenge", "logic_puzzles"],
    baselines=["simple_llm", "retrieval_llm", "rule_based", "insightspike"],
    metrics=["accuracy", "confidence", "response_time", "insight_detection"],
    sample_size=100,
    cross_validation_folds=5,
    random_seed=42
)
```

### データセット設定

| データセット | 説明 | サンプル数 | 特徴 |
|-------------|------|----------|------|
| `squad_v2` | Stanford質問応答 | 可変 | 読解理解 |
| `arc_challenge` | AI2推論課題 | 可変 | 常識推論 |
| `logic_puzzles` | 論理パズル | 可変 | 洞察要求 |

## 📈 結果の解釈

### ベースライン比較

```python
# 典型的な結果例
baseline_results = {
    'InsightSpike-AI': {'accuracy': 0.89, 'response_time': 0.8},
    'Retrieval+LLM': {'accuracy': 0.82, 'response_time': 1.8},
    'Simple LLM': {'accuracy': 0.78, 'response_time': 1.2},
    'Rule-based': {'accuracy': 0.65, 'response_time': 0.3}
}
```

### 統計的有意性

- **p < 0.05**: 統計的に有意
- **Cohen's d > 0.5**: 中程度以上の効果
- **CV std < 0.02**: 高い安定性

### アブレーション解析

```python
# コンポーネント寄与度
ablation_results = {
    'Full InsightSpike': 0.89,
    'No Insight Detection': 0.81,  # -8% (洞察検出の寄与)
    'No Memory System': 0.84,      # -5% (メモリの寄与)
    'No Graph Reasoning': 0.82,    # -7% (グラフ推論の寄与)
    'LLM Only': 0.76              # -13% (専用機能の寄与)
}
```

## 🎨 可視化とレポート

### 自動生成される図表

1. **ベースライン比較グラフ**: 各手法の性能比較
2. **応答時間比較**: 処理速度の比較
3. **アブレーション結果**: コンポーネント寄与度
4. **閾値感度曲線**: パラメータロバスト性
5. **クロスバリデーション**: 安定性評価

### エクスポート形式

- **JSON**: 詳細な数値データ
- **PNG**: 高解像度の可視化
- **Markdown**: 人間が読みやすいレポート
- **PDF**: 印刷可能な統合レポート

## 🔧 高度な使用法

### カスタムベースライン追加

```python
class CustomBaseline:
    async def __call__(self, question: str, context: str = "") -> Dict:
        # カスタム実装
        return {
            'answer': 'Custom response',
            'confidence': 0.85,
            'response_time': 1.0,
            'method': 'custom_baseline'
        }

# フレームワークに追加
framework.baseline_comparator.baselines['custom'] = CustomBaseline()
```

### カスタムデータセット

```python
def load_custom_dataset(sample_size: int) -> List[Dict]:
    return [
        {
            'id': 'custom_1',
            'question': 'Your question',
            'context': 'Additional context',
            'expected_answer': 'Expected response',
            'source': 'custom_dataset'
        }
        # ... more samples
    ]

# フレームワークに登録
framework.dataset_loader.custom_loader = load_custom_dataset
```

## 📝 ベストプラクティス

### 実験設計

1. **適切なサンプルサイズ**: 統計的検出力を確保
2. **複数データセット**: 汎化性能の確認
3. **固定シード**: 再現性の保証
4. **バランス設計**: 偏りのない評価

### 結果解釈

1. **統計的有意性**: p値だけでなく効果量も考慮
2. **実用的意義**: 性能向上の実用性を評価
3. **安定性**: クロスバリデーションでの一貫性
4. **コンポーネント寄与**: システム設計の妥当性

### レポーティング

1. **透明性**: 実験設定の詳細記録
2. **再現性**: 他者が追試可能な情報提供
3. **制限事項**: 実験の限界を明記
4. **将来的改善**: 次のステップの提案

## 🚨 注意事項とトラブルシューティング

### Colab環境での制約

- **メモリ制限**: 大きすぎるサンプルサイズは避ける
- **実行時間**: セッションタイムアウトに注意
- **依存関係**: バージョン競合の可能性

### 一般的な問題

#### ImportError

```python
# 解決方法
import sys
sys.path.insert(0, 'experiments')
sys.path.insert(0, 'src')
```

#### メモリ不足

```python
# サンプルサイズを縮小
quick_demo(sample_size=10)  # デフォルト20から削減
```

#### 実行時間過長

```python
# データセットを制限
comprehensive_demo(datasets=["logic_puzzles"], sample_size=20)
```

## 📚 参考文献・関連文書

- [GPT-O3 Review Response Summary](../GPT_O3_REVIEW_RESPONSE_SUMMARY.md)
- [InsightSpike-AI Technical Specifications](../docs/technical_specifications.md)
- [Experimental Validation Documentation](./validation/README.md)

## 🤝 貢献・フィードバック

実験フレームワークの改善提案や新しい評価手法の追加は歓迎します：

1. **Issue報告**: 問題や改善案の報告
2. **Pull Request**: 新機能やバグ修正の提案
3. **データセット追加**: 新しいベンチマークの提案
4. **評価指標拡張**: 新しい評価軸の提案

## 📄 ライセンス

このフレームワークはInsightSpike-AIプロジェクトのライセンスに従います。

---

**Happy Evaluating! 🧠✨**

*この評価フレームワークにより、InsightSpike-AIの科学的妥当性と実用性が客観的に実証されます。*
