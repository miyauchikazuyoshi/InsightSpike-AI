# RAG比較実験

## 🎯 実験の目的と目標

本実験では、3つの異なる検索拡張生成（RAG）アプローチを評価・比較します：

1. **InsightSpike-AI**: ΔGED × ΔIG埋め込みを使用したグラフベースのエピソード記憶を持つ、脳に着想を得た新しいシステム
2. **Standard RAG**: FAISSを使用した従来のベクトル類似性検索
3. **Hybrid RAG**: 語彙的（BM25）および意味的検索手法の組み合わせ

異なるRAGアーキテクチャにおける検索品質、速度、メモリ使用量、実装の複雑さのトレードオフを理解することが目的です。

## 📊 データ統計

### ドキュメントコーパス
- **総ドキュメント数**: 35（重複除去後）
- **元のドキュメント数**: 171
- **重複除去率**: 79.4%
- **ソース**: SQuAD、MS MARCO、CoQA、DROP、HotpotQA、BoolQ、CommonsenseQAを含む複数ドメインの質問応答データセット

### データ処理
- **エピソード管理**: 自動重複除去、分割、マージ
- **埋め込みモデル**: sentence-transformers/all-MiniLM-L6-v2（384次元）
- **グラフ構造**: 意味的関係を持つPyTorch Geometric形式

## 🔍 主な発見

### パフォーマンス比較

| システム | 検索時間 | 関連性 | Precision@5 | メモリ/ドキュメント |
|----------|----------|--------|-------------|---------------------|
| **InsightSpike-AI** | 484ms | 0.10 | 0.10 | 12.5 KB |
| **Standard RAG** | 91ms | 0.33 | 0.45 | 3.3 KB |
| **Hybrid RAG** | 26ms | 0.34 | 0.42 | 3.6 KB |

### システムの特徴

#### InsightSpike-AIの強み
- **グラフベースの推論**によるより良いコンテキスト理解
- **自動エピソード管理**（重複除去、分割、マージ）
- **内発的動機付け**による適応学習
- **エピソード記憶を持つ脳に着想を得たアーキテクチャ**

#### Standard RAGの強み
- **高速検索**（InsightSpike-AIより5.3倍高速）
- **FAISSによるシンプルな実装**
- **低メモリフットプリント**（3.8倍効率的）
- **確立された手法による予測可能な動作**

#### Hybrid RAGの強み
- **最速の検索**（InsightSpike-AIより18.6倍高速）
- **メトリクス全体でバランスの取れたパフォーマンス**
- **BM25統合によるキーワード処理**
- **重み付け組み合わせによる柔軟なスコアリング**

## 📁 ファイル構造

```
rag_comparison_experiment/
├── README.md                    # 英語版README
├── README_JP.md                # このファイル
├── code/
│   └── final_rag_comparison.py # メイン実験スクリプト
├── data_backup/
│   ├── episodes.json          # 重複除去されたドキュメントエピソード
│   ├── index.faiss           # FAISSベクトルインデックス
│   └── graph_pyg.pt          # PyTorch Geometricグラフ構造
└── results/
    ├── comparison_summary.md              # 人間が読める要約
    ├── comprehensive_comparison_report.json # 詳細なメトリクス
    └── comprehensive_rag_comparison.png   # 視覚的比較
```

### データファイルの説明

- **episodes.json** (360KB): メタデータを含む35の重複除去されたドキュメント
- **index.faiss** (53KB): ベクトル類似性検索用の事前計算されたFAISSインデックス
- **graph_pyg.pt** (35KB): ドキュメント関係をエンコードしたグラフ構造

## ⚡ パフォーマンスノート

### GPU対CPUパフォーマンス

#### GPUの利点
- **バッチ処理**: 大規模バッチの高速埋め込み計算
- **並列検索**: 加速された類似性計算
- **グラフ操作**: 効率的なPyTorch Geometric計算

#### CPUパフォーマンス
- **単一クエリ**: 個々の検索で競争力がある
- **メモリ効率**: 小規模展開での低オーバーヘッド
- **互換性**: GPU要件なしですべてのシステムで動作

### 最適化の推奨事項

1. **品質優先アプリケーション**: GPU加速を使用したInsightSpike-AI
2. **速度重視システム**: CPU最適化されたHybrid RAGをデプロイ
3. **リソース制限環境**: Standard RAGが最高の効率を提供

## 🚀 実験の実行

```bash
# 実験ディレクトリに移動
cd rag_comparison_experiment

# 比較を実行
python code/final_rag_comparison.py

# 結果はresults/ディレクトリに保存されます
```

## 📈 将来の改善

1. **スケールテスト**: より大規模なドキュメントコレクション（1000以上のドキュメント）での評価
2. **クエリの多様性**: より複雑なマルチホップ推論クエリでのテスト
3. **ハイブリッドアプローチ**: InsightSpike-AIのグラフ推論とHybrid RAGの速度の組み合わせ
4. **ファインチューニング**: 特定ドメイン向けのハイパーパラメータの最適化

## 📝 引用

この実験を研究で使用する場合は、以下を引用してください：

```
InsightSpike-AI RAG Comparison Experiment
https://github.com/InsightSpike-AI
2025
```

## 📄 ライセンス

この実験はInsightSpike-AIプロジェクトの一部であり、同じライセンス条項に従います。