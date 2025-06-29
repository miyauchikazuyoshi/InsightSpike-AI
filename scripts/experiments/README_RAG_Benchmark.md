# RAG Systems Benchmark - Local Execution

このスクリプトは、ColabノートブックのRAGベンチマーク実験をローカル環境で実行するためのPythonスクリプトです。

## 特徴

- **ローカル実行**: Colabに依存せず、ローカル環境で安定して実行
- **複数プロファイル**: 異なる実験規模に対応
- **複数RAGシステム**: InsightSpike、LangChain、LlamaIndex、Haystackの比較
- **詳細な結果**: JSON、CSV、可視化グラフの自動生成
- **堅牢なエラーハンドリング**: 依存関係の問題にも対応

## インストール

### 必須依存関係

```bash
# 基本的なML/NLPライブラリ
pip install numpy pandas matplotlib seaborn scikit-learn
pip install sentence-transformers faiss-cpu

# オプション: 外部RAGフレームワーク
pip install langchain llama-index haystack-ai
```

### InsightSpike-AI環境

```bash
cd /path/to/InsightSpike-AI
# 既存の環境をそのまま使用可能
```

## 使用方法

### 基本実行

```bash
# 軽量デモ実行
python scripts/experiments/rag_benchmark_local.py --profile demo

# 研究用完全実行
python scripts/experiments/rag_benchmark_local.py --profile research

# 発表用バランス実行
python scripts/experiments/rag_benchmark_local.py --profile presentation

# InsightSpike特化実験
python scripts/experiments/rag_benchmark_local.py --profile insightspike_only
```

### プロファイル詳細

#### `demo` プロファイル
- **説明**: 軽量デモ実行 - 基本機能確認用
- **データサイズ**: [1000]
- **クエリ数**: 50
- **データセット**: squad_fallback, test_fallback
- **システム**: llm_only, bm25_llm, insightspike
- **実行時間**: 約30秒

#### `research` プロファイル
- **説明**: 研究用完全実行 - 全機能・大規模データ
- **データサイズ**: [1000, 5000, 10000, 50000]
- **クエリ数**: 1000
- **データセット**: squad, ms_marco, synthetic
- **システム**: 全システム（依存関係があるもの）
- **実行時間**: 15-30分

#### `presentation` プロファイル
- **説明**: 発表用実行 - バランス重視
- **データサイズ**: [1000, 5000, 10000]
- **クエリ数**: 200
- **データセット**: squad, squad_fallback, synthetic
- **システム**: llm_only, bm25_llm, insightspike, langchain
- **実行時間**: 5-10分

#### `insightspike_only` プロファイル
- **説明**: InsightSpike特化実験 - 詳細分析用
- **データサイズ**: [1000, 5000, 10000, 20000]
- **クエリ数**: 500
- **データセット**: squad, ms_marco, synthetic
- **システム**: insightspike のみ
- **実行時間**: 10-15分

## 出力結果

### ディレクトリ構造

```
experiments/results/{experiment_id}/
├── benchmark_results.json      # JSON形式の詳細結果
├── benchmark_results.pkl       # Python pickle形式
├── benchmark.log              # 実行ログ
└── visualizations/
    ├── accuracy_comparison.png
    ├── response_time_comparison.png
    └── combined_performance.png
```

### 結果の内容

- **accuracy**: 各システムの回答精度
- **response_time**: 各システムの応答時間
- **system_averages**: システム別平均統計
- **詳細ログ**: 各クエリの実行詳細

## 実行例

### 成功例

```bash
$ python scripts/experiments/rag_benchmark_local.py --profile demo

🔍 RAG SYSTEMS BENCHMARK - LOCAL EXECUTION
============================================================
📋 Selected Profile: demo
📝 Description: Lightweight demo execution - basic functionality check
📊 Data sizes: [1000]
🔍 Max queries per dataset: 50
📚 Datasets: ['squad_fallback', 'test_fallback']
🤖 RAG systems: ['llm_only', 'bm25_llm', 'insightspike']

✅ Benchmark completed successfully!
📁 Results available in: experiments/results/demo_20250629_173106
```

### エラー対処

```bash
# MLライブラリが不足している場合
❌ ML libraries not available. Please install: pip install sentence-transformers scikit-learn faiss-cpu

# 外部RAGフレームワークが不足している場合
❌ LangChain initialization failed: No module named 'langchain'
→ スクリプトは利用可能なシステムのみで継続実行
```

## 利点

### Colabと比較した利点

1. **安定性**: ネットワーク接続やセッションタイムアウトの心配なし
2. **再現性**: 同一環境での反復実験が可能
3. **カスタマイズ**: ローカルでのコード変更・デバッグが容易
4. **パフォーマンス**: ローカルマシンのフルパワーを活用
5. **プライバシー**: データがローカルに留まる

### 研究での活用

- **論文執筆**: 再現可能な実験結果
- **開発**: 新機能の迅速なテスト
- **比較研究**: 異なるRAGシステムの客観的評価
- **チューニング**: パラメータ最適化の効率化

## トラブルシューティング

### よくある問題

1. **インポートエラー**
   ```bash
   pip install <missing-package>
   ```

2. **メモリ不足**
   - より小さなプロファイルを使用
   - sample_sizesを減少

3. **InsightSpikeが動作しない**
   - プロジェクトのパッケージ構造を確認
   - `src/insightspike`への正しいパスを確認

### ログ確認

```bash
# 詳細ログを確認
tail -f experiments/results/{experiment_id}/benchmark.log
```

## カスタマイズ

### 新しいプロファイル追加

```python
"my_profile": {
    "description": "カスタム実験",
    "sample_sizes": [500, 1000],
    "max_queries": 100,
    "datasets": ["squad_fallback"],
    "systems": ["insightspike"],
    "enable_visualization": True,
    "save_results": True,
    "memory_cleanup": True,
    "strict_error_handling": False
}
```

### 新しいRAGシステム追加

`RAGSystemManager`クラスに新しいシステムの初期化メソッドを追加

## バージョン履歴

- **v1.0**: 初期バージョン - Colabノートブックからの変換
- **v1.1**: エラーハンドリング改善、InsightSpike統合修正

## ライセンス

このスクリプトは InsightSpike-AI プロジェクトの一部として、同じライセンス条件の下で提供されます。
