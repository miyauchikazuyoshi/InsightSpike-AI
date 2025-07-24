# V2 実験クイックスタートガイド

## 問題点の修正完了

以下の問題を修正しました：

1. **L2MemoryManagerのembedding次元の不一致** 
   - all-MiniLM-L6-v2は384次元（768ではない）
   
2. **EmbeddingManagerのメソッド名の修正**
   - `embed()` → `encode()` に修正

3. **Episode構造体のインポートパス修正**
   - `insightspike.structures.episode` → `insightspike.core.episode`

4. **エラーハンドリングの改善**
   - より詳細なログとフォールバック処理を追加

## 実験の実行方法

### 1. 簡単なテスト（推奨）

```bash
# Mockプロバイダーで動作確認
poetry run python run_simple_test.py

# 全プロバイダーのクイックテスト
poetry run python test_quick.py
```

### 2. 完全な実験

```bash
# Mockプロバイダー（最速）
poetry run python src/run_experiment.py --provider mock

# Cleanプロバイダー（データリークなし）
poetry run python src/run_experiment.py --provider clean

# LocalプロバイダーでDistilGPT2使用（軽量）
poetry run python src/run_experiment.py --provider local --model distilgpt2

# LocalプロバイダーでTinyLlama使用（重い、初回は2GBダウンロード）
poetry run python src/run_experiment.py --provider local --model tinyllama
```

### 3. OpenAI APIを使用する場合

```bash
export OPENAI_API_KEY=your_api_key
poetry run python src/run_experiment.py --provider openai --model gpt-3.5-turbo
```

## 実験の流れ

1. 100個のRAT（Remote Associates Test）質問を生成
2. 知識ベースをエージェントに読み込み
3. 各質問に対して：
   - グラフ状態（before）を記録
   - 質問を処理
   - グラフ状態（after）を記録
   - ΔGED/ΔIG/ΔF を計算
4. 結果をCSVとJSONで保存
5. 統計とプロットを生成

## 期待される結果

- **results/metrics/**: 各質問のメトリクス（CSV形式）
- **results/outputs/**: 各質問の詳細な応答
- **results/visualizations/**: グラフ成長の可視化
- **README.md**: 実験結果のサマリー

## トラブルシューティング

### タイムアウトする場合
- より小さなモデル（distilgpt2）を使用
- 質問数を減らす（run_simple_test.pyを参考に）

### メモリ不足の場合
- `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- バッチサイズを小さくする

### モデルのダウンロードが遅い場合
- 初回のみ時間がかかります（キャッシュされます）
- ~/.cache/huggingface/ にキャッシュされます

## 次のステップ

1. ベースライン実装との比較
2. グラフ進化の可視化改善
3. より大規模なデータセットでの実験