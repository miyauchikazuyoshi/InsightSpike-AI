# Config優先順位問題の解決

## 問題の概要

InsightSpikeでconfigファイルの設定（特にLLMプロバイダー）が正しく反映されない問題がありました。

## 原因

1. **プロジェクトルートの`config.json`が常に優先**
   - `provider: "clean"`が固定されていた
   - 実験用の設定を上書きしていた

2. **設定ファイルの読み込み順序**
   - config.json → config.yaml の順で検索
   - 最初に見つかったファイルで終了

3. **浅いマージの実装**
   - ネストした設定が正しくマージされない

## 実施した修正

### 1. config.jsonの一時退避
```bash
mv config.json config.json.bak
```

### 2. 設定ファイルの優先順位変更
`src/insightspike/config/loader.py`を修正：
```python
# Before:
["config.yaml", "config.json", ".insightspike.yaml"]

# After:
["config.yaml", ".insightspike.yaml", "config.json"]
```

### 3. 推奨される使用方法

#### 実験時の設定
```bash
# 方法1: 環境変数で指定
export INSIGHTSPIKE_CONFIG_PATH=experiments/my_experiment/config.yaml
poetry run spike query "Test query"

# 方法2: 実験ディレクトリから実行
cd experiments/my_experiment
poetry run spike query "Test query"

# 方法3: 環境変数でLLMを直接指定
export INSIGHTSPIKE_LLM__PROVIDER=distilgpt2
export INSIGHTSPIKE_LLM__MODEL=distilgpt2
poetry run spike query "Test query"
```

## 設定の優先順位（高い順）

1. コマンドライン引数
2. 環境変数（`INSIGHTSPIKE_*`）
3. 明示的に指定された設定ファイル（`--config`）
4. 環境変数 `INSIGHTSPIKE_CONFIG_PATH`
5. カレントディレクトリの設定ファイル
   - config.yaml（優先）
   - .insightspike.yaml
   - config.json（最後）
6. プリセットのデフォルト値

## 今後の改善案

1. **深いマージの実装**
   - 現在の`_deep_merge`メソッドの改善
   - ネストした設定の正しいマージ

2. **設定検証の強化**
   - どの設定が適用されたかのログ出力
   - 設定の競合警告

3. **実験用プリセット**
   - `--preset experiment`で実験用設定を簡単に適用

## まとめ

config.jsonを退避し、YAMLファイルの優先順位を上げることで、実験時の設定が正しく反映されるようになりました。