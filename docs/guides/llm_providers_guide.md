# LLMプロバイダー設定ガイド

## 概要

InsightSpike-AIは複数のLLMプロバイダーに対応しています：

- **Mock** - テスト用（APIキー不要）
- **OpenAI** - GPT-3.5, GPT-4
- **Anthropic** - Claude 3 Sonnet, Claude 3 Opus

## クイックスタート

### 1. APIキーの取得

**OpenAI:**
1. https://platform.openai.com にアクセス
2. API Keysセクションで新しいキーを作成
3. `sk-` で始まるキーをコピー

**Anthropic:**
1. https://console.anthropic.com にアクセス
2. API Keysセクションで新しいキーを作成
3. `sk-ant-` で始まるキーをコピー

### 2. 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# APIキーを設定
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxx"
```

### 3. 設定ファイルの準備

```bash
# OpenAI用の設定
cp config_examples/openai_config.yaml config.yaml

# または Anthropic用の設定
cp config_examples/anthropic_config.yaml config.yaml
```

### 4. 使用開始

```bash
# OpenAIで質問
spike query "量子コンピューティングとは何ですか？"

# 特定の設定ファイルを使用
spike query --config config_openai.yaml "AIの未来について"

# プロバイダーを直接指定
spike query --llm-provider anthropic "意識とは何か？"
```

## 詳細設定

### OpenAI設定

```yaml
llm:
  provider: openai
  model: gpt-4  # モデル選択
  temperature: 0.7  # 0.0-2.0 (創造性)
  max_tokens: 1000  # 最大トークン数
  top_p: 0.9  # nucleus sampling
  timeout: 30  # タイムアウト（秒）
```

**利用可能なモデル:**
- `gpt-3.5-turbo` - 高速で安価
- `gpt-3.5-turbo-16k` - 長文対応
- `gpt-4` - 最高性能
- `gpt-4-turbo-preview` - GPT-4の高速版

### Anthropic設定

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229  # 最高性能
  temperature: 0.7
  max_tokens: 1000
```

**利用可能なモデル:**
- `claude-3-sonnet-20240229` - バランス型
- `claude-3-opus-20240229` - 最高性能

### 料金の目安

**OpenAI (2024年7月時点):**
- GPT-3.5-turbo: $0.0005/1K tokens (入力), $0.0015/1K tokens (出力)
- GPT-4: $0.03/1K tokens (入力), $0.06/1K tokens (出力)

**Anthropic (2024年7月時点):**
- Claude 3 Sonnet: $0.003/1K tokens (入力), $0.015/1K tokens (出力)
- Claude 3 Opus: $0.015/1K tokens (入力), $0.075/1K tokens (出力)

## プログラムからの使用

### Python APIでの使用

```python
from insightspike.config import load_config
from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from insightspike.implementations.agents.datastore_agent import DataStoreMainAgent

# 設定を読み込み
config = load_config(config_path="./config_openai.yaml")

# エージェントを初期化
datastore = SQLiteDataStore("./data/sqlite/my_knowledge.db")
agent = DataStoreMainAgent(datastore, config)

# 処理を実行
result = agent.process("質問内容")
print(result['response'])
```

### 動的なプロバイダー切り替え

```python
from insightspike.providers import ProviderFactory

# プロバイダーを直接作成
provider = ProviderFactory.create("openai", {
    "api_key": "your-key",
    "model": "gpt-4",
    "temperature": 0.5
})

# 生成
response = provider.generate("こんにちは")
```

## トラブルシューティング

### "API key not found"エラー

```bash
# 環境変数が設定されているか確認
echo $OPENAI_API_KEY

# .envファイルから読み込む場合
source .env
```

### "Rate limit exceeded"エラー

- APIの利用制限に達しています
- 少し時間を置くか、有料プランにアップグレードしてください

### "Model not found"エラー

- モデル名が正しいか確認してください
- 最新のモデル名はプロバイダーのドキュメントを参照

## ベストプラクティス

### 1. APIキーの管理
- **絶対に**Gitにコミットしないでください
- 環境変数または`.env`ファイルを使用
- 定期的にキーをローテーション

### 2. コスト管理
- 開発時は`gpt-3.5-turbo`や`claude-3-sonnet`を使用
- `max_tokens`を適切に設定
- 使用量をモニタリング

### 3. パフォーマンス
- 頻繁な質問はキャッシュを検討
- バッチ処理で効率化
- タイムアウトを適切に設定

## 高度な使い方

### カスタムシステムプロンプト

```yaml
llm:
  provider: openai
  model: gpt-4
  system_prompt: |
    あなたは量子物理学の専門家です。
    科学的に正確で、分かりやすい説明を心がけてください。
```

### 複数プロバイダーの併用

```python
# 用途に応じて使い分け
simple_agent = DataStoreMainAgent(datastore, load_config("config_gpt35.yaml"))
advanced_agent = DataStoreMainAgent(datastore, load_config("config_gpt4.yaml"))

# 簡単な質問はGPT-3.5で
simple_result = simple_agent.process("今日の天気は？")

# 複雑な推論はGPT-4で
complex_result = advanced_agent.process("量子もつれと意識の関係について")
```

## まとめ

InsightSpike-AIのLLMプロバイダー機能により：

1. **柔軟性**: 用途に応じてモデルを選択
2. **コスト効率**: 必要に応じて安価なモデルを使用
3. **高性能**: 最新のLLMの能力を活用
4. **簡単**: YAMLファイルで簡単設定

質問やサポートが必要な場合は、GitHubのIssueでお問い合わせください。