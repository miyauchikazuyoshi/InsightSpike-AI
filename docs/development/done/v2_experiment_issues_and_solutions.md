---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# V2実験の問題点と解決策

## 概要
comprehensive_gedig_evaluation_v2実験の実装中に遭遇した問題点と、その解決策をまとめます。

## 主な問題点

### 1. LLMプロバイダーの初期化問題

#### 問題
- TinyLlamaのロードが非常に遅い（初回ダウンロードで時間がかかる）
- LocalProviderでの実験が完了しない
- OpenAI APIの初期化でエラーが発生

#### 原因
- TinyLlamaは1.1Bパラメータのモデルで、初回ダウンロードに時間がかかる
- transformersライブラリのモデルロードがメモリを大量に消費
- OpenAI APIキーの受け渡しに問題がある可能性

#### 解決策
1. **モデルの事前ダウンロード**
   ```python
   # 実験前に以下を実行
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```

2. **より小さなモデルの使用**
   - distilgpt2（82Mパラメータ）を使用
   - または、MockProviderで動作確認後に本番実行

3. **バッチ処理の実装**
   - 質問を小さなバッチに分けて処理
   - 進捗を定期的に保存

### 2. エピソード保存エラー

#### 問題
```
Failed to store episode: 
Failed to add knowledge: Failed to store episode
```

#### 原因
- MainAgentの初期化時にメモリマネージャーが正しく設定されていない
- FAISSインデックスの初期化に問題がある可能性

#### 解決策
メインコードの修正が必要：
```python
# src/insightspike/implementations/agents/main_agent.py
def add_knowledge(self, text: str, metadata: Optional[Dict] = None) -> bool:
    try:
        # エラーハンドリングを改善
        episode = self.l2_memory.create_episode(text, metadata)
        if episode:
            self.l2_memory.store_episode(episode)
            return True
    except Exception as e:
        logger.warning(f"Failed to add knowledge: {e}")
        # エラーでも処理を継続
        return False
```

### 3. 設定システムの複雑さ

#### 問題
- レガシー設定とPydantic設定の混在
- 実験用の設定作成が煩雑

#### 解決策
実験用のヘルパー関数を作成：
```python
def create_experiment_config(provider='mock', model='mock-model'):
    """実験用の簡易設定作成"""
    return type('Config', (), {
        'graph': type('GraphConfig', (), {
            'similarity_threshold': 0.7,
            'conflict_threshold': 0.5,
            'ged_threshold': 0.3
        })(),
        'embedding': type('EmbeddingConfig', (), {
            'dimension': 768,
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        })(),
        'llm': type('LLMConfig', (), {
            'provider': provider,
            'model_name': model
        })(),
        'memory': type('MemoryConfig', (), {
            'max_episodes': 1000,
            'compression_enabled': False
        })(),
        'insight': type('InsightConfig', (), {
            'detection_threshold': 0.5,
            'min_confidence': 0.3
        })()
    })()
```

## メインコードに必要な修正

### 1. LocalProviderの改善
```python
# src/insightspike/implementations/layers/layer4_llm_interface.py

def _initialize_local(self) -> bool:
    """Initialize local transformers model"""
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers package not installed")
        return False
    
    try:
        # キャッシュディレクトリを設定
        import os
        os.environ['TRANSFORMERS_CACHE'] = '~/.cache/huggingface'
        
        # タイムアウトを設定
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_info()
        
        # より詳細なログ
        logger.info(f"Loading tokenizer for {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir='~/.cache/huggingface'
        )
        
        # ... rest of initialization
```

### 2. エラーハンドリングの改善
```python
# src/insightspike/implementations/agents/main_agent.py

def process_question(self, question: str) -> CycleResult:
    """Process question with better error handling"""
    try:
        # 既存の処理
        result = self._process_internal(question)
        return result
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        # フォールバック結果を返す
        return CycleResult(
            response="Error occurred during processing",
            spike_detected=False,
            confidence=0.0,
            processing_time=0.0
        )
```

### 3. 実験用ユーティリティの追加
```python
# src/insightspike/experiments/utils.py (新規作成)

class ExperimentRunner:
    """実験実行のヘルパークラス"""
    
    @staticmethod
    def quick_test(provider='mock', n_questions=5):
        """クイックテスト実行"""
        # 設定作成
        config = create_experiment_config(provider)
        
        # エージェント初期化
        agent = MainAgent(config)
        
        # 質問生成と実行
        results = []
        for i in range(n_questions):
            result = agent.process_question(f"Test question {i}")
            results.append(result)
        
        return results
```

## 推奨される実験手順

### 1. 段階的アプローチ
1. **MockProviderで動作確認**
   - 実験フレームワークの動作確認
   - メトリクス計算の検証

2. **CleanProviderで実行**
   - データリークのない実験
   - ベースライン性能の確認

3. **LocalProvider（軽量モデル）**
   - distilgpt2などの小さなモデルで実験
   - グラフ成長の確認

4. **本番実験**
   - TinyLlamaまたはOpenAI APIで実行
   - 完全な100問での評価

### 2. 実験の並列化
```bash
# 複数の実験を並列実行
poetry run python src/run_experiment.py --provider mock &
poetry run python src/run_experiment.py --provider clean &
wait
```

### 3. 結果の段階的保存
- 10問ごとに中間結果を保存
- 実験が中断しても再開可能な仕組み

## 今後の改善点

1. **プロバイダーの事前検証**
   - 実験開始前にプロバイダーの利用可能性をチェック
   - 利用できない場合は代替プロバイダーを提案

2. **プログレスバーの実装**
   - tqdmなどを使用して進捗を可視化
   - 予想残り時間の表示

3. **リトライメカニズム**
   - LLM呼び出しの失敗時に自動リトライ
   - タイムアウト設定の調整

4. **メモリ効率の改善**
   - バッチ処理でのメモリ解放
   - 大規模実験での安定性向上

## まとめ

V2実験の主な課題は：
1. LLMプロバイダーの初期化と実行時間
2. エラーハンドリングの不足
3. 実験フレームワークの複雑さ

これらを解決することで、安定した実験実行が可能になります。