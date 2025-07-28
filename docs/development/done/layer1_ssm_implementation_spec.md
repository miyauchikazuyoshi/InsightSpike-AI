# Layer1 SSMベース実装仕様書

## 1. 概要

Layer1は、生データストリームをリアルタイムで処理し、エピソード単位に変換する軽量な前処理層です。SSM（State Space Model）を採用することで、O(L)の計算量で長距離依存を捉えながら、低レイテンシ処理を実現します。

## 2. アーキテクチャ設計

### 2.1 全体構成
```
┌─────────────────────┐
│   Input Stream      │
│  (Text/Audio/...)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────┐
│   Layer1 SSM        │────▶│  Episode Queue  │
│  ・Ring Buffer      │     │  (asyncio)      │
│  ・State Tracking   │     └────────┬────────┘
│  ・Boundary Detect  │              │
└─────────────────────┘              ▼
                              ┌──────────────┐
                              │    Layer2    │
                              │ (L2Memory)   │
                              └──────────────┘
```

### 2.2 コンポーネント設計

#### 2.2.1 StreamProcessor
```python
class Layer1StreamProcessor:
    def __init__(self, config: Layer1Config):
        self.buffer_size = config.buffer_size  # 4096
        self.chunk_size = config.chunk_size    # 256
        self.flush_threshold = config.flush_threshold  # 512
        
        # SSMモデル（初期は既存のembedderで代替可）
        self.embedder = EmbeddingManager()  # 既存利用
        # 将来的にMamba等に置き換え
        # self.ssm_model = MambaModel(config.ssm_config)
        
        # バッファ管理
        self.token_buffer = deque(maxlen=self.buffer_size)
        self.state_cache = None
        
        # エンティティ抽出（オプション）
        self.ner_enabled = config.enable_ner
        if self.ner_enabled:
            self.ner_model = self._load_ner_model()
```

#### 2.2.2 エピソード境界検出
```python
class BoundaryDetector:
    def __init__(self, config):
        self.patterns = {
            'user_turn': r'\n\n|。{2,}|\?\s*$',
            'topic_shift': None,  # 将来的に実装
            'silence': 2.0,  # 秒
        }
        
    def should_flush(self, 
                    tokens: List[str], 
                    elapsed_time: float,
                    buffer_size: int) -> bool:
        # トークン数による強制フラッシュ
        if buffer_size >= self.flush_threshold:
            return True
            
        # パターンマッチング
        text = ''.join(tokens[-10:])  # 末尾確認
        if re.search(self.patterns['user_turn'], text):
            return True
            
        # 時間ベース（音声入力用）
        if elapsed_time > self.patterns['silence']:
            return True
            
        return False
```

## 3. データ構造定義

### 3.1 L1Episode
```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
import networkx as nx

@dataclass
class L1Episode:
    """Layer1が生成するエピソード"""
    # 必須フィールド
    embedding: np.ndarray      # shape: (768,), dtype: float16
    text: str                  # 元テキスト
    token_count: int          # トークン数
    timestamp: float          # 生成時刻
    
    # メタデータ
    entropy_score: float      # 予測エントロピー（0-1）
    boundary_type: str        # 'user_turn', 'forced', 'silence'
    
    # オプション（NER有効時）
    entity_graph: Optional[nx.DiGraph] = None
    
    def to_episode(self) -> Episode:
        """既存のEpisode型に変換"""
        return Episode(
            text=self.text,
            embedding=self.embedding,
            c_value=0.5,  # 初期値
            metadata={
                'token_count': self.token_count,
                'entropy': self.entropy_score,
                'boundary': self.boundary_type
            }
        )
```

### 3.2 設定構造
```python
@dataclass
class Layer1Config:
    # バッファ設定
    buffer_size: int = 4096      # 最大保持トークン数
    chunk_size: int = 256        # 処理チャンクサイズ
    flush_threshold: int = 512   # 強制フラッシュ閾値
    
    # モデル設定
    embedding_dim: int = 768
    use_fp16: bool = True
    
    # SSM設定（将来拡張用）
    ssm_model: str = "sentence-transformer"  # or "mamba-tiny"
    ssm_layers: int = 8
    ssm_hidden: int = 768
    
    # NER設定
    enable_ner: bool = False
    ner_model: str = "spacy_sm"
    max_entities: int = 10
    
    # 性能設定
    batch_timeout: float = 0.1   # 秒
    async_processing: bool = True
```

## 4. 処理フロー

### 4.1 メイン処理ループ
```python
async def process_stream(self, token_stream):
    """非同期ストリーム処理"""
    chunk_buffer = []
    start_time = time.time()
    
    async for token in token_stream:
        # バッファに追加
        self.token_buffer.append(token)
        chunk_buffer.append(token)
        
        # チャンク処理
        if len(chunk_buffer) >= self.chunk_size:
            await self._process_chunk(chunk_buffer)
            chunk_buffer.clear()
        
        # 境界検出
        if self._should_flush(time.time() - start_time):
            episode = await self._create_episode()
            await self.output_queue.put(episode)
            self._reset_buffer()
            start_time = time.time()
```

### 4.2 エピソード生成
```python
async def _create_episode(self) -> L1Episode:
    """バッファからエピソードを生成"""
    # テキスト結合
    text = self._decode_tokens(list(self.token_buffer))
    
    # エンベディング生成（現状は既存のEmbeddingManager使用）
    embedding = await self._generate_embedding(text)
    
    # エントロピー計算（簡易版）
    entropy = self._calculate_entropy(embedding)
    
    # NER処理（オプション）
    entity_graph = None
    if self.ner_enabled:
        entity_graph = await self._extract_entities(text)
    
    return L1Episode(
        embedding=embedding.astype(np.float16),
        text=text,
        token_count=len(self.token_buffer),
        timestamp=time.time(),
        entropy_score=entropy,
        boundary_type=self.last_boundary_type,
        entity_graph=entity_graph
    )
```

## 5. 既存システムとの統合

### 5.1 L2MemoryManagerとの接続
```python
class L2MemoryManager:
    def __init__(self, config, layer1_enabled=False):
        # 既存の初期化
        super().__init__(config)
        
        # Layer1統合
        if layer1_enabled:
            self.layer1 = Layer1StreamProcessor(config.layer1_config)
            self.episode_queue = asyncio.Queue()
            self._start_episode_consumer()
    
    async def _consume_episodes(self):
        """Layer1からのエピソードを非同期で処理"""
        while True:
            l1_episode = await self.episode_queue.get()
            episode = l1_episode.to_episode()
            
            # 既存のadd_episode処理
            self.add_episode(episode)
```

### 5.2 段階的移行計画

#### Phase 1: 既存embedderベース（現在可能）
- SentenceTransformerを使用
- 基本的なバッファリングとチャンク処理
- 既存のL2MemoryManagerと統合

#### Phase 2: 軽量モデル導入（次期）
- DistilBERT等の軽量モデルに移行
- リアルタイム性能の向上
- ストリーミング対応強化

#### Phase 3: SSM統合（将来）
- Mamba/RWKV等のSSMモデル導入
- O(L)の真のストリーミング処理
- 長距離依存の効率的な処理

## 6. 性能目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| レイテンシ | < 50ms/episode | エンベディング生成含む |
| スループット | > 1000 token/sec | GPU使用時 |
| メモリ使用量 | < 500MB | バッファ+モデル |
| CPU使用率 | < 30% | 通常処理時 |

## 7. テスト戦略

### 7.1 ユニットテスト
```python
def test_boundary_detection():
    detector = BoundaryDetector(config)
    
    # ユーザーターン終了
    assert detector.should_flush(['こんにちは', '。', '\n\n'], 0.5, 3)
    
    # 強制フラッシュ
    assert detector.should_flush(['x'] * 600, 0.1, 600)
    
    # 時間ベース
    assert detector.should_flush(['hello'], 3.0, 1)
```

### 7.2 統合テスト
```python
async def test_layer1_integration():
    # Layer1 + L2Memory統合テスト
    config = create_test_config()
    memory = L2MemoryManager(config, layer1_enabled=True)
    
    # ストリーム入力
    stream = create_test_stream("これはテストです。\n\n新しい話題。")
    await memory.layer1.process_stream(stream)
    
    # エピソード確認
    assert len(memory.episodes) == 2
    assert memory.episodes[0].text == "これはテストです。"
    assert memory.episodes[1].text == "新しい話題。"
```

## 8. 実装優先順位

1. **必須機能（Phase 1）**
   - [ ] 基本的なStreamProcessorクラス
   - [ ] 既存EmbeddingManagerとの統合
   - [ ] 境界検出ロジック
   - [ ] L2MemoryManagerとの接続

2. **拡張機能（Phase 2）**
   - [ ] 非同期処理の最適化
   - [ ] NERサポート
   - [ ] エントロピー計算の高度化

3. **将来機能（Phase 3）**
   - [ ] SSMモデル統合
   - [ ] マルチモーダル対応
   - [ ] 分散処理対応