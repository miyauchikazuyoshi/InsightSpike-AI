# エージェント専門化計画

## 概要

MainAgentを言語処理と空間処理に特化した専門エージェントに分離し、より効率的でクリーンなアーキテクチャを実現する。

## 背景

### 現状の課題
- MainAgentがすべてのタスクを処理（責務が不明確）
- 言語タスク（384次元）と空間タスク（5次元）で最適な処理が異なる
- タスク特有の最適化が困難

### 提案する解決策
- 共通基盤（BaseAgent）の抽出
- タスク特化エージェントの実装
- 段階的な移行プロセス

## アーキテクチャ設計

### 現在の構造
```
MainAgent
├─ Layer1: Error Monitor
├─ Layer2: Memory Manager
├─ Layer3: Graph Reasoner
├─ Layer4: LLM Interface
└─ すべてのタスクを処理
```

### 将来の構造
```
BaseAgent（抽象基底クラス）
├─ 共通レイヤー管理
├─ 基本的なライフサイクル
└─ 共通インターフェース

LanguageAgent extends BaseAgent
├─ 384次元ベクトル処理
├─ Sleep Mode優先（探索的）
└─ セマンティック処理特化

MazeAgent extends BaseAgent
├─ 5次元ベクトル処理
├─ Wake Mode固定（効率的）
├─ 空間推論特化
└─ Layer5: Visualization（オプション）
```

## 実装計画

### Phase 1: 実験による検証（現在進行中）

#### 1.1 実験的MazeAgentの実装
```python
# experiments/maze_visualization/src/maze_agent.py
class ExperimentalMazeAgent:
    def __init__(self, base_agent: MainAgent):
        self.base = base_agent
        self.vector_dim = 5
        self.mode = ProcessingMode.WAKE
        
    def process_maze(self, maze_query):
        # 迷路特化の処理
        # ビジュアライゼーション
        # 5次元ベクトルでの効率的探索
```

#### 1.2 性能評価
- MainAgent vs ExperimentalMazeAgent
- メモリ使用量: 1/77削減を確認
- 処理速度: 50%以上の高速化目標

### Phase 2: BaseAgent設計（1週間）

#### 2.1 共通機能の抽出
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """すべてのエージェントの基底クラス"""
    
    def __init__(self, config, datastore):
        self.config = config
        self.datastore = datastore
        self.layers = self._initialize_layers()
        
    def _initialize_layers(self):
        """共通レイヤーの初期化"""
        return {
            'error_monitor': ErrorMonitor(),
            'memory_manager': Memory(self.datastore),
            'graph_reasoner': L3GraphReasoner() if available else None,
            'llm_interface': get_llm_provider(self.config)
        }
        
    @abstractmethod
    def process_query(self, query):
        """各エージェントで実装必須"""
        pass
        
    @abstractmethod
    def get_vector_dimension(self) -> int:
        """ベクトル次元数を返す"""
        pass
```

#### 2.2 インターフェース定義
```python
class AgentInterface(Protocol):
    """エージェントが満たすべきインターフェース"""
    def process_query(self, query: Any) -> Result
    def add_knowledge(self, knowledge: str) -> None
    def get_metrics(self) -> Dict[str, float]
```

### Phase 3: 専門エージェント実装（2週間）

#### 3.1 LanguageAgent
```python
class LanguageAgent(BaseAgent):
    """言語処理に特化したエージェント"""
    
    def __init__(self, config, datastore):
        super().__init__(config, datastore)
        self.embedder = SentenceTransformer()
        self.default_mode = ProcessingMode.SLEEP
        
    def get_vector_dimension(self) -> int:
        return 384
        
    def process_query(self, query: str) -> CycleResult:
        # 現在のMainAgentの処理をそのまま移行
        # セマンティック検索
        # 言語理解に最適化
```

#### 3.2 MazeAgent
```python
class MazeAgent(BaseAgent):
    """空間推論に特化したエージェント"""
    
    def __init__(self, config, datastore):
        super().__init__(config, datastore)
        self.vector_dim = 5
        self.default_mode = ProcessingMode.WAKE
        self.visualizer = None
        
    def get_vector_dimension(self) -> int:
        return 5
        
    def process_query(self, maze_query: Dict) -> MazeResult:
        # 実験コードから移行
        # 効率的な経路探索
        # オプションでビジュアライゼーション
```

### Phase 4: 移行戦略（1週間）

#### 4.1 後方互換性の維持
```python
# 移行期間中の互換性レイヤー
class MainAgent:
    """既存APIとの互換性を保つラッパー"""
    
    def __init__(self, config, datastore):
        # クエリタイプに応じて適切なエージェントを選択
        self.language_agent = LanguageAgent(config, datastore)
        self.maze_agent = MazeAgent(config, datastore)
        
    def process_question(self, query):
        if self._is_maze_query(query):
            return self.maze_agent.process_query(query)
        else:
            return self.language_agent.process_query(query)
```

#### 4.2 段階的移行
1. BaseAgentを実装
2. LanguageAgentを作成（MainAgentから機能移行）
3. MazeAgentを統合（実験コードから）
4. 互換性レイヤーでラップ
5. 十分なテスト後、古いMainAgentを廃止

## 期待される成果

### 1. アーキテクチャの改善
- **単一責任の原則**: 各エージェントが明確な責務
- **開放閉鎖の原則**: 新しいエージェント追加が容易
- **依存性逆転の原則**: 抽象に依存

### 2. パフォーマンス向上
- **MazeAgent**: 5次元ベクトルで77倍のメモリ効率
- **LanguageAgent**: 言語処理に最適化された設定
- **拡張性**: 新しいタスク特化エージェントを追加可能

### 3. 保守性向上
- タスクごとに独立したテスト
- 影響範囲の限定
- 明確なインターフェース

## リスクと対策

### リスク1: 既存コードへの影響
**対策**: 
- 互換性レイヤーによる段階的移行
- 包括的なテストスイート
- 既存APIの維持

### リスク2: 複雑性の増加
**対策**:
- シンプルな基底クラス設計
- 明確なドキュメント
- 実装例の提供

### リスク3: パフォーマンスオーバーヘッド
**対策**:
- エージェント選択の最適化
- 必要に応じてキャッシング
- プロファイリングによる検証

## タイムライン

### 現在〜2週間
- 実験的MazeAgentの実装と検証
- パフォーマンス測定
- 設計の妥当性確認

### 2〜3週間後
- BaseAgent設計と実装
- インターフェース定義
- テストフレームワーク準備

### 3〜5週間後
- LanguageAgent実装
- MazeAgent統合
- 統合テスト

### 5〜6週間後
- 互換性レイヤー実装
- ドキュメント整備
- 本番環境への移行準備

## 成功基準

1. **機能面**
   - 既存のすべてのテストが通る
   - 新しいエージェントが期待通り動作

2. **性能面**
   - MazeAgent: 50%以上の高速化
   - メモリ使用量: 適切に削減

3. **設計面**
   - 拡張が容易
   - コードの可読性向上
   - 責務の明確化

## 次のステップ

1. **実験の継続**: maze_visualizationでの検証
2. **設計レビュー**: BaseAgentインターフェースの確定
3. **プロトタイプ作成**: 最小限の実装で検証

この計画により、InsightSpikeはより柔軟で効率的なマルチエージェントシステムへと進化します。