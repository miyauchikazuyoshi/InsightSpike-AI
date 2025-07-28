# InsightSpike-AI Class Relationships

## Overview
This document visualizes the complex relationships between major classes in InsightSpike-AI after the 2025-01 refactoring.

## Main Architecture (Post-Refactoring)

```mermaid
graph TB
    %% Main Agent Layer
    MainAgent[MainAgent<br/>統合エージェント]
    ConfigurableAgent[ConfigurableAgent<br/>設定可能エージェント]
    
    %% Core Layers (新アーキテクチャ)
    L1[Layer1: SSMQueryTransformer<br/>SSMクエリ変換層]
    L2[Layer2: WorkingMemory<br/>作業記憶層]
    L3[Layer3: GraphReasoner<br/>グラフ推論層]
    L4[Layer4: LLMInterface<br/>言語生成層]
    
    %% Data Storage (中心的役割)
    DataStore[DataStore<br/>永続化ストレージ]
    SQLiteStore[SQLiteStore<br/>SQLite実装]
    FileSystemStore[FileSystemStore<br/>ファイルシステム実装]
    
    %% Graph Management (簡素化)
    ScalableGraphBuilder[ScalableGraphBuilder<br/>グラフ構築]
    GraphMemorySearch[GraphMemorySearch<br/>グラフ検索]
    
    %% Episode Splitting (新機能)
    HybridEpisodeSplitter[HybridEpisodeSplitter<br/>ハイブリッド分割]
    
    %% Adaptive System
    AdaptiveProcessor[AdaptiveProcessor<br/>適応処理]
    
    %% Algorithms
    GED[GraphEditDistance<br/>グラフ編集距離]
    IG[InformationGain<br/>情報利得]
    
    %% Data Structures
    Episode[Episode<br/>エピソード]
    QueryState[QueryState<br/>クエリ状態]
    
    %% Agent Relationships
    MainAgent -->|uses| L1
    MainAgent -->|uses| L2
    MainAgent -->|uses| L3
    MainAgent -->|uses| L4
    MainAgent -->|uses| DataStore
    
    ConfigurableAgent -->|extends| MainAgent
    
    %% DataStore中心アーキテクチャ
    DataStore -->|implements| SQLiteStore
    DataStore -->|implements| FileSystemStore
    L2 -->|uses| DataStore
    L3 -->|uses| DataStore
    
    %% Layer1の新実装
    L1 -->|SSM processing| L1
    
    %% Layer2の簡素化
    L2 -->|uses| ScalableGraphBuilder
    L2 -->|uses| GraphMemorySearch
    L2 -->|manages| Episode
    
    %% Layer3の処理
    L3 -->|uses| GED
    L3 -->|uses| IG
    L3 -->|analyzes| Episode
    
    %% Episode分割の流れ
    L3 -->|triggers| HybridEpisodeSplitter
    HybridEpisodeSplitter -->|creates| Episode
    HybridEpisodeSplitter -->|stores| DataStore
    
    %% Adaptive処理
    AdaptiveProcessor -->|monitors| MainAgent
    AdaptiveProcessor -->|adjusts| L1
    AdaptiveProcessor -->|adjusts| L2
    AdaptiveProcessor -->|adjusts| L3
```

## Key Changes from Previous Architecture

### 1. L2MemoryManager廃止
- 以前: L2MemoryManagerが中央集権的にメモリ管理
- 現在: Layer2_WorkingMemoryが軽量な作業記憶として機能

### 2. DataStore中心アーキテクチャ
- 以前: 各層が独自にデータ管理
- 現在: DataStoreが一元的に永続化を管理

### 3. SSMベースのLayer1
- 以前: 単純なクエリ変換
- 現在: State Space Modelによる高速な不確実性検出

### 4. ハイブリッドエピソード分割
- 新機能: ベクトル空間とテキスト構造の両方を考慮した分割

## Component Interactions

```mermaid
sequenceDiagram
    participant User
    participant MainAgent
    participant L1_SSM
    participant L2_WM
    participant DataStore
    participant L3_GR
    participant L4_LLM
    
    User->>MainAgent: Query
    MainAgent->>L1_SSM: Transform query
    L1_SSM->>L1_SSM: SSM processing
    L1_SSM->>MainAgent: Transformed query
    
    MainAgent->>L2_WM: Process query
    L2_WM->>DataStore: Load episodes
    DataStore->>L2_WM: Episodes
    L2_WM->>L2_WM: Build graph
    
    MainAgent->>L3_GR: Analyze graph
    L3_GR->>L3_GR: Calculate GED/IG
    L3_GR->>L3_GR: Detect insights
    
    alt Spike detected
        L3_GR->>L4_LLM: Generate response
        L4_LLM->>MainAgent: Response
        MainAgent->>DataStore: Store new episode
    else No spike
        MainAgent->>User: Direct response
    end
    
    MainAgent->>User: Final response
```

## Memory Flow

```mermaid
graph LR
    subgraph Input
        Query[User Query]
    end
    
    subgraph Processing
        SSM[SSM Transform]
        Graph[Graph Building]
        Analysis[GED/IG Analysis]
    end
    
    subgraph Storage
        Episodes[Episodes]
        Embeddings[Embeddings]
        Graphs[Graph Cache]
    end
    
    subgraph Output
        Response[LLM Response]
        NewEpisode[New Episode]
    end
    
    Query --> SSM
    SSM --> Graph
    Graph --> Analysis
    
    Episodes --> Graph
    Embeddings --> Graph
    
    Analysis --> Response
    Analysis --> NewEpisode
    
    NewEpisode --> Episodes
    Graphs --> Graph
```

## Class Hierarchy (Simplified)

```mermaid
classDiagram
    class Agent {
        <<interface>>
        +process_question()
        +add_knowledge()
    }
    
    class MainAgent {
        +layers: List[Layer]
        +data_store: DataStore
        +process_question()
        +add_knowledge()
    }
    
    class ConfigurableAgent {
        +config: Config
        +initialize_layers()
    }
    
    class Layer {
        <<interface>>
        +process()
    }
    
    class Layer1_SSM {
        +ssm_model: SSMModel
        +transform_query()
    }
    
    class Layer2_WorkingMemory {
        +graph_builder: ScalableGraphBuilder
        +build_context()
    }
    
    class Layer3_GraphReasoner {
        +ged_calculator: GraphEditDistance
        +ig_calculator: InformationGain
        +analyze()
    }
    
    class Layer4_LLMInterface {
        +llm_provider: LLMProvider
        +generate()
    }
    
    class DataStore {
        <<interface>>
        +add_episode()
        +get_episodes()
        +search()
    }
    
    class SQLiteStore {
        +connection: Connection
        +add_episode()
        +get_episodes()
    }
    
    Agent <|-- MainAgent
    MainAgent <|-- ConfigurableAgent
    Layer <|-- Layer1_SSM
    Layer <|-- Layer2_WorkingMemory
    Layer <|-- Layer3_GraphReasoner
    Layer <|-- Layer4_LLMInterface
    DataStore <|-- SQLiteStore
    MainAgent --> DataStore
    MainAgent --> Layer
```

## Future Architecture (Quantum geDIG)

```mermaid
graph TB
    subgraph "Classical geDIG (Current)"
        ClassicalNode[Point Nodes<br/>点ノード]
        ClassicalEdge[Fixed Edges<br/>固定エッジ]
        ClassicalSpace[Discrete Space<br/>離散空間]
    end
    
    subgraph "Quantum geDIG (Future)"
        QuantumNode[Gaussian Nodes<br/>ガウシアンノード]
        QuantumEdge[Probabilistic Edges<br/>確率的エッジ]
        QuantumSpace[Continuous Field<br/>連続場]
    end
    
    ClassicalNode -.->|Evolution| QuantumNode
    ClassicalEdge -.->|Evolution| QuantumEdge
    ClassicalSpace -.->|Evolution| QuantumSpace
    
    QuantumNode -->|Uncertainty| GaussianDistribution[N(μ, Σ)]
    QuantumEdge -->|Interaction| WassersteinDistance[Wasserstein Distance]
    QuantumSpace -->|Density| KnowledgeField[Knowledge Density Field]
```