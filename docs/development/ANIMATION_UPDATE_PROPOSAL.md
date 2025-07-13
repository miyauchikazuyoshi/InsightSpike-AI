# アニメーション改善提案

## 現実的なフロー表現

### Frame 1: クエリのベクトル化
```
Query: "How are entropies related?"
    ↓
[Embedding: 0.23, -0.45, 0.67, ...]
```

### Frame 2: メモリ検索（L2）
```
Memory Bank
┌─────────────┐
│ Episode 1   │ → Similarity: 0.89
│ Episode 2   │ → Similarity: 0.34  
│ Episode 3   │ → Similarity: 0.92
└─────────────┘
↓ Top-k retrieval
```

### Frame 3: グラフ構築（L3）
```
Retrieved Docs → Build Local Graph
    [Thermo] ←→ [Physics]
       ↓
    [Entropy] ←→ [Info Theory]
```

### Frame 4: GNN処理（オプション）
```
IF use_gnn=True:
    Node Features → GCN Layers → Enhanced Features
    (クエリ自体は変化しない)
```

### Frame 5: ΔGED/ΔIG計算
```
Before: Disconnected
After: Connected
ΔGED = -0.92 ✨
ΔIG = +0.56 📈
```

### Frame 6: LLM生成（L4）
```
Context: [Retrieved + Graph Analysis]
    ↓
LLM: "Thermodynamic and information 
      entropy are mathematically..."
    ↓
New Episode → Store in Memory
```

### Frame 7: サイクル繰り返し
```
Cycle 1 → Cycle 2 → ... → Convergence
```