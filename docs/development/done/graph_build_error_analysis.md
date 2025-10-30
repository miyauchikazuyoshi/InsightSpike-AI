---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# グラフ構築エラー分析

## エラー内容
```
Failed to build graph: too many values to unpack (expected 2)
```

## 調査結果

### 1. エラーが発生するタイミング
- MainAgentがprocess_questionを呼ぶとき
- Layer3でグラフを構築しようとするとき
- しかし、単体でScalableGraphBuilderを使うとエラーは発生しない

### 2. 空のグラフの状態
```
Previous graph: Data(x=[0, 384], edge_index=[2, 0], num_nodes=0)
```
- ノード数が0なのに、edge_indexの形状が[2, 0]となっている
- これは空のグラフを表現する正しい形式

### 3. 可能性のある原因

#### a) エンベディングの形式問題
- MainAgentから渡される文書のembeddingフィールドの形式が期待と異なる
- numpy配列ではなく、リストやタプルが渡されている可能性

#### b) retrieve/searchメソッドの戻り値
- CachedMemoryManagerのsearchメソッドが期待と異なる形式を返している
- タプルのアンパックで問題が起きている

#### c) 空のグラフ処理
- ノード数0の場合の処理で問題が発生

### 4. エラーが発生する具体的な場所
ScalableGraphBuilderの以下の処理のどこか：
1. `_get_embeddings` - ドキュメントからエンベディングを抽出
2. `_build_from_scratch` - vector indexのsearchメソッドの戻り値
3. edge_indexの構築時

## 解決策

### 短期的解決
1. エラーハンドリングを改善して、具体的なエラー箇所を特定
2. 空のグラフケースを適切に処理

### 長期的解決
1. ドキュメントフォーマットの統一
2. メソッドインターフェースの明確化
3. 型アノテーションの追加