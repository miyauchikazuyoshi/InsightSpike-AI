# MainAgent API リファレンス

## MainAgentクラスの公開メソッド一覧

### 1. **initialize() -> bool**
- **機能**: すべてのコンポーネントを初期化
- **使用例**: `agent.initialize()`
- **戻り値**: 初期化成功時True

### 2. **process_question(question: str) -> str**
- **機能**: 質問を処理して回答を生成
- **使用例**: `answer = agent.process_question("What is AI?")`
- **戻り値**: 生成された回答テキスト

### 3. **add_document(text: str, c_value: float = 0.5, metadata: Optional[Dict] = None) -> bool**
- **機能**: ドキュメントをメモリに追加（グラフ更新なし）
- **使用例**: `agent.add_document("New knowledge", c_value=0.7)`
- **戻り値**: 追加成功時True
- **注意**: グラフは更新されない

### 4. **add_episode_with_graph_update(text: str, c_value: float = 0.5) -> Dict[str, Any]**
- **機能**: エピソードをメモリに追加し、同時にグラフを更新
- **使用例**: `result = agent.add_episode_with_graph_update("New insight")`
- **戻り値**: 
  ```python
  {
      "episode_idx": int,
      "vector": numpy.ndarray,
      "text": str,
      "c_value": float,
      "graph_analysis": Dict,
      "success": bool
  }
  ```
- **重要**: グラフも更新される唯一のデータ追加メソッド

### 5. **get_memory_graph_state() -> Dict[str, Any]**
- **機能**: メモリとグラフの現在の状態を取得
- **使用例**: `state = agent.get_memory_graph_state()`
- **戻り値**: メモリとグラフの統計情報を含む辞書

### 6. **get_stats() -> Dict[str, Any]**
- **機能**: エージェントの統計情報を取得
- **使用例**: `stats = agent.get_stats()`
- **戻り値**: エピソード数、グラフノード数などの統計

### 7. **save_state() -> bool**
- **機能**: エージェントの状態（メモリとグラフ）をディスクに保存
- **使用例**: `success = agent.save_state()`
- **戻り値**: 保存成功時True
- **保存内容**:
  - L2メモリ: `data/episodes.json`, `data/index.faiss`
  - L3グラフ: `data/graph_pyg.pt`

### 8. **load_state() -> bool**
- **機能**: エージェントの状態（メモリとグラフ）をディスクから読み込み
- **使用例**: `success = agent.load_state()`
- **戻り値**: 読み込み成功時True

### 9. **to_dict() -> Dict[str, Any]**
- **機能**: 後方互換性のための辞書変換
- **使用例**: `agent_dict = agent.to_dict()`
- **戻り値**: エージェントの設定を含む辞書

## 使用パターン

### パターン1: データ追加のみ（グラフ更新なし）
```python
agent = MainAgent()
agent.initialize()
agent.add_document("New knowledge")
agent.save_state()  # メモリのみ保存
```

### パターン2: グラフ付きデータ追加（推奨）
```python
agent = MainAgent()
agent.initialize()
result = agent.add_episode_with_graph_update("New insight")
agent.save_state()  # メモリとグラフを保存
```

### パターン3: 質問応答
```python
agent = MainAgent()
agent.initialize()
agent.load_state()  # 既存データを読み込み
answer = agent.process_question("What is quantum computing?")
```

## 重要な注意点

1. **グラフ更新**: `add_episode_with_graph_update()`を使わないとグラフは更新されない
2. **永続化**: `save_state()`を呼ばないとデータは保存されない
3. **初期化**: 使用前に必ず`initialize()`を呼ぶ必要がある
4. **グラフの性質**: L3グラフは推論用の一時的なグラフで、永続的な知識グラフではない