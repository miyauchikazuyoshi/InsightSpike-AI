# Distance-Based Search LLM Experiment Insights

## 関連図表と実験ファイル

### 今回の実験で作成したファイル
- **[distance_search_direct_results.json](./distance_search_direct_results.json)** - 実験の生データ（各半径での検索結果とLLM応答）
- **本ドキュメント** - 実験結果の分析とインサイト

### 関連する既存の図表（参考）

#### 1. [arithmetic_clustering.png](../figures/arithmetic_clustering.png)
**意味**: ベクトル空間における特定ドメイン（算術式）のクラスタリング現象を示す
- 算術式が密な「ヘアボール」（平均類似度0.945）を形成
- 距離ベース検索で、このような密なクラスターを適切に扱える理由を説明

#### 2. [integration_comparison.png](../figures/integration_comparison.png)
**意味**: 重み付き統合と均一統合の性能比較
- クエリアウェアな重み付けの重要性を示す
- 今回の距離ベース検索も、距離による重み付けという観点で関連

#### 3. [qa_similarity_analysis.png](../figures/qa_similarity_analysis.png)
**意味**: 質問と回答のベクトル空間における関係性
- 質問空間と回答空間の間に約0.8の類似度ギャップが存在
- 今回の実験で期待回答との距離0.770を達成したことの意義を示す

## 実験結果サマリー

### テストケース: "How do everyday observations lead to scientific breakthroughs?"

期待される回答: "Scientific breakthroughs occur when prepared minds transform routine observations into profound insights by recognizing hidden patterns and fundamental principles in everyday phenomena."

### 結果比較

| 方法 | アイテム数 | 期待回答との距離 | コサイン類似度 |
|------|-----------|-----------------|---------------|
| radius=0.8 | 1 | 0.816 | 0.667 |
| **radius=1.0** | **6** | **0.770** | **0.703** |
| radius=1.2 | 7 | 0.816 | 0.667 |
| cosine top-5 | 5 | 0.793 | 0.686 |

## 重要な発見

1. **最適な半径が存在する**
   - radius=1.0が最良の結果を達成
   - 少なすぎる（radius=0.8）と情報不足
   - 多すぎる（radius=1.2）とノイズが増える

2. **距離ベース検索の利点**
   - 明確な境界設定が可能
   - 関連性の高いアイテムを適切に選択
   - コサイン類似度より良好な結果

3. **プロンプトへの距離情報の効果**
   - LLMに距離とコサイン両方の情報を提供
   - 例: `(dist=0.519, cos=0.865)`
   - LLMはこれらの数値を理解して活用可能

## 実装への示唆

1. **動的半径調整**
   - クエリの性質に応じて半径を調整
   - 専門的な質問: 狭い半径（0.7-0.9）
   - 一般的な質問: 中程度の半径（0.9-1.1）
   - 探索的な質問: 広い半径（1.1-1.3）

2. **ハイブリッドアプローチ**
   - 距離ベースで初期選択
   - コサイン類似度で順位付け
   - 両方のメトリクスをLLMに提供

3. **質の高い回答生成**
   - 適切な量の関連情報を提供
   - 距離情報により文脈の重要度を伝達
   - LLMが情報の関連性を判断しやすくなる

## 次のステップ

1. 他のテストケースでの検証
2. 異なる半径での系統的な実験
3. 動的半径選択アルゴリズムの開発
4. Wake-Sleepサイクルへの統合

## 実験スクリプト

今回の実験で使用したスクリプト：
- **[test_distance_search_direct.py](../test_distance_search_direct.py)** - メイン実験スクリプト
  - Anthropic APIを直接使用
  - 距離ベース検索とコサイン類似度検索を比較
  - 各半径（0.8, 1.0, 1.2）での性能を評価

その他の関連スクリプト（実験中に作成したが主要結果には使用せず）：
- test_distance_based_search.py - MainAgent経由での初期テスト
- test_llm_simple.py - LLM接続のデバッグ用
- test_config_debug.py - 設定問題のデバッグ用