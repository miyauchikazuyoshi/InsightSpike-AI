# Level 3: 同型発見アルゴリズムの設計

## 1. 核心の問題

```
T* = argmin_T GED(T(G₁), G₂)
```

この式を実装するには、以下を定義する必要がある：
1. **変換T**とは何か（変換空間の定義）
2. **どう探索するか**（アルゴリズム）
3. **どう検証するか**（実験設計）

---

## 2. 変換空間Tの定義

### 2.1 グラフ変換の種類

| 変換タイプ | 操作 | 例 |
|-----------|------|-----|
| ノードリラベル | ノードの名前/属性を変更 | "惑星" → "電子" |
| エッジリラベル | エッジの関係を変更 | "公転" → "軌道運動" |
| ノード追加/削除 | 構造の拡張/縮小 | 新概念の導入 |
| エッジ追加/削除 | 関係の追加/削除 | 新関係の発見 |
| サブグラフ置換 | 部分構造の入れ替え | モチーフ変換 |

### 2.2 変換の階層

```
Level A: 表層変換（リラベルのみ）
  - ノード名の対応付け
  - 関係名の対応付け
  - 構造は保存

Level B: 構造変換（トポロジー変更）
  - ノード追加/削除
  - エッジ追加/削除
  - 構造が変化

Level C: 概念変換（抽象化/具体化）
  - ハイパーノード化（複数ノード → 1ノード）
  - 展開（1ノード → 複数ノード）
  - 抽象度の変更
```

### 2.3 実装の優先順位

**Phase 1: 表層変換（Level A）から始める**

理由：
- 最も単純で検証しやすい
- 既存のグラフ同型アルゴリズムが使える
- 失敗した場合の原因特定が容易

---

## 3. アルゴリズム設計

### 3.1 基本アプローチ: ノードマッピング最適化

```python
def find_isomorphism_transform(G1: Graph, G2: Graph) -> Transform:
    """
    G1をG2に近づける最適な変換を発見する

    Returns:
        Transform: ノードマッピング + 必要な編集操作
    """
    # Step 1: 構造類似度で候補ペアを絞り込む
    node_pairs = find_candidate_mappings(G1, G2)

    # Step 2: 各マッピングでGEDを計算
    best_mapping = None
    best_ged = float('inf')

    for mapping in enumerate_mappings(node_pairs):
        # マッピングを適用したG1'を作成
        G1_transformed = apply_mapping(G1, mapping)

        # GEDを計算
        ged = compute_ged(G1_transformed, G2)

        if ged < best_ged:
            best_ged = ged
            best_mapping = mapping

    # Step 3: 残差編集操作を抽出
    edit_ops = extract_edit_operations(G1, G2, best_mapping)

    return Transform(mapping=best_mapping, edits=edit_ops, cost=best_ged)
```

### 3.2 候補マッピングの効率的な探索

**問題**: 全探索は O(n!) で不可能

**解決策**: 構造的特徴で候補を絞り込む

```python
def find_candidate_mappings(G1: Graph, G2: Graph) -> List[Tuple[Node, Node]]:
    """構造的に類似したノードペアを発見"""
    candidates = []

    for n1 in G1.nodes:
        for n2 in G2.nodes:
            # 構造的特徴の類似度
            struct_sim = compute_structural_similarity(n1, n2)

            # 意味的特徴の類似度（埋め込み）
            semantic_sim = cosine(embed(n1), embed(n2))

            # 次数の類似度
            degree_sim = 1 - abs(degree(n1) - degree(n2)) / max(degree(n1), degree(n2))

            # 総合スコア
            score = α * struct_sim + β * semantic_sim + γ * degree_sim

            if score > threshold:
                candidates.append((n1, n2, score))

    return sorted(candidates, key=lambda x: -x[2])
```

### 3.3 ビームサーチによる効率化

```python
def beam_search_isomorphism(G1: Graph, G2: Graph, beam_width: int = 10) -> Transform:
    """ビームサーチで効率的に最適マッピングを探索"""

    # 初期状態: 空のマッピング
    beam = [PartialMapping(mapped={}, remaining=G1.nodes)]

    while beam[0].remaining:
        next_beam = []

        for partial in beam:
            # 次にマップするノードを選択
            next_node = partial.remaining[0]

            # 候補となるターゲットノード
            for target in get_candidates(next_node, G2, partial.mapped):
                new_mapping = partial.extend(next_node, target)
                new_mapping.score = evaluate_partial_ged(new_mapping, G1, G2)
                next_beam.append(new_mapping)

        # 上位beam_width個を保持
        beam = sorted(next_beam, key=lambda x: x.score)[:beam_width]

    return beam[0].to_transform()
```

---

## 4. 分子設計AIからの技術転用

### 4.1 Scaffold Hoppingアルゴリズム

創薬で使われている手法：

| 手法 | アイデア | 知識グラフへの適用 |
|------|---------|-------------------|
| RECAP | 結合切断ルール | 関係タイプでグラフを分割 |
| BRICS | より精密な切断ルール | ドメイン固有の分割ルール |
| MMP | Matched Molecular Pairs | 最小差異ペアの発見 |
| Graph Transformer | 分子グラフの変換学習 | 知識グラフの変換学習 |

### 4.2 適用案: MMP (Matched Molecular Pairs) の知識グラフ版

```python
def find_matched_knowledge_pairs(corpus: List[Graph]) -> List[Tuple[Graph, Graph, EditOp]]:
    """
    最小編集で変換可能なグラフペアを発見
    (分子設計のMMPアルゴリズムを知識グラフに適用)
    """
    pairs = []

    for i, G1 in enumerate(corpus):
        for G2 in corpus[i+1:]:
            # 単一編集で変換可能か？
            edit = find_single_edit_transform(G1, G2)

            if edit is not None:
                pairs.append((G1, G2, edit))

    return pairs
```

---

## 5. 検証実験の設計

### 5.1 実験1: 合成データでの検証

**目的**: アルゴリズムが正しく動作するか確認

```python
def synthetic_isomorphism_test():
    """既知の変換を持つグラフペアで検証"""

    # 元グラフを作成
    G_original = create_test_graph()

    # 既知の変換を適用
    known_transform = Transform(
        mapping={"A": "X", "B": "Y", "C": "Z"},
        edits=[AddEdge("X", "Z")]
    )
    G_transformed = apply_transform(G_original, known_transform)

    # アルゴリズムで変換を発見
    discovered_transform = find_isomorphism_transform(G_original, G_transformed)

    # 検証
    assert discovered_transform.mapping == known_transform.mapping
    assert discovered_transform.edits == known_transform.edits
```

**成功基準**: 100%の正解率（合成データなので）

### 5.2 実験2: 科学史の再現（拡張版）

**目的**: Level 2で成功した科学史シナリオをLevel 3で再検証

| シナリオ | 入力 | 期待される変換 |
|---------|------|---------------|
| ボーア | 太陽系グラフ, 原子グラフ | {太陽→原子核, 惑星→電子, 重力→電磁力} |
| ケクレ | ウロボロスグラフ, 分子グラフ | {蛇→炭素鎖, 輪→環状構造} |
| ダーウィン | マルサスグラフ, 進化グラフ | {人口→個体数, 資源競争→生存競争} |

**成功基準**: 3/3シナリオで意味のある変換を発見

### 5.3 実験3: 未知のアナロジー発見

**目的**: 人間が気づいていないアナロジーを発見できるか

```python
def discover_novel_analogies(knowledge_base: List[Graph]) -> List[AnalogySuggestion]:
    """
    知識ベース内で、まだ知られていないアナロジーを発見
    """
    suggestions = []

    for G1, G2 in all_pairs(knowledge_base):
        # 異なるドメインのグラフのみ
        if G1.domain == G2.domain:
            continue

        # 変換を探索
        transform = find_isomorphism_transform(G1, G2)

        # 低コストの変換 = 潜在的なアナロジー
        if transform.cost < threshold:
            suggestions.append(AnalogySuggestion(
                source=G1,
                target=G2,
                transform=transform,
                novelty_score=compute_novelty(G1, G2)
            ))

    return sorted(suggestions, key=lambda x: -x.novelty_score)
```

**成功基準**: 専門家が「面白い」と評価するアナロジーを1つ以上発見

---

## 6. 実装ロードマップ

### Week 1-2: 基盤実装
- [ ] Transform クラスの定義
- [ ] ノードマッピング探索（brute force）
- [ ] GED計算の統合
- [ ] 合成データテスト

### Week 3-4: 効率化
- [ ] 候補絞り込み（構造的特徴）
- [ ] ビームサーチ実装
- [ ] 計算時間のベンチマーク

### Week 5-6: 検証実験
- [ ] 科学史シナリオの再検証
- [ ] 新規アナロジー発見実験
- [ ] 結果の分析と論文への反映

---

## 7. 技術的チャレンジ

### 7.1 計算量の壁

**問題**: GED計算はNP困難

**対策**:
1. 構造的特徴で候補を絞り込む（前処理）
2. 近似アルゴリズム（ビームサーチ、遺伝的アルゴリズム）
3. グラフサイズの制限（小さなサブグラフから始める）

### 7.2 意味的妥当性

**問題**: 構造的に最適な変換が意味的に正しいとは限らない

**対策**:
1. 意味的類似度を制約として追加
2. 人間によるフィルタリング
3. ドメイン知識の活用

### 7.3 評価の困難さ

**問題**: 「正しい変換」の定義が曖昧

**対策**:
1. 合成データでの検証（ground truth あり）
2. 科学史の再現（既知のアナロジー）
3. 専門家評価（新規発見）

---

## 8. 成功の定義

### Minimum Viable Success
- 合成データで100%正解
- 科学史シナリオ3/3で意味のある変換を出力

### Strong Success
- 上記 + 新規アナロジーを1つ以上発見
- 計算時間が実用的（1000ノードグラフで1分以内）

### Breakthrough Success
- 専門家が「これは新しい洞察だ」と認める発見
- 論文として発表可能な結果

---

## 9. 次のアクション

1. [x] 設計文書作成（本ドキュメント）
2. [ ] Transform クラスの実装
3. [ ] 合成データテストの作成
4. [ ] 基本アルゴリズムの実装
5. [ ] 科学史シナリオでの検証
