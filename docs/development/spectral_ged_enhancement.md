# スペクトルGED拡張提案

## 背景

現在のGED実装は基本的な隣接構造ベースの編集距離を採用していますが、構造の「質的変化」を評価できません。
レビュアーの提案に基づき、ラプラシアン行列の固有値スペクトルを用いた構造評価の追加を検討します。

## 理論的根拠

### なぜスペクトル評価が有効か

1. **情報理論との独立性**: ラプラシアン固有値は純粋に構造的な特性であり、IGと重複しない
2. **構造の規則性評価**: 固有値分布から構造の秩序性・安定性を評価可能
3. **数学的健全性**: GEDとIGの相関を低く保ち、geDIGの理論的整合性を維持

## 実装提案

```python
def _calculate_spectral_structure_score(self, g: nx.Graph) -> float:
    """
    ラプラシアン行列の固有値分布から構造スコアを計算
    
    Returns:
        構造の規則性スコア（低いほど規則的）
    """
    if g.number_of_nodes() < 2:
        return 0.0
    
    try:
        # ラプラシアン行列を計算
        L = nx.laplacian_matrix(g).toarray()
        
        # 固有値を計算（実数のみ）
        eigvals = np.linalg.eigvalsh(L)
        
        # 固有値の分散を構造の不規則性指標として使用
        # （分散が小さい = より規則的な構造）
        spectral_score = np.std(eigvals)
        
        return spectral_score
        
    except Exception as e:
        logger.warning(f"Spectral score calculation failed: {e}")
        return 0.0

def _calculate_enhanced_ged(self, g1: nx.Graph, g2: nx.Graph) -> Dict[str, float]:
    """
    スペクトル評価を含む拡張GED計算
    """
    # 既存の隣接構造ベースGED
    base_ged_result = self._calculate_normalized_ged(g1, g2)
    
    # スペクトル構造スコア
    spectral_before = self._calculate_spectral_structure_score(g1)
    spectral_after = self._calculate_spectral_structure_score(g2)
    
    # 構造の規則性改善（スコアが減少 = より規則的に）
    spectral_improvement = spectral_before - spectral_after
    
    # 最終的なGED = 基本GED - α × 構造改善
    alpha = 0.3  # スペクトル重み（調整可能）
    enhanced_ged = base_ged_result['normalized_ged'] - alpha * np.tanh(spectral_improvement)
    
    return {
        **base_ged_result,
        'spectral_improvement': spectral_improvement,
        'enhanced_ged': np.clip(enhanced_ged, -1.0, 1.0)
    }
```

## 期待される効果

1. **構造の質的改善を評価可能に**
   - ノード追加により構造が整理される場合、GEDが減少
   - 単なるノード数増加では改善しない

2. **IGとの明確な分離**
   - スペクトル特性は情報量と独立
   - geDIGの理論的健全性を維持

3. **より洗練された洞察検出**
   - 構造的に意味のある変化を検出
   - ノイズ的な変化を抑制

## 実装優先度

- **Priority**: Medium
- **Complexity**: Low-Medium
- **Impact**: High（理論的完成度の向上）

## 参考文献

- Spectral Graph Theory (Fan Chung, 1997)
- Graph Signal Processing (Shuman et al., 2013)