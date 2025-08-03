# レビューへの返答と実験結果

## レビューの要点

1. **行動決定が「勾配」を使っていない** → 候補行動ごとのΔGED/ΔIG計算を実装
2. **未探索ボーナスはチートか？** → 自己観測なら正当（count-based exploration）
3. **スケール不一致** → log1p圧縮で対処

## 実装結果

### Version 1: 過去の経験の加重平均（元実装）
```python
# 類似エピソードのgeDIG値を加重平均
weighted_value += ep.gedig_value * similarity
```
- カバレッジ: **26.8%**
- 最大訪問回数: 129回

### Version 2: 真の勾配ベース（レビュー提案）
```python
# 候補行動ごとにΔGED/ΔIG予測
ged_hat, ig_hat = self.estimate_deltas(current_pos, action)
score = math.log1p(ged) * math.log1p(ig)
```
- カバレッジ: **12.2%**（悪化！）
- 最大訪問回数: 530回（悪化！）

## なぜ勾配ベースが失敗したか

1. **過度の局所性**
   - 「今の一歩」だけを評価すると、近視眼的になる
   - 過去の経験の「蓄積効果」を失う

2. **探索と活用のバランス崩壊**
   - ΔGED/ΔIGが急速に減衰し、どの行動も低スコアに
   - 結果として同じ場所でループ

3. **Look-aheadのオーバーヘッド**
   - 2手先計算が重く、効果も限定的

## 結論：ハイブリッドアプローチが最適

```python
def decide_action_hybrid(self, current_pos, possible_actions):
    for action in possible_actions:
        # 1. 過去の経験から基礎スコア
        base_score = self.get_experience_score(current_pos, action)
        
        # 2. 勾配による調整（軽量版）
        ged, ig = self.estimate_deltas(current_pos, action)
        gradient_boost = math.log1p(ged * ig) * 0.3  # 30%の重み
        
        # 3. 統合スコア
        action_scores[action] = base_score + gradient_boost
```

## レビューアーへの返答

> 「現ロジックは経験値テーブルを参照して次手を選ぶ設計」

**その通りです！** しかし実験の結果、純粋な勾配ベースより**経験値テーブル＋勾配ブースト**のハイブリッドが有効と判明しました。

> 「未探索ボーナスはチートか？」

**正当です。** Count-based explorationは強化学習の標準手法（Bellemare et al. 2016）。重要なのは：
- エージェント自身の観測に基づく
- 滑らかな減衰関数 `1/(1+log(1+v))`
- ΔIGに自然に組み込む

## 次のステップ

1. **ハイブリッド実装**：経験＋勾配の良いとこ取り
2. **パラメータ最適化**：Optunaでged_weight/ig_weight調整
3. **アブレーション実験**：各要素の寄与を定量化

レビューは理論的に正しいが、実装では**実用性とのバランス**が重要でした！