# 実験設計補強計画 - 論文品質向上ロードマップ

## 🎯 現状評価：8/10 → 9.5/10 への道筋

### ✅ 既に優秀な点（維持）
- **多様なベースライン**: Q-Learning, ε-greedy, UCB, Random
- **統計手法の妥当性**: Shapiro-Wilk → Welch t-test → 多重比較補正 → Cohen's d
- **再現パイプライン**: ワンコマンド実験→自動集計→ダッシュボード
- **洞察可視化**: ΔGED/ΔIGタイムライン、EurekaSpike発火記録

### ✅ **COMPLETED: 実装済み補強項目**

#### ✅ 1. アブレーション実験フレームワーク - COMPLETED
**実装済み**: `experiments/ablation_study_framework.py`
- ✅ geDIG OFF (LLM + 外発報酬のみ)
- ✅ LLM OFF (geDIG内発報酬 + 外発報酬)  
- ✅ C-value Boost OFF (geDIG + LLM、記憶強化なし)
- ✅ Memory Conflict Penalty OFF
- ✅ Full InsightSpike (All ON)
- ✅ 統計的有意性テスト（Cohen's d計算）
- ✅ コンポーネント貢献度分析

#### ✅ 2. 統計可視化強化 - COMPLETED
**実装済み**: `scripts/colab/advanced_experimental_visualization.py`
- ✅ ダッシュボードにp値stars (*, **, ***)
- ✅ Cohen's d数値を図中表示
- ✅ 信頼区間エラーバー（適切な誤差伝播）
- ✅ 効果量解釈ライン (0.2=small, 0.5=medium, 0.8=large)
- ✅ Welch's t-test（不等分散対応）
- ✅ 統計的検定力分析

#### ✅ 3. ハイパーパラメータ公平性 - COMPLETED
**実装済み**: `experiments/fair_hyperparameter_optimization.py`
- ✅ 全アルゴリズム統一ベイズ最適化（Optuna）
- ✅ 探索範囲・試行回数の明示（透明性保証）
- ✅ チューニング手順の透明化
- ✅ 収束曲線と効率性分析

#### ✅ 4. RAG系実験包括分析 - COMPLETED  
**実装済み**: `experiments/rag_enhanced_experiment_framework.py`
- ✅ Multi-retriever比較（BM25, DPR, Hybrid-RAG, InsightSpike-RAG）
- ✅ Document-level precision/recall分析
- ✅ Cost-performance trade-off定量化
- ✅ Temporal knowledge drift（HotpotQA-Chronos）
- ✅ 検索器バイアス抑制（異種検索器）

#### ✅ 5. Continual Learning実験 - COMPLETED
**実装済み**: `experiments/continual_learning_experiment_framework.py`  
- ✅ Split-MNIST benchmark (Task-IL vs Class-IL)
- ✅ Forgetting measures（max acc_old - acc_now）
- ✅ FIFO vs LRU vs C-value vs InsightSpike比較
- ✅ Memory効率曲線（性能 vs ストレージ）
- ✅ Memory node lifetime dynamics
- ✅ 動的insight→memory統合可視化

### 🔥 残存補強項目（論文提出前推奨）

#### 4. ドメイン多様性 - Priority: MEDIUM
**問題**: 迷路タスク単一ドメイン依存
**解決策**:
- [ ] MountainCar環境追加
- [ ] 25マス扉開閉パズル追加  
- [ ] クロスドメイン性能比較

#### 5. 過学習チェック - Priority: MEDIUM
**問題**: Train/Test分離なし
**解決策**:
- [ ] 迷路マップTrain/Test分割
- [ ] 初見マップでの性能測定
- [ ] 一般化性能の定量評価

### 🧪 RAG実験補強（アカデミック訴求）

#### 現状の強み
- ✅ Static RAG vs InsightSpike-RAG比較枠組み
- ✅ HotpotQA + TriviaQA標準データセット
- ✅ EM/F1 + Top-k HitRate + ΔGED/ΔIGスパイク数

#### 補強ポイント
- [ ] **検索器多様性**: BM25, DPR baseline追加
- [ ] **時間分割テスト**: Wiki 2018→2025知識更新検証
- [ ] **計算コスト測定**: Latency, Token消費量比較
- [ ] **Retrieval Precision/Recall**: 文書単位精度評価

### 🧠 動的記憶実験補強（継続学習）

#### 現状の強み  
- ✅ C-value記憶保持/破棄ポリシー
- ✅ 記憶ヒット率・忘却率測定
- ✅ ΔGED/ΔIG×Memoryサイズ可視化

#### 補強ポイント
- [ ] **標準CLベンチ**: Split-MNIST, Atari Hard-Switch
- [ ] **忘却率指標**: Forgetting Measure = max acc_old - acc_now
- [ ] **Memory比較**: C-value vs FIFO vs LRU統計比較
- [ ] **Memory効率**: 使用量(MB) vs 成績トレードオフ

## 🎯 実装スケジュール

### Phase 1: 緊急補強（論文提出前）
1. **アブレーション実験**: 5パターン×統計検定
2. **統計可視化**: p値stars, Cohen's d, エラーバー
3. **ハイパーパラメータ**: 公平性担保・手順明示

### Phase 2: 学術補強（査読対応）  
1. **ドメイン拡張**: MountainCar, パズル環境
2. **過学習チェック**: Train/Test分離
3. **RAG多様化**: BM25, DPR, 時間分割

### Phase 3: 完成度向上（Revision対応）
1. **継続学習**: Split-MNIST, 忘却率測定
2. **Memory比較**: C-value優位性定量化
3. **再現性**: Zenodo/OSFアーカイブ

## 🏆 ゴールライン（9.5/10品質）

### RAG系必達目標
- ✅ ベースライン3系統 + アブレーション + 時間分割
- ✅ Top-k Hit/EM/計算コストで有意優位性
- ✅ p < 0.01, Cohen's d > 0.5

### 動的記憶系必達目標  
- ✅ Split-MNIST等でForgetting < 10%
- ✅ C-value > FIFO/LRU統計有意
- ✅ Memory効率トレードオフ明示

### 論文提出可能基準
- ✅ アブレーション結果表完備
- ✅ 効果量・信頼区間図表注記
- ✅ ハイパーパラメータ探索手順明示
- ✅ 全コード・ログアーカイブ

**推定実装期間**: Phase 1 (3-4日), Phase 2 (1週間), Phase 3 (1週間)
**現在の完成度**: 8.0/10 → **目標**: 9.5/10
