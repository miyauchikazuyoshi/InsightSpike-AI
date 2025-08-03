# 論文構成案2（ChatGPT提案）

## タイトル案
**Free-Energy Inspired Intrinsic Reward for Autonomous Graph Agents**  
—ΔIG – ΔGED が駆動する"思考フロー"の実証

## Abstract（150-200 words）
- 問題提起：既存 GNN/RL は外発報酬頼みで「何を」考えるかまで指示が要る
- 提案：FEP を ΔIG–ΔGED に写像→ intrinsic reward として採用
- 仮説：報酬符号が正なら自然連想、反なら破綻
- 実験：Newton-リンゴなど 3 タスク + 符号逆転 A/B テスト
- 結果：正符号系で ΔIG ↑, ΔGED ↓, 意味一致率 +x%
- 貢献：
  1. FEP-GNN 対応の初実証
  2. 符号 CI スクリプト OSS
  3. "思考フロー可視化" ダッシュボード公開

## 1. Introduction（1.5-2 ページ）
- P1：AI はタスク指向だが自然思考は連想的。一歩内側へ踏み込む必要がある
- P2：FEP 概要 → accuracy–complexity の差を最小化が生物的合理性
- P3：ΔIG ↔ accuracy、ΔGED ↔ complexity と同型 ⇒ 勾配則で学習可能
- P4：本研究の問い：この内発報酬だけで GNN は自然連想を再現できるか？
- P5：貢献箇条書き

## 2. Method
### 2.1 ΔIG – ΔGED 報酬
- 数式：R_t = λ * ΔIG_t - μ * ΔGED_t
- 2段正規化 (Churn / Accretion)
- FEP との写像表（Accuracy→ΔIG, Complexity→ΔGED, Free Energy→–R）

### 2.2 符号逆転テスト
- Maximizer vs Minimizer
- 同シード・同ハイパラで鏡対称学習曲線なら実装 OK
- OSS CI 例（GitHub Actions）

### 2.3 GNN エージェント
- Encoder = GraphSAGE（例）→ policy head, value head
- ΔIG：Whitened SimCSE cosine
- ΔGED：O(E) 近似 + α/β キャリブレーション

## 3. Experiments
### E1：静的概念連想
- "物理→ニュートン→リンゴ" シナリオ
- 評価：CosSim(LLM output, gold)

### E2：Insight-RAG mini
- 50 QA ドメイン
- 自動評価：Top-k recall + human coherence

### E3：Ablation
- λ/μ スイープ, Whitening on/off
- 指標：ΔIG 分布, grad norm, reward trend

各セットで Maximizer vs Minimizer 曲線 + テーブル  
Figure 1: 同心円スケッチ→SVG 清書

## 4. Results
- 符号テスト：正符号系は reward↑, ΔGED↓, 収束；負符号は発散 or NaN
- 連想タスク：CosSim +12 pt、human「自然」評価 0.86 → 0.22
- Ablation：Whitening で ΔIG variance ↓30%、λ=1, μ=0.5 が最安定

## 5. Discussion
- "アハ体験" vs "連続連想" — スパイクでなく微分則の積分で洞察が現れる
- Transformer MHA との比較 — アテンションは "一次で完結"；本手法は逐次勾配で深さを作る
- 応用余地 — タスクフリーのコ・パイロット、教育対話、RAG フィルタリング

## 6. Limitations
- デコーダは汎用 LLM；専用 graph-to-text が未構築
- Sentence-BERT 空間の"意味勾配"は経験則依存
- 大規模グラフ（>10k ノード）は O(E) 近似でも未検証

## 7. Future Work
- 自前デコーダ（アテンション付き GNN→Transformer）
- 画像・マルチモーダル ΔIG（CLIP 空間で検証）
- Large-Scale graph RAG エンジンへの組込み

## 8. Conclusion
We demonstrated that a simple intrinsic reward, equivalent to the negative of variational free energy, enables a graph neural agent to self-organise its thought flow without external supervision.