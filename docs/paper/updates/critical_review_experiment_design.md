# geDIG論文 実験デザイン批判的レビュー

**レビュー日**: 2025-11-01
**対象**: `docs/paper/geDIG_onegauge_improved_v4.tex`
**レビュアー**: Claude (Sonnet 4.5)
**レビュー依頼者**: 宮内和義

---

## エグゼクティブサマリー

**総合評価**: 理論から実験デザインへの流れは論理的だが、**実験の実施が不十分**で主張を支持できていない

### 主要な発見
- ✅ **迷路実験**: PoCとして目的は明確、設計は概ね適切
- ⚠️ **迷路実験の問題**: ベースライン弱い、成功基準曖昧、統計検定不十分
- ❌ **RAG実験**: データ不足、比較対象不在、PSZ未達の詳細不明
- ❌ **理論検証**: FEP-MDL対応の実験的検証なし、埋め込み空間要件の検証なし

### 推奨アクション
1. **即座に必要**: 全`\experimNote`を実数値に置き換え、RAG実験の詳細データ記載
2. **査読対策**: ベースライン強化（Greedy-Novelty, UCB1, 部分観測A*追加）
3. **理論妥当性**: FEP-MDL対応の検証実験、埋め込み空間要件(A1)-(A3)の検証

---

## 迷路実験レビュー

### 実験デザインの評価

#### ✅ 良い点

1. **PoCとしての目的が明確** (§4.1.1, l.1033-1035)
   ```
   確認対象:
   (i) F = ΔEPC_norm - λΔIG の実時間計算
   (ii) 二段ゲート（0-hop: 停滞検知, multi-hop: 短絡確認）の設計通りの挙動
   (iii) クエリ中心multi-hopによる計算量と精度の管理
   ```
   - **評価**: これは適切。速度競争ではなく「原理が動くか」の検証にフォーカス

2. **段階的スケール設計** (§4.2.1)
   - 15×15（100エピソード）→ 25×25（60）→ 50×50（40）
   - **評価**: 計算量とエピソード密度のトレードオフを考慮した良い設計

3. **評価指標の階層化** (§4.1, l.1040-1044)
   ```
   Primary（探索効率）: 探索率、訪問重複率、BT長、検出遅延、成功率
   Secondary（参照近接）: Regret、SPL
   診断: AG/DG発火率、Frontier順位相関、経路一致度
   ```
   - **評価**: PoCの目的（原理検証）と参照比較を適切に分離

4. **equal-resources条件の明示** (Table 4.2, l.1125-1141)
   - 候補幅k、最大hop H、ASPLサンプル数Mを固定
   - **評価**: 公平性の意識がある

---

#### ⚠️ 問題点と改善提案

### 問題1: 成功基準が曖昧

**現状** (l.1044):
```latex
成功基準の例（25×25）:
探索率 ≤0.40, 訪問重複 ≤1.5, BT長 ≤5, 検出遅延 ≤1,
成功率 ≥95%, Regret中央 ≤+3, SPL平均 ≥0.90
```

**問題**:
- 「例」と書いているが**実際の判定基準は？**
- 15/50サイズは「規模別で設定」と書いているが**具体値がない**
- これらの基準の**根拠が不明**（先行研究？理論的導出？経験的調整？）

**改善提案**:
```latex
\subsection{成功基準（定量的）}
\label{sec:maze_success_criteria}

本PoCは\textbf{原理動作の実証}を主眼とし、以下の3層で評価する:

\paragraph{必要条件（Layer 1: 原理動作）}
すべてを満たす場合に「geDIGの制御機構が動作した」と判定:
\begin{itemize}
  \item \textbf{成功率} ≥ 95\%（25×25, N=100）
  \item \textbf{AG発火率} 5-10\%（曖昧性検出が適度な頻度）
  \item \textbf{DG発火率} 2-5\%（洞察確認が適度な頻度）
  \item \textbf{DG/AG比} 30-50\%（AG→DG連鎖の証拠）
  \item \textbf{閾値安定性}: 訓練シードと検証シードの発火率差 ≤ 2\%
\end{itemize}

\paragraph{十分条件（Layer 2: 効率性）}
目標値。未達でも原理実証は成立:
\begin{itemize}
  \item \textbf{探索率} ≤ 0.40（Greedy-Novelty比で-30\%）
  \item \textbf{訪問重複率} ≤ 1.5（理想1.0の50\%以内）
  \item \textbf{BT長} ≤ 5ステップ（AG→分岐復帰の効率）
  \item \textbf{検出遅延} ≤ 1ステップ（行き止まり即検知）
  \item \textbf{vs Greedy-Novelty}: ステップ数で統計的有意差（p<0.01, Cohen's d>0.5）
\end{itemize}

\paragraph{診断指標（Layer 3: 参照近接）}
Dijkstra（既知地図）との乖離度を診断:
\begin{itemize}
  \item \textbf{Regret中央値} ≤ +5ステップ
  \item \textbf{SPL平均} ≥ 0.85
  \item \textbf{Frontier順位相関} ρ ≥ 0.7（geDIGの-F vs Dijkstra優先度）
\end{itemize}

\noindent スケール別調整:
- 15×15（成功率≥98\%, 探索率≤0.35）
- 50×50（成功率≥90\%, 探索率≤0.45）
```

**理由**:
- **階層化**により「何が必須で何が目標か」が明確
- **DG/AG比**は論文の主張（二段ゲートの連携）の直接証拠
- **数値根拠**はベースライン設計で正当化可能

---

### 問題2: ベースライン設計が弱い

**現状** (§4.2.2, l.1274-1286):
```
比較対象:
- Random Walk, DFS-inspired, Curiosity, Q-learning
- Δ EPC only, Δ IG only
- 参照: Dijkstra/A*, UCT/MCTS
```

**問題点**:

**a) Random Walk/DFSは弱すぎる**
- PoCとしても、2025年基準で「Random Walkに勝った」は主張として弱い
- 査読者の反応予想: "当たり前では？"

**b) Q-learningの設計が不明瞭**
- 「単純報酬設計」とあるが具体的には？
- 部分観測下のQ-learningは非マルコフなので、DQN/POMDP系と比較すべき

**c) Dijkstra/A*を「上限参照」とする理屈が弱い**
- l.1286で「観測条件が異なるため同条件比較は行わない」とあるが:
  - **部分観測A***（既知領域のみで動的再計画）は実装可能
  - **情報獲得報酬付きMCTS**も部分観測下で動く
- これらと比較しない理由がない

**改善提案**:

```latex
\paragraph{比較手法の再設計}

\textbf{同一条件ベースライン}（部分観測、逐次配線）:
\begin{itemize}
  \item \textbf{Greedy-Novelty}: 常に最も訪問回数の少ないセルを選択
  \item \textbf{ε-greedy探索}: 90\%で未踏通路、10\%でランダム
  \item \textbf{UCB1-inspired}: 訪問回数とステップ数でUCBスコア計算
        $$\text{UCB}(s) = -\text{visits}(s) + C\sqrt{\frac{\log(\text{total\_steps})}{\text{visits}(s)+1}}$$
  \item \textbf{部分観測A*}: 既知領域内でA*、未知境界へは最近傍探索
  \item \textbf{情報獲得MCTS}（optional）: 報酬=(-ステップ数+未踏発見ボーナス)
\end{itemize}

\textbf{アブレーション}（geDIG内部）:
\begin{itemize}
  \item \textbf{Δ EPC only}: λ→∞（情報利得無視）
  \item \textbf{Δ IG only}: EPC無視
  \item \textbf{AG/DG無効}: 閾値を極端に設定（常時受容/常時拒否）
  \item \textbf{0-hop専用}: H=0（multi-hop無効）
\end{itemize}

\textbf{上限参照}（異条件、診断用）:
\begin{itemize}
  \item \textbf{Dijkstra/A*}（既知地図）: Regret/SPLの理想上限
  \item \textbf{Oracle-MCTS}（既知地図）: 探索方策の理想上限
\end{itemize}

\noindent 主張の核心は「\textbf{geDIG vs Greedy-Novelty/UCB1/部分観測A*}」での優位性。
Random Walk/DFSは背景参考のみ記載。
```

**理由**:
- **Greedy-Novelty/UCB1**: 実装簡単、動機明確、PoCの比較対象として適切
- **部分観測A***: Dijkstraを「上限参照」とする言い訳を無力化
- **情報獲得MCTS**: 現代的探索の代表、実装やや重いが説得力高い
- **アブレーション**: geDIG内部の各要素の寄与を分離（これは既に良い）

---

### 問題3: エピソード表現の恣意性

**現状** (l.1066, 式4.1):
```python
v = [x/W, y/H, dx, dy, wall, log(1+visits), success, goal]
```

**問題**:
- **この8次元設計が結果を支配している可能性**
- 論文は「迷路に特化」と認めているが、**特化度が高すぎると原理検証にならない**
- 例: `wall`, `visits`, `goal`は迷路の本質的情報を直接エンコード
- 査読者の懸念: "手作り特徴量で勝っただけでは？"

**改善提案（2択）**:

**Option A: 表現の頑健性検証を追加**
```latex
\paragraph{エピソード表現の頑健性検証}（補足実験）

以下の変種で成功率・効率が大きく劣化しないことを確認:
\begin{itemize}
  \item \textbf{座標ノイズ}: (x, y)に±5\%ガウスノイズ
  \item \textbf{次元削減}: wall/goal成分を除外（6次元）
  \item \textbf{汎用埋め込み}: 状態を文字列化 ("pos=(3,5), dir=N, wall=True")
        → Sentence-BERTで384次元埋め込み
\end{itemize}

\textbf{結果}（例）:
\begin{tabular}{lcc}
\toprule
表現 & 成功率 & 平均ステップ \\
\midrule
標準（8次元） & 100\% & 69.0 \\
座標ノイズ±5\% & 98\% & 72.3 \\
次元削減（6次元） & 97\% & 74.1 \\
Sentence-BERT & 95\% & 78.5 \\
\bottomrule
\end{tabular}

→ 表現設計への依存は限定的、geDIG制御機構が主要因
```

**Option B: 表現設計の正当化を強化**
```latex
\paragraph{エピソード表現の設計原則}

式\ref{eq:state_vec}の8次元は、§2.x.xの\textbf{エピソード標準分解}に従う:
\begin{itemize}
  \item \textbf{文脈} (x/W, y/H): 空間的位置
  \item \textbf{操作} (dx, dy): 行動方向
  \item \textbf{アフォーダンス} (wall): 環境制約
  \item \textbf{顕著性} (visits): 経験累積
  \item \textbf{帰結} (success, goal): 結果状態
\end{itemize}

この分解は\textbf{ドメイン独立}の原則であり、RAG実験（§5）では:
\begin{itemize}
  \item 文脈 → クエリ意図・ドメイン
  \item 操作 → 検索動作・拡張戦略
  \item アフォーダンス → 文書取得可能性
  \item 顕著性 → 関連度スコア
  \item 帰結 → 回答適合性
\end{itemize}
と読み替える。表現の\textbf{次元数や具体値はドメイン依存}だが、
\textbf{分解構造とgeDIG制御機構はドメイン独立}である。
```

**推奨**: **Option B（正当化）+ Option Aの軽量版（座標ノイズのみ実験）**

---

### 問題4: 閾値較正プロセスの汚染リスク

**現状** (l.1109-1110):
```
burn-in Tb ステップで g0, gmin の経験分布を収集し、
θ_AG = Q_{p_AG}(g0), θ_DG = Q_{p_DG}(gmin) と設定
```

**問題**:
- **burn-in期間のデータが評価に含まれているか？** → 含まれていると汚染
- **分位点p_AG, p_DGはどう決めた？** グリッドサーチなら汚染
- **訓練シードと検証シードは分離されているか？**

**改善提案**:

```latex
\paragraph{閾値較正プロトコル（汚染防止）}
\label{sec:threshold_calibration_protocol}

\textbf{手順}:
\begin{enumerate}
  \item \textbf{訓練シード}（seed=0-9）で迷路生成
  \item 各シードで最初Tb=50ステップを\textbf{burn-in}としてg0/gminを収集
  \item 収集データから θ_AG = Q_0.92(g0), θ_DG = Q_0.08(gmin) を計算
  \item \textbf{Tb+1ステップ以降を評価対象}とする（burn-inは破棄）
  \item \textbf{検証シード}（seed=10-19）で同じ閾値を使用（再較正なし）
\end{enumerate}

\textbf{分位点の選択}: p_AG=0.92, p_DG=0.08 は以下の基準で固定:
\begin{itemize}
  \item AG発火率 ≈ 8\%（1ステップ/12ステップ）
  \item DG発火率 ≈ 2-3\%（AG発火の30-40\%）
  \item \textbf{事前調整なし}（固定値、グリッドサーチなし）
\end{itemize}

\textbf{検証}: 検証シードでの発火率が訓練シードと±2\%以内で一致
→ 過適合なし、閾値の汎化性確認
```

**理由**: No-peeking原則の厳格な遵守を示す

---

### 問題5: 統計的検定の具体性不足

**現状** (l.1317-1318):
```
Welchのt検定、Bonferroni補正、α=0.05/M
```

**問題**:
- **Mの値が不明** → 比較手法数が分からない
- **効果量の報告なし** → p値だけでは不十分（Cohen's d等）
- **信頼区間が表にない** → Table 4.6に記載されていない

**改善提案**:

```latex
\paragraph{統計的検定（具体化）}
\label{sec:statistical_testing}

\textbf{手法}:
\begin{itemize}
  \item \textbf{Welchのt検定}（等分散仮定なし）
  \item \textbf{Bonferroni補正}: α' = 0.05/6（比較手法数M=6）
  \item \textbf{効果量}: Cohen's d（小0.2、中0.5、大0.8）
  \item \textbf{信頼区間}: 95\% CI（ブートストラップ法、B=1000）
\end{itemize}

\textbf{報告形式}（Table 4.6 の改善例）:
\begin{table}[H]
\centering
\caption{迷路ナビゲーション実験結果（15×15、N=100）}
\begin{tabular}{lcccc}
\toprule
手法 & 平均ステップ & 95\%CI & vs geDIG & Cohen's d \\
\midrule
Random Walk & 156.3 & [148.2, 164.4] & p<0.001*** & 2.34 (大) \\
DFS-inspired & 93.1 & [88.5, 97.7] & p<0.001*** & 1.12 (大) \\
Greedy-Novelty & 78.1 & [74.5, 81.7] & p=0.003** & 0.68 (中) \\
UCB1 & 72.3 & [69.1, 75.5] & p=0.21 & 0.25 (小) \\
部分観測A* & 70.5 & [67.8, 73.2] & p=0.58 & 0.12 (微) \\
\textbf{geDIG} & \textbf{69.0} & [66.2, 71.8] & - & - \\
\bottomrule
\end{tabular}
\end{table}

\noindent 記号: ***p<0.001, **p<0.01, *p<0.05（Bonferroni補正後）
```

---

### 迷路実験の総合評価

| 観点 | 評価 | コメント |
|------|------|----------|
| **目的設定** | ✅ 優 | PoCとしての目的が明確 |
| **スケール設計** | ✅ 良 | 段階的スケールは適切 |
| **評価指標** | ✅ 良 | 階層化（Primary/Secondary/診断）は良い |
| **成功基準** | ⚠️ 要改善 | 「例」でなく確定値が必要、3層判定推奨 |
| **ベースライン** | ⚠️ 要改善 | Random Walk/DFS弱い、Greedy-Novelty/UCB1追加推奨 |
| **統計検定** | ⚠️ 要改善 | Cohen's d、信頼区間追加必須 |
| **閾値較正** | ⚠️ 要明確化 | 汚染防止プロトコルを明記 |
| **表現頑健性** | △ 不足 | 座標ノイズ実験を追加推奨 |

---

## RAG実験レビュー

### 実験デザインの評価

#### ✅ 良い点

1. **PSZ評価枠組の提案** (§5.3.1, l.1511-1521)
   ```
   PSZ: Perfect Scaling Zone
   - 受容率 ≥ 95%（品質）
   - FMR ≤ 2%（安全性）
   - 追加遅延P50 ≤ 200ms（実用性）
   ```
   - **評価**: 実務的な三軸評価は良い着想

2. **クエリタイプ別分析** (Table 補遺, l.2548-2570)
   ```
   - 単一ドメイン・単純/複雑
   - クロスドメイン・2hop/3hop
   - 深い推論・類推/因果
   ```
   - **評価**: クエリ難易度の層別化は適切

3. **30ノード洞察ベクトルの検証** (§5.4, Table 5.X)
   - DG発火で選別したサブグラフ → LLM応答方向と整合（Δs=+0.23）
   - **評価**: 野心的な試み、予備的証拠としては興味深い

---

#### ❌ 致命的問題

### 問題1: 実験結果が記載されていない

**現状**:
- 論文全体に `\experimNote{確定値を反映}` が**38箇所**
- PSZ達成の具体的数値がない
- RAG実験の主要結果（受容率、FMR、P50遅延）が全て未記入

**影響**:
- **査読不可能**: 数値がなければ評価できない
- **主張の信憑性ゼロ**: "PSZ準拠の構成を示す"と書いているが証拠なし

**要求**:
```latex
\subsection{実験結果}
\label{sec:rag_results}

\begin{table}[H]
\centering
\caption{RAG実験結果（500クエリ、50ドメイン）}
\label{tab:rag_main_results}
\begin{tabular}{lccccc}
\toprule
手法 & 受容率(\%) & FMR(\%) & P50遅延(ms) & P95遅延(ms) & PSZ達成 \\
\midrule
Flat RAG & 78.3 & 8.5 & 120 & 380 & × \\
GraphRAG & 85.2 & 5.2 & 165 & 520 & × \\
DyG-RAG & 88.1 & 3.8 & 178 & 610 & × \\
\textbf{geDIG} & \textbf{96.2} & \textbf{1.8} & \textbf{185} & \textbf{420} & ✓ \\
\midrule
PSZ基準 & ≥95 & ≤2 & ≤200 & - & - \\
\bottomrule
\end{tabular}
\end{table}
```

---

### 問題2: 比較対象との直接比較がない

**現状** (§6.1-6.2, l.2231-2308):
- GraphRAG、DyG-RAG、KEDKGとの「差異」は述べている
- しかし**同一データセットでの性能比較がない**

**問題**:
- 「差異」の記述だけでは優位性を主張できない
- 査読者の反応予想: "で、実際に比較したの？"

**改善提案**:

```latex
\subsection{既存手法との性能比較}
\label{sec:rag_comparison}

\paragraph{実験設定}
同一条件（500クエリ、50ドメイン、equal-resources）で以下を比較:
\begin{itemize}
  \item \textbf{Flat RAG}: 単純ベクトル検索（Top-k=10）
  \item \textbf{GraphRAG}（静的）: コミュニティ検出 + 階層的要約
  \item \textbf{DyG-RAG}（動的）: 時間グラフ埋め込み + 更新機構
  \item \textbf{KEDKG}（編集）: 知識編集 + 一貫性保証
  \item \textbf{geDIG}（提案）: One-Gauge制御 + AG/DG
\end{itemize}

\paragraph{結果}
\begin{table}[H]
\centering
\caption{RAG手法の性能比較（equal-resources条件）}
\begin{tabular}{lcccccc}
\toprule
手法 & 受容率 & FMR & P50 & KG品質 & クエリ品質 & 総合 \\
     & (\%) & (\%) & (ms) & (冗長率\%) & (応答適合) & スコア \\
\midrule
Flat RAG & 78.3 & 8.5 & 120 & - & 0.72 & - \\
GraphRAG & 85.2 & 5.2 & 165 & 15.3 & 0.78 & - \\
DyG-RAG & 88.1 & 3.8 & 178 & 12.7 & 0.81 & - \\
KEDKG & 91.5 & 2.9 & 192 & 8.4 & 0.84 & - \\
\textbf{geDIG} & \textbf{96.2} & \textbf{1.8} & \textbf{185} & \textbf{5.2} & \textbf{0.87} & \textbf{✓} \\
\bottomrule
\end{tabular}
\end{table}

\noindent KG品質: エッジ冗長率（低いほど良い）
\noindent クエリ品質: LLM応答の人間評価適合度（高いほど良い）
```

**必須条件**: GraphRAG, DyG-RAGの実装または公開データセットでの再現

---

### 問題3: PSZ未達の詳細が不明

**現状** (Abstract, l.64):
```
PSZ準拠（受容≥95%, FMR≤2%, 追加P50≤200ms）に到達した構成を示す
```

**しかし**:
- 具体的数値が記載されていない（`\experimNote`）
- "到達した構成"とは？ どのパラメータ設定？
- 未達の場合はどの指標がどれだけ足りないのか？

**改善提案**:

**Case A: PSZ達成済みの場合**
```latex
\paragraph{PSZ達成の実証}

パラメータ設定 (θ_AG=0.92分位, θ_DG=0.08分位, λ=1.0, γ=1.0) において:
\begin{itemize}
  \item 受容率: 96.2\% (目標95\%, +1.2\%達成)
  \item FMR: 1.8\% (目標2\%, -0.2\%達成)
  \item 追加P50: 185ms (目標200ms, -15ms達成)
\end{itemize}

\textbf{PSZ達成を確認}。パラメータ感度分析（§X）により、
λ∈[0.8,1.2], γ∈[0.8,1.2]の範囲でPSZ維持を確認。
```

**Case B: PSZ未達の場合**
```latex
\paragraph{PSZ到達の限定的達成}

現時点の最良構成（θ_AG=0.90, θ_DG=0.10, λ=1.1, γ=0.9）:
\begin{itemize}
  \item 受容率: 94.1\% (目標95\%, \textcolor{red}{-0.9\%未達})
  \item FMR: 1.8\% (目標2\%, 達成)
  \item 追加P50: 195ms (目標200ms, 達成)
\end{itemize}

\textbf{制約}: 受容率を95\%に引き上げるとFMRが2.3\%に悪化（トレードオフ）。
\textbf{今後の課題}: θの動的調整、multi-objectiveベイズ最適化を検討。
```

---

### 問題4: 30ノード洞察ベクトルの主張が弱い

**現状** (§5.4, Table 5.X):
```
DG発火で選別した30ノード級サブグラフから導出したベクトルが
LLM応答方向と整合（Δs=+0.23）
```

**問題**:
- **Δs=+0.23の統計的有意性は？** → p値、信頼区間なし
- **再現性は？** → 何クエリで測定？分散は？
- **メカニズムは？** → なぜ30ノードで整合するのか説明なし
- **ベースラインは？** → ランダムサブグラフと比較したか？

**改善提案**:

```latex
\paragraph{洞察ベクトル整合の検証}

\textbf{実験設定}:
\begin{itemize}
  \item 測定対象: 100クエリ（DG発火クエリのみ）
  \item サブグラフ: DG発火時点のh-hop誘導部分グラフ（平均32±8ノード）
  \item ベクトル導出: ノード埋め込みの平均（L2正規化）
  \item 方向整合: コサイン類似度（LLM最終応答ベクトルとの比較）
\end{itemize}

\textbf{結果}:
\begin{table}[H]
\centering
\caption{洞察ベクトルとLLM応答の方向整合}
\begin{tabular}{lcccc}
\toprule
サブグラフ種別 & 平均Δs & 95\%CI & vs Random & p値 \\
\midrule
ランダム選択 & -0.05 & [-0.12, 0.02] & - & - \\
AG発火のみ & +0.08 & [0.01, 0.15] & +0.13 & p=0.08 \\
\textbf{DG発火} & \textbf{+0.23} & [0.15, 0.31] & +0.28 & p<0.001*** \\
\bottomrule
\end{tabular}
\end{table}

\textbf{解釈}:
DG発火サブグラフは統計的有意にLLM応答と整合。
メカニズム仮説: DG発火=短絡検出 → 推論経路の橋渡し。
\textbf{限界}: エンコーダ専用モデルのため言語化不可、因果関係未確定。
```

---

### 問題5: データセット詳細の欠如

**現状**:
- "500クエリ/50ドメイン"とあるが:
  - どの50ドメイン？
  - クエリの具体例は？
  - データセットは公開される？

**改善提案**:

```latex
\subsection{データセット詳細}
\label{sec:rag_dataset}

\paragraph{構成}
\begin{itemize}
  \item \textbf{総クエリ数}: 500（訓練300、検証100、テスト100）
  \item \textbf{ドメイン}: 50カテゴリ（技術10、医学8、法律7、歴史9、科学8、その他8）
  \item \textbf{ソース}: 公開データセット（HotpotQA, 2WikiMultihopQA）+ 自作100
  \item \textbf{難易度分布}:
    \begin{itemize}
      \item 単一hop: 150クエリ（30\%）
      \item 2-hop: 200クエリ（40\%）
      \item 3-hop以上: 150クエリ（30\%）
    \end{itemize}
\end{itemize}

\paragraph{クエリ例}
\begin{itemize}
  \item \textbf{単純}: "Transformerの発表年は？" → 直接検索
  \item \textbf{2-hop}: "Attentionを提案した論文の筆頭著者の所属機関は？"
  \item \textbf{3-hop}: "2017年に発表された注意機構を用いた論文のうち、
                        最も引用されている論文の著者が次に発表した論文のテーマは？"
\end{itemize}

\paragraph{公開計画}
検証用100クエリをGitHubで公開予定（テスト100は評価汚染防止のため非公開）
```

---

### 問題6: Equal-resources条件の不徹底

**現状** (§7.3.2, l.2358-2367):
- Equal-resources原則は述べられている
- しかし**実際に守られているか検証可能な記載がない**

**問題**:
- LLMバジェット: 入力8k/出力512と書いているが、実測値は？
- ANN設定: "共有のHNSWパラメータ"とあるが、全手法で同じ埋め込み器？
- 遅延算入: "gating/多hop/部分グラフ計算の時間は追加遅延に算入"とあるが、
            実際の内訳は？

**改善提案**:

```latex
\paragraph{Equal-resources検証表}

\begin{table}[H]
\centering
\caption{Equal-resources条件の遵守確認（500クエリ平均）}
\begin{tabular}{lccccc}
\toprule
手法 & 埋め込み & ANN呼出 & LLM入力 & LLM出力 & 総遅延 \\
     & 呼出数 & (ef=200) & トークン & トークン & (ms) \\
\midrule
Flat RAG & 1 & 1 & 3,245 & 487 & 1,120 \\
GraphRAG & 1 & 3 & 3,312 & 502 & 1,165 \\
DyG-RAG & 1 & 2 & 3,278 & 495 & 1,178 \\
geDIG & 1 & 2.1 & 3,301 & 498 & 1,185 \\
      &   &     &       &     & (+gating 58ms) \\
\midrule
制約 & 1 & ≤3 & ≤8,000 & ≤512 & - \\
\bottomrule
\end{tabular}
\end{table}

\noindent すべての手法でSentence-BERT (all-MiniLM-L6-v2) 使用、
HNSW (M=32, ef=200) 共通設定。
```

---

### RAG実験の総合評価

| 観点 | 評価 | コメント |
|------|------|----------|
| **PSZ枠組** | ✅ 優 | 三軸評価（品質/安全/遅延）は実務的 |
| **クエリ層別** | ✅ 良 | タイプ別分析は適切 |
| **30ノード洞察** | △ 予備的 | 興味深いが統計的検証不足 |
| **実験結果** | ❌ **致命的** | 数値が全て未記入（`\experimNote`） |
| **比較対象** | ❌ **致命的** | GraphRAG等との直接比較なし |
| **PSZ達成証拠** | ❌ **致命的** | 達成の具体的数値なし |
| **データセット** | ❌ 不足 | 詳細不明、公開計画不明 |
| **Equal-resources** | ⚠️ 要検証 | 原則は良いが実測値の記載なし |

---

## 理論検証の欠如

### 問題1: FEP-MDLブリッジの実験的検証がない

**理論的主張** (§2.3, l.159-243):
```
命題: 仮定(B1)-(B4)の下で
F = ΔEPC_norm - λΔIG_norm ∝ ΔMDL + O(1/N)
```

**問題**:
- **実験による検証がない**: 実際にF ∝ ΔMDLが成立しているか測定していない
- **残差O(1/N)の評価なし**: Nに対する誤差の振る舞いを確認していない

**改善提案**:

```latex
\subsection{FEP-MDL対応の実験的検証}
\label{sec:fep_mdl_validation}

\paragraph{検証実験}
迷路実験の各ステップでFとΔMDLを同時測定:
\begin{itemize}
  \item F: 式\ref{eq:F_job_evidence_intro}で計算
  \item ΔMDL: グラフの最小記述長（エッジリスト圧縮 + ノード特徴）
        $$\text{MDL}(G) = L_{\text{struct}}(G) + L_{\text{feat}}(G|V)$$
\end{itemize}

\paragraph{結果}
\begin{figure}[H]
\centering
\includegraphics[width=0.7\linewidth]{f_vs_mdl_scatter.pdf}
\caption{FとΔMDLの散布図（15×15迷路、N=100エピソード、1000ステップ）。
         相関係数 r=0.87 (p<0.001)、線形回帰 ΔMDL = 1.23F + 0.05。}
\end{figure}

\paragraph{N依存性}
\begin{table}[H]
\centering
\caption{ノード数Nに対する残差|F - ΔMDL|の振る舞い}
\begin{tabular}{lccc}
\toprule
N（ノード数） & 残差平均 & 残差標準偏差 & O(1/N)フィット \\
\midrule
10 & 0.085 & 0.023 & 0.092 \\
30 & 0.031 & 0.012 & 0.031 \\
100 & 0.010 & 0.005 & 0.009 \\
300 & 0.003 & 0.002 & 0.003 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{結論}: F ∝ ΔMDLの比例関係を実験的に確認。残差はO(1/N)で減衰。
```

---

### 問題2: 埋め込み空間要件(A1)-(A3)の検証なし

**理論的要求** (§3.6, l.295-310):
```
(A1) 意味勾配保存
(A2) 局所滑らかさ
(A3) スケール正規化
```

**問題**:
- **迷路では自明**: 8次元手動設計だから当然満たす
- **RAGでは未検証**: Sentence-BERTが本当に(A1)-(A3)を満たすか確認していない

**改善提案**:

```latex
\subsection{埋め込み空間要件の実験的検証}
\label{sec:embedding_validation}

\paragraph{検証対象}
RAG実験で使用したSentence-BERT (all-MiniLM-L6-v2) が
要件(A1)-(A3)を満たすか検証。

\paragraph{(A1) 意味勾配保存の検証}
\begin{itemize}
  \item \textbf{手法}: 意味的類似度（人間評価）vs ベクトル距離（L2）の相関
  \item \textbf{データ}: STS-Benchmark（5,749文対、類似度スコア0-5）
  \item \textbf{結果}: Spearman ρ = 0.78（ベースライン0.75、要件≥0.7達成）
\end{itemize}

\paragraph{(A2) 局所滑らかさの検証}
\begin{itemize}
  \item \textbf{手法}: 近傍ベクトルの距離変化率測定
        $$\text{smoothness} = \mathbb{E}_{v,v':\|v-v'\|<\epsilon}
        \frac{\|f(v)-f(v')\|}{\|v-v'\|}$$
  \item \textbf{結果}: smoothness = 1.08（理想=1.0、要件<1.5達成）
\end{itemize}

\paragraph{(A3) スケール正規化の検証}
\begin{itemize}
  \item \textbf{手法}: 全ベクトルのL2ノルム分布
  \item \textbf{結果}: 平均0.998±0.015（要件: 1.0±0.05達成）
\end{itemize}

\textbf{結論}: Sentence-BERTは要件(A1)-(A3)を満たす。
```

---

## 再現性の問題

### 問題1: コードが公開されていない

**現状** (Abstract, l.69):
```
Code: https://github.com/miyauchikazuyoshi/InsightSpike-AI
```

**しかし**:
- 実験コードは「順次公開予定」（§謝辞後, l.2502）
- **実験済みならコードは既にあるはず**
- 公開しない理由は？

**要求**:
- 全実験の再現コード（迷路、RAG、30ノード洞察）
- 設定ファイル（ハイパーパラメータ）
- データ生成スクリプト（迷路）
- 評価スクリプト（統計検定含む）

**推奨ディレクトリ構造**:
```
experiments/
├── maze-navigation/
│   ├── run_experiment.py
│   ├── configs/
│   │   ├── maze_15x15.yaml
│   │   ├── maze_25x25.yaml
│   │   └── maze_50x50.yaml
│   ├── baselines/
│   │   ├── greedy_novelty.py
│   │   ├── ucb1.py
│   │   └── partial_obs_astar.py
│   └── reproduce.sh
├── rag-experiment/
│   ├── run_rag.py
│   ├── configs/
│   │   └── rag_500q_50d.yaml
│   ├── data/
│   │   └── queries_validation.jsonl (100クエリ)
│   └── reproduce.sh
└── README.md (再現手順)
```

---

### 問題2: ハイパーパラメータの記載不足

**迷路** (Table 4.2): 比較的詳細 ✓

**RAG**: 不足
- 埋め込み器の詳細は？ (Sentence-BERTのどのモデル？)
- ANNパラメータは？ (HNSW: M, efConstruction, ef?)
- LLMは？ (GPT-4? Claude? 温度設定は？)
- k_link, H, λ, γの値は？

**改善提案**:

```latex
\begin{table}[H]
\centering
\caption{RAG実験のハイパーパラメータ}
\label{tab:rag_hyperparameters}
\begin{tabular}{ll}
\toprule
パラメータ & 設定 \\
\midrule
埋め込み器 & Sentence-BERT (all-MiniLM-L6-v2, 384次元) \\
ANN & HNSW (M=32, efConstruction=200, ef=200) \\
LLM & GPT-4-turbo (temperature=0.2, max\_tokens=512) \\
k\_link & 50（候補エッジ数） \\
H & 3（最大hop数） \\
λ & 1.0（構造-情報トレードオフ） \\
γ & 1.0（IG内のΔH-ΔSP配分） \\
θ\_AG & 0.92分位（burn-in 50クエリで較正） \\
θ\_DG & 0.08分位 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 優先度別アクション項目

### 🔴 優先度1: 致命的（投稿前に必須）

#### 迷路実験
- [ ] 成功基準を3層（必要/十分/診断）に再設計・明記
- [ ] Greedy-Novelty, UCB1ベースラインを実装・比較
- [ ] 統計検定にCohen's d、95% CI追加
- [ ] 閾値較正プロトコル（訓練/検証分離）を明記

#### RAG実験
- [ ] **全`\experimNote`を実数値に置き換え**（38箇所）
- [ ] **PSZ達成の具体的数値を記載**（受容率、FMR、P50）
- [ ] **GraphRAG, DyG-RAGとの直接比較**（同一データセット）
- [ ] データセット詳細を記載（50ドメインの具体例、クエリ例）

#### 再現性
- [ ] **実験コードを完全公開**（迷路、RAG、30ノード洞察）
- [ ] RAGハイパーパラメータ表を追加（埋め込み器、ANN、LLM詳細）

---

### 🟡 優先度2: 重要（査読対策に強く推奨）

#### 迷路実験
- [ ] 部分観測A*を実装（Dijkstra「上限参照」の言い訳を強化）
- [ ] エピソード表現の標準分解との対応を明記
- [ ] 座標ノイズ実験（表現頑健性の軽量検証）

#### RAG実験
- [ ] 30ノード洞察の統計的検証（p値、信頼区間、ベースライン比較）
- [ ] Equal-resources検証表（埋め込み呼出数、トークン数、遅延内訳）
- [ ] クエリタイプ別の詳細分析（single/2-hop/3-hopでのAG/DG発火率）

#### 理論検証
- [ ] **FEP-MDL対応の実験的検証**（F vs ΔMDL散布図、N依存性）
- [ ] **埋め込み空間要件(A1)-(A3)の検証**（STS-B相関、smoothness測定）

---

### 🟢 優先度3: あれば良い（説得力強化）

#### 迷路実験
- [ ] 情報獲得MCTSを実装（現代的探索の代表）
- [ ] 汎用埋め込み（Sentence-BERT）での迷路実験

#### RAG実験
- [ ] 検証用100クエリのGitHub公開
- [ ] クラウドソーシングによる人間評価（受容率の妥当性検証）

---

## 査読者の予想される指摘

### 迷路実験
1. **"Random Walkに勝っただけでは？"** → Greedy-Novelty, UCB1追加で対策
2. **"8次元手作り特徴量で勝っただけでは？"** → 座標ノイズ実験で対策
3. **"Dijkstraとなぜ比較しない？"** → 部分観測A*追加で対策
4. **"統計的有意性は？"** → Cohen's d, 95% CI追加で対策

### RAG実験
1. **"数値がないので評価不能"** → 全`\experimNote`埋めで対策
2. **"GraphRAGと実際に比較したの？"** → 直接比較実験必須
3. **"PSZ達成の証拠は？"** → 具体的数値記載必須
4. **"30ノード洞察はたまたまでは？"** → 統計検定、ベースライン比較で対策

### 理論
1. **"FEP-MDL対応は本当？"** → 実験的検証必須
2. **"埋め込み空間要件は満たされている？"** → 検証実験必須
3. **"仮定(B1)-(B4)が強すぎる"** → 今後の課題として明記（短期対策困難）

---

## 総合評価と推奨

### 現状の問題（再掲）
| カテゴリ | 評価 | 対策優先度 |
|---------|------|-----------|
| 迷路実験デザイン | ⚠️ 要改善 | 🟡 中 |
| RAG実験デザイン | ✅ 良（枠組） | - |
| **迷路実験結果** | ⚠️ 不十分 | 🟡 中 |
| **RAG実験結果** | ❌ **未記載** | 🔴 **最高** |
| **理論検証** | ❌ **不在** | 🟡 中 |
| **再現性** | ❌ **不足** | 🔴 **最高** |

### 推奨される行動計画

#### Step 1: 数値埋め（1-2週間）
- RAG実験を完遂し、全`\experimNote`を実数値に
- PSZ達成の具体的証拠を記載
- 迷路実験の信頼区間、Cohen's d計算

#### Step 2: ベースライン強化（2-3週間）
- Greedy-Novelty, UCB1実装（迷路）
- GraphRAG, DyG-RAG実装または公開データ再現（RAG）
- 統計検定の充実

#### Step 3: 理論検証（1-2週間）
- FEP-MDL対応の測定実験
- 埋め込み空間要件の検証（STS-B等）

#### Step 4: 再現性パッケージ（1週間）
- コード整理・公開
- READMEとドキュメント整備
- データセット公開（検証用100クエリ）

**推定総工数**: 5-8週間

---

## 結論

**実験デザイン自体は論理的で、PoCとして成立可能**。しかし:

1. **迷路実験**: 原理検証としては良いが、ベースライン弱い、統計検定不足
2. **RAG実験**: 枠組（PSZ）は良いが、**数値が全て未記入で評価不能**
3. **理論検証**: FEP-MDL対応、埋め込み空間要件の実験的検証が不在

**このまま投稿すると確実にリジェクトされる**。特に:
- RAG実験の未記入（`\experimNote`38箇所）は致命的
- GraphRAG等との直接比較なしは説得力不足
- 再現性の欠如は現代の査読基準で受け入れられない

**推奨**:
1. まず🔴優先度1（RAG数値埋め、コード公開）を完遂
2. 次に🟡優先度2（ベースライン強化、理論検証）に着手
3. 全体が揃ってから投稿を検討

**ポジティブな見方**:
- 理論の枠組みは興味深い
- 実験デザインの方向性は適切
- PSZ達成できれば十分な貢献になる
- 必要なのは「実験の完遂」と「結果の記載」

---

**次のステップ**: このレビューを元に、優先度1のタスクリストを作成し、実験完遂のタイムラインを立てることを推奨します。
