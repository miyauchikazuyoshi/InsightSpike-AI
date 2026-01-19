# 論文ストーリー: GED編集操作 = 閃きの実体

## 1. 現在の論文 (v6) の位置づけ

```
geDIG v6: 「いつ情報を受け入れるか」の制御
├── F = ΔEPC - λΔIG (単一ゲージ)
├── AG/DG 2段階ゲート
├── Maze + RAG 実験
└── FEP-MDL 対応
```

## 2. 新しい洞察の位置づけ

```
拡張: 「閃き」の計算的モデル
├── GED編集操作 = 閃きの実体
├── 同型発見 = アインシュタイン的閃き
├── 構造類似度 = アナロジー検出
└── 分子設計AIとの対応
```

---

## 3. ストーリーライン案

### A案: 段階的拡張ストーリー

```
レベル0: 情報の取捨選択
  「どの情報を受け入れるか」
  → 既存RAGの問題

レベル1: 構造的評価
  「構造変化のコストと利得」
  → geDIG F = ΔEPC - λΔIG

レベル2: アナロジー検出
  「異なるドメイン間の構造類似」
  → 太陽系 ≈ 原子

レベル3: 同型発見
  「矛盾を解消する変換の発見」
  → アインシュタインの相対論

レベル4: 理論生成
  「AIが新しい理論を発見する」
  → パラダイムシフト
```

### B案: 対比ストーリー

```
既存AI              vs        geDIG + 同型発見
────────────────────────────────────────────────
パターンマッチ                 構造理解
統計的相関                     因果的理解
知識検索                       知識創造
「似ている」の発見             「同一である」の発見
```

### C案: 科学史ストーリー

```
1. ボーアの原子モデル (1913)
   太陽系 ≈ 原子 → アナロジー

2. アインシュタインの相対論 (1905)
   電磁気学 ⊕ 古典力学 → 時空構造
   「矛盾を解消する変換」= ローレンツ変換

3. geDIGの提案 (2026)
   「2つの構造を同型にする最小編集」
   = 閃きの計算的モデル
```

---

## 4. 論文への入れ方

### Option 1: 既存論文 (v6) の拡張セクション

**場所**: Section 7 (Discussion) の後に新セクション追加

```latex
\section{Toward Computational Theory of Insight}

\subsection{GED as the Essence of Insight}
The edit operations that minimize GED between two structures
can be interpreted as the "content" of an insight...

\subsection{Levels of Structural Discovery}
\begin{itemize}
  \item Level 1: Pattern matching (similarity)
  \item Level 2: Analogy (structural correspondence)
  \item Level 3: Isomorphism discovery (unification via transformation)
\end{itemize}

\subsection{Connection to Molecular Design AI}
Our framework shares the same mathematical foundation
as molecular graph edit distance in drug discovery...
```

**メリット**: 既存の実験結果と繋がる
**デメリット**: 論文が長くなる

### Option 2: 新しい論文として独立

**タイトル案**:
- "Graph Edit Distance as a Computational Model of Insight"
- "From Analogy to Isomorphism: A Unified Theory of Creative Discovery"
- "Structural Unification: How AI Can Discover New Theories"

**構成**:
```
1. Introduction
   - 閃きの計算的モデルの必要性

2. Background
   - GED in molecular design
   - Analogical reasoning in AI
   - geDIG framework (brief)

3. Theory
   - GED edit operations as insight
   - Levels of structural discovery
   - Isomorphism discovery algorithm

4. Experiments
   - Cross-Domain Analogy QA (F1 +60%)
   - Science History Simulation (3/3 success)
   - (追加) 同型発見実験

5. Discussion
   - Connection to FEP/MDL
   - Relation to molecular design AI
   - Implications for AGI

6. Conclusion
   - AI that discovers theories
```

**メリット**: インパクトが明確、独立した貢献
**デメリット**: 新しい実験が必要かも

### Option 3: Future Work + Position Paper

**既存論文のFuture Work**:
```latex
\paragraph{Toward Isomorphism Discovery}
While this paper focuses on "when to accept" decisions,
the GED framework naturally extends to discovering
transformations that unify contradictory structures.
Just as Einstein unified electromagnetism and mechanics
through Lorentz transformation, an AI system could
discover such unifying transformations by minimizing
the edit distance between conflicting knowledge structures.
This direction connects to molecular design AI,
where scaffold hopping finds functionally equivalent
but structurally different molecules.
```

**Position Paper (別途)**:
- ワークショップや短い論文として発表
- 実験より理論的主張を中心に

---

## 5. Related Workへの追加

```latex
\subsection{Molecular Design and Graph Edit Distance}
Graph-based drug discovery uses molecular GED to find
"scaffold hops" - molecules with different structures
but similar functions \cite{...}.
Our framework applies the same principle to knowledge:
finding conceptually equivalent but structurally different
representations.

\subsection{Analogical Reasoning}
Structure Mapping Theory (Gentner, 1983) proposes that
analogy involves mapping relational structures.
geDIG operationalizes this as graph structural similarity,
enabling computational detection of cross-domain analogies.

\subsection{Topological Data Analysis}
Persistent homology captures topological features invariant
under continuous deformation. Our structural similarity
detection can be viewed as a discrete analog for knowledge graphs.
```

---

## 6. 推奨案

**短期 (JSAI 2026向け)**:
- Option 3: 既存論文のFuture Workに入れる
- 実験結果 (Cross-Domain QA, Science History) は Appendix に

**中期 (6ヶ月以内)**:
- Option 2: 独立した論文として執筆
- 同型発見アルゴリズムを実装・実験

**長期 (1年以内)**:
- Phase 7-8 を実装
- 「AIが理論を発見する」デモを作成

---

## 7. キーメッセージ

### 一文で言うと:

> **「創薬AIが分子の同型を探すように、geDIGは理論の同型を探す。
>    その編集操作こそが閃きの実体である。」**

### 三段論法:

```
1. 分子設計AIは分子グラフの編集で新薬を発見する
2. 知識もグラフ構造である
3. よって、知識グラフの編集で新理論を発見できる
```

### インパクト:

```
現在のAI: 「既存知識の検索と組み合わせ」
提案AI:   「新しい知識構造の発見」

これは検索エンジン → 科学者 への質的飛躍
```

---

## 8. 図のアイデア

### 図1: 閃きのレベル

```
     Level 3: 同型発見
        ┌─────────────────┐
        │ T(A) ≡ T(B)     │ ← 変換Tを発見
        │ (Einstein)      │
        └────────┬────────┘
                 │
     Level 2: アナロジー
        ┌────────┴────────┐
        │ A ≈ B            │ ← 構造が似ている
        │ (Bohr)           │
        └────────┬────────┘
                 │
     Level 1: パターンマッチ
        ┌────────┴────────┐
        │ similarity(a,b) │ ← 要素が似ている
        └─────────────────┘
```

### 図2: 分子設計との対応

```
Molecular Design              Knowledge Discovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ┌─────┐                      ┌─────────┐
  │ 分子A │                      │ 理論A   │
  └──┬──┘                      └────┬────┘
     │ GED                          │ GED
     │ (原子追加/削除)              │ (概念追加/削除)
     ▼                              ▼
  ┌─────┐                      ┌─────────┐
  │ 分子B │                      │ 理論B   │
  └─────┘                      └─────────┘
  同じ薬効                       同じ説明力
  (Scaffold Hopping)           (Theory Unification)
```

---

## 9. 次のアクション

1. [ ] JSAI 2026 ドラフトに Future Work として追加
2. [ ] 図1, 図2 を作成
3. [ ] Related Work のドラフト作成
4. [ ] 独立論文のアウトライン作成
5. [ ] Phase 6 (埋め込み統一) の実装開始
