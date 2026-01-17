# geDIG 普及・検証プラン

> **目標**: geDIGを「知性の設計原理」として学術的・実用的に認知させる

**作成日**: 2026-01-16
**最終更新**: 2026-01-17
**ステータス**: Phase 1（実験完了・拡散継続） / Phase 2 準備中
**著者**: miyauchikazuyoshi

---

## Executive Summary

geDIG（graph edit Distance + Information Gain）は、動的知識グラフの更新タイミングを統一的に制御するフレームワークである。本計画書では、geDIGを段階的に普及させるための3フェーズ戦略を定義する。

**核心となる仮説**:
> 構造コストと情報利得のバランス（F = ΔEPC_norm − λ·ΔIG）は、知的システムに共通する設計原理である

**戦略の要点**:
1. 小さく見せて、大きく気づかせる
2. 主張より先にデモを見せる
3. 予測→検証のサイクルで信頼を構築する

---

## 現状分析

### 資産（Strengths）

| 資産 | 状態 | 備考 |
|------|------|------|
| geDIGコアアルゴリズム | ✅ 実装済み | `src/insightspike/algorithms/gedig_core.py` |
| Maze PoC | ✅ 動作 | 15×15で成功率98%、可視化あり |
| HotPotQAフル実験 | ✅ 完走 | dev 7,405件 + baselines |
| 論文 v6 | ⚠️ ドラフト | 日英両方、図表一部欠損 |
| 理論的枠組み | ✅ 整備 | FEP-MDL対応 |
| Streamlitデモ | ✅ 作成済み | `apps/hotpotqa_demo.py` / `spaces/gedig-demo/` |

### 課題（Weaknesses）

| 課題 | 深刻度 | 対策 | 状態 |
|------|--------|------|------|
| スケーラビリティ不足 | 高 | 51×51で成功率55% → アルゴリズム改善 | 未着手 |
| 強いベースラインとの比較が限定的 | 高 | ColBERT/DPR等を追加 | 🔄 部分対応 (BM25/Contriever/Static GraphRAG) |
| デモの公開・導線が弱い | 中 | Spaces公開 + README/LP導線 | 🔄 進行中 |
| HotPotQA結果の再現性/要約整備 | 中 | 実行手順・差分分析の標準化 | 🔄 進行中 |

---

## Phase 1: 狭いドメインで勝つ

**期間**: 4週間
**目標**: HotPotQAで競争力のある結果を出し、「RAGの更新問題を解くツール」として認知される

### 1.1 HotPotQA実験の完成

**目的**: 標準ベンチマークでの定量的評価

#### タスク一覧

| ID | タスク | 担当 | 期限 | 依存 |
|----|--------|------|------|------|
| 1.1.1 | データパイプライン整備 | - | Week 1 | - |
| 1.1.2 | BM25ベースライン実装 | - | Week 1 | 1.1.1 |
| 1.1.3 | Contrieverベースライン実装 | - | Week 2 | 1.1.1 |
| 1.1.4 | geDIG適応・統合 | - | Week 2 | 1.1.1 |
| 1.1.5 | GraphRAGベースライン（optional） | - | Week 3 | 1.1.1 |
| 1.1.6 | 評価スクリプト統一 | - | Week 1 | - |

#### 評価メトリクス

```yaml
primary:
  - EM (Exact Match)
  - F1 (Answer F1)
  - Supporting Facts F1

secondary:
  - FMR (False Merge Rate)  # geDIG特有
  - Latency (P50, P95)
  - Memory Usage
```

#### 成功基準

- [ ] EM/F1 で BM25+LLM を **5%以上**上回る
- [ ] FMR が閾値法より **50%以上**低い
- [ ] P50 latency が **500ms以下**

#### 現状（2026-01-17）

- dev 7,405件で geDIG は BM25 に対して **EM +0.9pt / F1 +1.5pt**
- static_graphrag が最良（F1 0.5594）。差分分析が次の焦点
- 目標（+5%以上）には未達。改善余地の特定が優先

### 1.2 インタラクティブデモ

**目的**: 「見ればわかる」状態を作る

#### 仕様

```
入力: 自然言語の質問
出力:
  - 回答
  - geDIGの判断過程（accept/reject/exploreの可視化）
  - 比較（geDIG有り vs 無し）
```

#### 技術スタック

| コンポーネント | 技術 | 理由 |
|----------------|------|------|
| Frontend | Streamlit or Gradio | 素早くプロトタイプ可能 |
| Backend | FastAPI | 既存コードとの統合が容易 |
| 可視化 | Plotly / D3.js | グラフ構造の表示 |
| デプロイ | Hugging Face Spaces | 無料、共有しやすい |

#### 現状（2026-01-17）

- ローカルデモ: `apps/hotpotqa_demo.py`
- Spaces用: `spaces/gedig-demo/`（デプロイ先/手順は別途）
- Gradio試作: `spaces/gedig-demo/app_gradio.py`（未デプロイ）

#### 画面構成案

```
┌─────────────────────────────────────────┐
│  geDIG Demo - Dynamic RAG Controller    │
├─────────────────────────────────────────┤
│                                         │
│  [質問を入力]                            │
│  ┌─────────────────────────────────┐    │
│  │ What is the capital of France?  │    │
│  └─────────────────────────────────┘    │
│                            [実行]       │
│                                         │
├───────────────┬─────────────────────────┤
│  geDIG有り    │  geDIG無し（Top-k）    │
├───────────────┼─────────────────────────┤
│  回答: Paris  │  回答: Paris           │
│  検索数: 2    │  検索数: 5             │
│  FMR: 0%      │  FMR: 40%              │
│               │                         │
│  [グラフ可視化] │  [取得文書リスト]       │
│     ○──○      │  - Doc1 (relevant)     │
│     │╲ │      │  - Doc2 (noise)        │
│     ○──○      │  - Doc3 (noise)        │
│               │  - Doc4 (relevant)     │
│  AG: 1回      │  - Doc5 (noise)        │
│  DG: 1回      │                         │
└───────────────┴─────────────────────────┘
```

### 1.3 技術ブログ記事

**目的**: 検索流入、認知獲得

#### 記事構成

```markdown
Title: 「RAGの『いつ更新するか』問題を解く ― geDIG入門」

1. 問題提起（200字）
   - RAGの課題：What（何を取得）は解決、When（いつ更新）は未解決

2. geDIGの直感（300字）
   - 構造コスト vs 情報利得
   - 人間の判断との類似性

3. 動かしてみる（コード付き、500字）
   - pip install
   - 最小サンプル

4. HotPotQAでの結果（300字）
   - 表1つ、グラフ1つ

5. なぜ動くのか（200字）
   - FEP/MDLへの言及（深入りしない）

6. 次のステップ（100字）
   - リンク集
```

#### 公開先

| プラットフォーム | 言語 | 優先度 |
|------------------|------|--------|
| Zenn | 日本語 | 高 |
| Qiita | 日本語 | 高 |
| Medium | 英語 | 中 |
| dev.to | 英語 | 低 |

#### 現状（2026-01-17）

- 日本語/英語の下書き作成済み: `docs/blog/`
- 公開はこれから（Zenn/Qiita/Medium/dev.to）

### Phase 1 成果物チェックリスト

- [x] `experiments/hotpotqa-benchmark/results/` に結果JSON → ✅ dev 7,405件 + baselines
- [x] `apps/hotpotqa_demo.py` にStreamlitアプリ
- [ ] Hugging Face Spacesにデプロイ済みデモ → `spaces/gedig-demo/`（未デプロイ）
- [ ] 技術ブログ記事（日本語）公開 → 下書き完成（`docs/blog/`）
- [ ] README.mdにデモへのリンク追加 → 要確認/未対応

#### 補足（リポジトリ内の実体）

- Streamlitローカル版: `apps/hotpotqa_demo.py`
- Spaces配信用: `spaces/gedig-demo/`
- Gradio試作: `spaces/gedig-demo/app_gradio.py`
- 直近結果: `experiments/hotpotqa-benchmark/results/`

---

## Phase 2: 原理を匂わせる

**期間**: 6週間
**目標**: 「なぜ動くのか」を説明し、他ドメインへの適用可能性を示す

### 2.1 理論の整理

#### 「5分でわかるgeDIG」資料

```
対象: 技術者、研究者（ML/NLP背景）
形式: スライド10枚 or 1ページドキュメント
内容:
  1. 問題（1枚）
  2. 解法（2枚）: F = 構造コスト - λ·情報利得
  3. 直感（2枚）: 人間の判断との対応
  4. 結果（2枚）: Maze + RAG
  5. 理論（2枚）: FEP/MDLとの関係
  6. 次へ（1枚）: リンク、連絡先
```

#### FEP/MDL対応表（1ページ）

| 概念 | FEP | MDL | geDIG |
|------|-----|-----|-------|
| 構造 | 内部モデル | モデル記述長 | ΔEPC_norm |
| 情報 | 予測誤差 | データ記述長 | ΔIG (ΔH + γ·ΔSP) |
| 目標 | 自由エネルギー最小化 | 全記述長最小化 | F最小化 |
| 探索 | Active Inference | - | AG |
| 統合 | Belief Update | 圧縮 | DG |

### 2.2 他ドメインでのPoC

#### 候補ドメイン

| ドメイン | 難易度 | インパクト | 優先度 |
|----------|--------|------------|--------|
| コード補完 | 中 | 高 | ★★★★☆ |
| 対話システム | 中 | 高 | ★★★★☆ |
| Gridworldエージェント | 低 | 中 | ★★★☆☆ |
| 推薦システム | 高 | 高 | ★★☆☆☆ |

#### コード補完PoC仕様

```
問題: 複数のコード補完候補からどれを採用するか

入力:
  - 現在のコード（コンテキスト）
  - 補完候補リスト（from Copilot/CodeLlama等）

geDIG適用:
  - 構造: 既存コードとのAST整合性
  - 情報利得: 補完による不確実性減少

出力:
  - 採用する補完候補
  - 判断理由（F値の内訳）
```

### 2.3 Transformer内部分析

#### 実験設計

```yaml
モデル:
  - GPT-2 Small (124M) # 軽量、解析しやすい
  - BERT-base (110M)   # 既存実験の拡張

データ:
  - WikiText-103 (言語モデリング)
  - SQuAD (QA、推論が必要)

測定:
  - 各層のAttentionをグラフ化
  - F値 = structural_cost(A) - λ·information_gain(A)
  - 層ごとのF値変化を記録

分析:
  - F値と最終性能（perplexity/accuracy）の相関
  - 「良い層」と「悪い層」のF値分布の違い
  - Head pruningとF値の関係
```

#### 介入実験（因果検証）

```
仮説: F値を下げる方向にAttentionを誘導すると性能が上がる

方法:
  1. 通常の学習
  2. F値を補助損失として追加した学習
     Loss = L_original + α·F_attention
  3. 性能比較

期待される結果:
  - α > 0 で性能向上 → 仮説支持
  - 変化なし or 悪化 → 仮説棄却 or 調整必要
```

### Phase 2 成果物チェックリスト

- [x] 「5分でわかるgeDIG」スライド → ✅ `docs/concepts/gedig_in_5_minutes.md` (日英)
- [x] FEP/MDL対応ドキュメント → ✅ `docs/concepts/universal_principle_hypothesis.md` (日英)
- [ ] コード補完PoCの動作デモ
- [ ] Transformer分析レポート
- [ ] 介入実験の結果

---

## Phase 3: 統一原理として提示

**期間**: 8週間+
**目標**: 「知性の設計原理」として学術的に認知される

### 3.1 論文投稿戦略

#### ターゲット学会

| 学会 | 難易度 | 締切（例年） | 適合性 |
|------|--------|--------------|--------|
| JSAI全国大会 | 低 | 2月頃 | 高（国内認知） |
| NeurIPS Workshop | 中 | 9月頃 | 高（理論+実験） |
| ICML Workshop | 中 | 5月頃 | 高 |
| AAAI | 高 | 8月頃 | 中 |
| NeurIPS Main | 高 | 5月頃 | 要強化 |

#### 投稿ロードマップ

```
2026年:
  Q1: JSAI 2026 投稿（Maze + RAG結果）
  Q2: 結果を受けてフィードバック収集
  Q3: NeurIPS Workshop 投稿準備
  Q4: Workshop投稿 or AAAI準備

2027年:
  Q1-Q2: メインカンファレンス挑戦
```

### 3.2 オープンソース戦略

#### パッケージ整備

```bash
# 目標: これだけで動く
pip install insightspike

# 最小使用例
from insightspike import geDIG

controller = geDIG()
decision = controller.evaluate(
    current_graph=G,
    candidate_node=new_node,
)
# decision.accept / decision.explore / decision.reject
```

#### ドキュメント構成

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── concepts.md          # 直感的説明
├── tutorials/
│   ├── 01_maze_demo.ipynb
│   ├── 02_rag_integration.ipynb
│   ├── 03_custom_domain.ipynb
│   ├── 04_transformer_analysis.ipynb
│   └── 05_theory_deep_dive.ipynb
├── api/
│   └── reference.md
└── theory/
    ├── fep_mdl_bridge.md
    └── faq.md
```

#### コミュニティ構築

| プラットフォーム | 用途 | 優先度 |
|------------------|------|--------|
| GitHub Discussions | Q&A、提案 | 高 |
| Discord | リアルタイム議論 | 中 |
| Twitter/X | 告知、認知 | 高 |
| Zenn Publication | 日本語コンテンツ | 中 |

### 3.3 生物学的検証（長期）

#### 共同研究候補

| 分野 | アプローチ | 探索方法 |
|------|------------|----------|
| 神経科学 | FEPコミュニティ | Karl Friston研究室、国内FEP研究者 |
| 植物学 | 成長モデル | L-system研究者、形態形成 |
| 複雑系 | 自己組織化 | Santa Fe Institute |

#### 検証可能な予測

```
予測1: 「適応的な生物システムはF ≈ 0付近で動作する」
  - 測定: 神経活動からF値を推定
  - 比較: 健常 vs 病的状態

予測2: 「F最小化が上手いエージェントは汎化性能が高い」
  - 測定: 強化学習エージェントのF値
  - 比較: 汎化タスクでの性能

予測3: 「人間の『ひらめき』でF値が急落する」
  - 測定: 問題解決中のEEG/fMRI
  - イベント: 解けた瞬間のF値変化
```

### Phase 3 成果物チェックリスト

- [ ] JSAI 2026 論文投稿
- [ ] PyPIパッケージ公開
- [ ] ドキュメントサイト公開
- [ ] GitHub Star 100+
- [ ] Workshop採択 or 投稿

---

## リソース計画

### 必要スキル・リソース

| リソース | 現状 | 必要なアクション |
|----------|------|------------------|
| ML実装スキル | ✅ | - |
| 論文執筆 | ⚠️ | 査読対応の経験を積む |
| Web開発（デモ） | ⚠️ | Streamlit習得 |
| コミュニティ運営 | ❌ | 発信を始める |
| 神経科学知識 | ❌ | 共同研究者を探す |

### 計算リソース

| 実験 | 必要GPU時間 | 対応 |
|------|-------------|------|
| HotPotQA | 〜10時間 | Colab Pro |
| Transformer分析 | 〜50時間 | Colab Pro or クラウド |
| 大規模RAG | 〜100時間 | クラウド（AWS/GCP） |

---

## リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| HotPotQAで負ける | 中 | 高 | エラー分析→改善、ドメイン変更 |
| 理論が間違っている | 低 | 高 | 予測→検証サイクル、謙虚な主張 |
| 競合が先に発表 | 低 | 中 | 速度重視、差別化ポイント明確化 |
| 関心を持たれない | 中 | 中 | デモ重視、ストーリー改善 |

---

## マイルストーン

```
2026年
├── 1月: 計画策定 ✅ / HotPotQA完走 / デモv1作成
├── 2月: HotPotQA実験完成、デモv1
├── 3月: JSAI投稿、ブログ公開
├── 4月: フィードバック収集、改善
├── 5月: 他ドメインPoC開始
├── 6月: Transformer分析
├── 7月: 「5分でわかるgeDIG」公開
├── 8月: Workshop投稿準備
├── 9月: NeurIPS Workshop投稿
├── 10月: PyPIパッケージ公開
├── 11月: 結果待ち、次期計画
└── 12月: 振り返り、2027年計画
```

---

## 直近4週間の詳細タスク

### Week 1 (2026-01-16 〜 01-22)

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 1-2 | HotPotQAデータセット準備 | `data/hotpotqa_*.jsonl` |
| Day 3-4 | BM25ベースライン実装・評価 | `baselines/bm25.py`, 結果JSON |
| Day 5-6 | geDIG適応・初回実験 | `src/hotpotqa_gedig.py` |
| Day 7 | 結果分析・課題整理 | 分析ノート |

### Week 2 (2026-01-23 〜 01-29)

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 1-2 | パラメータチューニング | 最適パラメータ記録 |
| Day 3-4 | Streamlitデモ骨子 | `apps/demo/app.py` |
| Day 5-6 | 比較可視化実装 | デモ画面完成 |
| Day 7 | ブログ記事ドラフト | `draft_blog.md` |

### Week 3 (2026-01-30 〜 02-05)

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 1-2 | HotPotQAの差分分析（勝ち/負け） | エラー分析ノート |
| Day 3-4 | Spacesデプロイ + README導線 | 公開URL・リンク |
| Day 5-6 | 日本語ブログ公開（Zenn/Qiita） | 公開URL |
| Day 7 | 結果の1枚サマリ作成 | `docs/blog/` or `docs/design/` |

### Week 4 (2026-02-06 〜 02-12)

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 1-2 | 強いベースライン追加計画 | 実験計画メモ |
| Day 3-4 | デモ改善（可視化/UX） | v1.1 |
| Day 5-6 | 英語ブログ公開（Medium/dev.to） | 公開URL |
| Day 7 | Phase 2着手（コード補完PoC） | PoC設計メモ |

---

## 付録

### A. 参考文献

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Rissanen, J. (1978). Modeling by shortest data description.
- Edge et al. (2024). GraphRAG: Unlocking LLM discovery on narrative private data.

### B. 関連リポジトリ

- `experiments/hotpotqa-benchmark/` - HotPotQA実験
- `experiments/maze-query-hub-prototype/` - Maze PoC
- `experiments/transformer_gedig/` - Transformer分析
- `apps/` - デモアプリケーション

### C. 連絡先

- Email: miyauchikazuyoshi@gmail.com
- Twitter/X: @kazuyoshim5436
- GitHub: https://github.com/miyauchikazuyoshi/InsightSpike-AI

---

**Document Version**: 1.2
**Last Updated**: 2026-01-17
