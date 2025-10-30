# RAG ベースライン比較計画書

最終更新: 2025-10-14  
作成者: 開発チーム

---

## 1. 目的と範囲

1. geDIG (v3-lite) と既存方式 (GraphRAG, DyG-RAG風, KEDKG) を **equal-resources** 条件で比較する。  
2. 比較対象の指標は以下に統一する。  
   - 受容率 / FMR / PER / 遅延 (P50/P95)  
   - AG/DG に相当する「取得・更新トリガ」指標  
   - g0/gmin に相当する構造ゲージ (実装できる範囲で代替指標を算出)  
3. 評価データは既存の 25 / 168 / 500 クエリセットを使用し、必要に応じて 1000 クエリを追加生成する。  
4. 比較結果は論文第 6 章の表・図へ反映する。

---

## 2. 対象方式と入手計画

| 方式 | 取得手段 | 主要依存 | 備考 |
|------|----------|----------|------|
| GraphRAG (Microsoft Research) | GitHub `microsoft/graphrag` | `langchain`, `pygraphviz`, `faiss`, `pydantic<2` 等 | コミュニティ検出・要約パイプラインを SBERT へ差し替える |
| DyG-RAG風ベースライン | 軽量実装 (PyTorch) | `torch`, `numpy`, `networkx` | クエリの時系列インサートと自己再帰的回答を模擬 |
| KEDKG / Dynamic KG Edit 系 | 公開コード (PyTorch Geometric) | `torch`, `torch-geometric`, `networkx` | 知識編集パイプラインを RAG クエリで駆動 |

- リポジトリは `external/` 以下にサブモジュールとして追加し、Poetry 環境とは分離した venv を併設する。  
- ライセンス確認 (GraphRAG: MIT, DyG-RAG風: 推定 MIT, KEDKG: 要確認)。問題があれば内部実装に限定。

---

## 3. データ・フォーマット整備

1. `data/sample_queries*.jsonl` を以下の形式へ変換するスクリプトを新設する (例: `scripts/convert_for_baselines.py`)。
   - GraphRAG: `documents.tsv` (doc_id, title, content), `questions.jsonl`
   - DyG-RAG風: temporal graph `edges.csv`, `nodes.csv`, クエリ `queries.jsonl`
   - KEDKG: エディットログ `updates.json`, `kg_edges.json`
2. 共通メタ情報 (support/distractor, domain, template) を保持し、後段の比較に再利用する。
3. 生成スクリプトは `experiments/rag-dynamic-db-v3-lite` と分離し `experiments/rag-baselines-data/` に配置する。

---

## 4. 実装計画 (段階別)

### Phase 0: 準備 (0.5〜1.0 日)
- 各リポジトリを `external/` に clone。  
- ライセンス・依存ライブラリ確認、サンプルコードの smoke test。

### Phase 1: GraphRAG 統合 (1.5〜2.0 日)
1. SBERT 版 EmbeddingPipeline を実装し、コミュニティ検出結果を JSON で出力。  
2. 25/168/500 クエリでバッチ実行 → 集計 JSON (per, accept, latency 等) を生成。  
3. geDIG と並べて比較するスクリプト (`scripts/aggregate_baselines.py`) を作成。

### Phase 2: DyG-RAG風ベースライン構築 (2.0〜2.5 日)
1. 既存の時系列グラフサンプルにあわせてデータ整形。  
2. クエリ毎に動的挿入 → 回答 → ログ保存を行うランナーを実装。  
3. geDIG の AG/DG に対応する更新判定を、DyG-RAG風 のメトリクスへマッピング。

### Phase 3: KEDKG 系統 (2.0 日)
1. KG エディット実装を RAG クエリへ適用するアダプタを作成。  
2. エディット結果を geDIG のゲージ指標に変換 (Δエントロピー等)。

### Phase 4: 集計・可視化 (1.0 日)
1. `EVALUATION_GUIDE.md` と論文の Table を更新。  
2. 新しい図 (PSZ scatter, AG/DG ヒストグラム, エディット精度比較) を追加。  
3. 比較分析レポートを `docs/results/baseline_comparison_YYYYMMDD.md` にまとめる。

---

## 5. 評価指標・ログ仕様

- 受容率・FMR: 共通判定ロジックを `scripts/evaluate_results.py` に切り出す。  
- 遅延: 方式ごとのステップ数または推定時間を `P50/P95` で算出。  
- g0/gmin 代替値:
  - GraphRAG: コミュニティ統合スコア (`modularity` 差分) を g0、要約信頼度を gmin に対応させる。  
  - DyG-RAG風: Temporal degree / novelty 指標を組み合わせて g0/gmin を推計。  
  - KEDKG: エディット受理確率やエントロピー減少量を使用。  
- 各方式のログは JSON Lines (`results/baselines/graph_rag_*.jsonl`) 形式に統一する。

---

## 6. リスクと対応策

| リスク | 対応策 |
|--------|--------|
| 依存ライブラリの競合 (例: transformers バージョン) | 方式ごとに独立した venv を作成し、メイン環境へ影響を与えない。 |
| 実行時間が長い (GraphRAG のコミュニティ検出など) | まず 25 クエリで挙動確認 → 168 → 500 と段階的に拡張。必要に応じてコミュニティ数を制限。 |
| 指標の互換性が取れない | 代替指標 (構造差分、エディット確率) を定義し、論文で明記する。 |
| データ形式の変換ミス | 変換後のサンプルを共有し、手動で 1 件ずつ検証。 |

---

## 7. 次のアクション

1. ✅ `external/` ディレクトリを作成し、GraphRAG リポジトリを追加済み。  
2. ✅ 25 クエリ向けのデータ変換スクリプトを先行実装（`scripts/convert_for_baselines.py`）。  
3. ✅ GraphRAG 風ベースライン（コミュニティ構築＋SBERT互換 Embedder）を実装し、最初の比較結果を取得。  
4. ☐ 進捗は本ファイルと `EVALUATION_GUIDE.md` に随時反映する。

---

- [2025-10-14] GraphRAG 風ベースラインと DyG-RAG 風ベースラインの初回実行が完了。指標値を EVALUATION_GUIDE.md に追記。
本計画書は実装コードとは別管理とし、手順や仮定の変更は本ドキュメントを更新することで追跡する。必要に応じて issue/タスク管理にもリンクさせること。
