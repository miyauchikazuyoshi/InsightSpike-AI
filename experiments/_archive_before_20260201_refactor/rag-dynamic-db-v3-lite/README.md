# RAG Dynamic DB v3-lite

このディレクトリは論文「geDIG: One-Gauge制御によるオンライン知識運用と洞察検出」で提示した RAG 評価系の軽量リプロダクション環境です。`rag-dynamic-db-v3` で検証してきたフローを簡潔化しつつ、最新版で議論している以下の要素をそのまま確認できるように再構成しました。

- Sentence-BERT など (A1)--(A3) 要件を満たす埋め込み空間を前提とした retrieval + memory 更新
- geDIG Core (`insightspike.algorithms.gedig_core.GeDIGCore`) による ΔGED / ΔIG 計算と多ホップ評価
- 二段ゲート (AG/DG) に基づくイベント駆動制御とログ出力
- Static RAG / Frequency / Cosine 類似度といったベースラインとの比較
- PER / 受容率 / FMR / レイテンシ (PSZ 指標) の測定スクリプト

## ディレクトリ構成

```
configs/
    experiment_geDIG_vs_baselines.yaml   実験設定 (データ・閾値・ベースライン一覧)
data/
    sample_queries.jsonl                 小規模サンプルデータ (本番は独自データに差替え)
src/
    config_loader.py                     YAML から設定を読み込むユーティリティ
    dataset.py                           JSONL データのロードと検証
    embedder.py                          Sentence-BERT もしくはランダム初期化の埋め込み生成
    retriever.py                         BM25 + 埋め込み類似度のハイブリッド取得 (A1--A3 対応)
    graph_memory.py                      動的グラフメモリと geDIG 用エクスポート
    gedig_scoring.py                     GeDIGCore をラップし AG/DG 判定を付与
    strategies.py                        geDIG / Static / Frequency / Cosine の戦略定義
    metrics.py                           PER / 受容率 / FMR / レイテンシ / C-value の集計
    pipeline.py                          実験オーケストレーション
    run_experiment.py                    エントリポイント (CLI)
results/
    *.json                               実行結果 (timestamp 付きで保存)
```

## クイックスタート

```bash
cd experiments/rag-dynamic-db-v3-lite
python -m src.run_experiment --config configs/experiment_geDIG_vs_baselines.yaml
```

実行すると `results/` 配下に `rag_v3_lite_<timestamp>.json` が生成され、各戦略ごとのメトリクス (PER, acceptance, FMR, latency, PSZ 内の有無など) と geDIG 閾値ログ、AG/DG の発火統計が記録されます。

## 前提

- Python 3.10+
- `pip install -r ../../requirements.txt` (既存環境と共通)
- Sentence-BERT を利用する場合は `sentence-transformers` を追加インストールしてください。インストールされていない場合は isotropic なランダムベクトルで代用します。

## 実験を拡張するには

1. `data/` に独自の JSONL データセットを配置し、`configs/*.yaml` の `dataset.path` を変更します。
2. 埋め込み (`embedder`) を任意のモデルに差し替えたい場合は `src/embedder.py` を拡張します。A1--A3 の性質 (意味勾配・L2正規化・局所滑らかさ) を満たすよう注意してください。
3. 新しいベースラインを追加する際は `src/strategies.py` に Strategy クラスを追加し、config の `baselines` に追記します。
4. Phase 2 (オフライン最適化) を評価する場合は `pipeline.py` の `run_phase_two()` を拡張し、batch 再配線ロジックを実装してください。

### 合成データセットの生成

軽量な比較用に、テンプレートベースの合成データセットを生成するスクリプトを用意しています。

```bash
python scripts/generate_dataset.py \
  --num-queries 500 \
  --output data/sample_queries_500.jsonl

python scripts/generate_dataset.py \
  --num-queries 1000 \
  --output data/sample_queries_1000.jsonl
```

生成後に `configs/experiment_geDIG_vs_baselines.yaml` の `dataset.path` を新しいファイルに差し替えてください。必要に応じてドメイン定義 (`scripts/generate_dataset.py`) を拡張することで、より多様な比較データを構成できます。

## 既知の制約

- サンプルデータセットは玩具規模です。論文で使っている 500 アイテム / 50 ドメインの設定に合わせるには独自データへの差し替えが必要です。
- 現状のレイテンシ測定はシミュレーション値を返すため、本番計測では外部サービスの計測結果に置き換えてください。
- PSZ 判定は config に記載した閾値で計算しています。論文と同じ 95% / 200ms / FMR 2% を既定値にしていますが、用途に合わせて調整してください。

---

この v3-lite の目的は「論文で述べた geDIG の制御ロジックとベースライン比較を最小労力で再現できる足場」を提供することです。重い依存や大規模データが不要な一方で、`rag-dynamic-db-v3` に含まれる全ての派生スクリプトを再現しているわけではありません。必要に応じて両者を行き来しながら開発を進めてください。
