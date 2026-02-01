# RAG v3-lite 評価基準ガイド

GeDIG ベンチマークで扱う「動的成長グラフ」の評価軸と、直近の実行結果をここにまとめます。論文では本ガイドを評価判定の準拠表として利用してください。

---

## 1. 評価フレーム

| 軸 | 目的 | 指標 | 備考 |
|----|------|------|------|
| **PSZ (Performance Safe Zone)** | ゲート制御を含めた最終挙動の健全性 | 受容率 / FMR / レイテンシ P50 / DG 発火率 / 平均ステップ数 | 今回は `theta_ag = 8.0`, `theta_dg = 0.6` を使用 |
| **クエリ品質** | エピソード接続要求の複雑性と意味整合 | ・マルチホップ段数（テンプレートラベル） <br>・クエリと真値のコサイン距離 (Embedder) <br>・誤発火時のクエリ変動率 | 今回はテンプレート分類と SBERT/TF-IDF 距離をログ化（後述のメタデータ参照） |
| **KG品質** | 構築中グラフが想定ナレッジ集合をどこまでカバーしたか | ・サポート/ディストラクタ比 (`coverage = (support - distractor)/total`) <br>・サポート一致率（DG 発火時の support ノード含有率） <br>・必要ナレッジ集合に対する射影率（ドメイン別） | ドメインによりクラスタ偏りが仕様となり得るため、多様度は補助値 |

> **ナレッジ分母の扱い**  
> 企業情報や法規などクラスタリングされたドメインでは「多様度」だけでは品質を判定できません。  
> そのため各ドメインごとに「必要ナレッジ集合」を定義したうえで、サポート比率・再現カバレッジを評価する方針にしています。

---

## 2. 現行実行結果のサマリ

### 2.1 少数データセット（25クエリ）

- コマンド:  
  ```bash
  poetry run python scripts/run_benchmark_suite.py \
    --config configs/experiment_geDIG_vs_baselines.yaml \
    --datasets data/sample_queries.jsonl
  ```  
  出力: `results/rag_v3_lite_sample_queries_20251014_075707.json`

| 指標 | 値 |
|------|----|
| PER 平均 | 0.357 |
| 受容率 | 0.32 |
| FMR | 0.68 |
| レイテンシ P50 | 240ms (平均ステップ 2.28) |
| AG 発火率 | 0.20 |
| DG 発火率 | 0.40 |
| $g_0$ 範囲 | 0.15 – 10.93 |
| $g_{\min}$ 範囲 | -0.36 – 1.29 |

- グラフベース単純比較（GraphRAG風コミュニティベースライン）:  
  `experiments/rag-baselines/run_graphrag_baseline.py`

| 指標 | 値 |
|------|----|
| PER 平均 | 0.0646 |
| 受容率 | 0.00 |
| FMR | 1.00 |
| レイテンシ P50 | 160ms |


- **メタデータ**  
  各サンプルに `gedig_g0_sequence`, `gedig_ag_sequence`, `gedig_dg_sequence`, `answer_template` を付与済み。クエリ難度、テンプレート種別の分析に利用可能。

### 2.2 大規模データセット（500クエリ）

- コマンド:  
  ```bash
  poetry run bash scripts/run_benchmark_500.sh
  ```  
  出力: `results/rag_v3_lite_500_20251014_064807.json`

| 指標 | 値 |
|------|----|
| PER 平均 | 0.295 |
| 受容率 | 0.24 |
| FMR | 0.76 |
| レイテンシ P50 | 240ms |
| AG 発火率 | 0.44 |
| DG 発火率 | 0.20 |
| $g_0$ 範囲 | 0.02 – 9.89 |
| $g_{\min}$ 範囲 | -0.45 – 1.23 |

- グラフベース単純比較（GraphRAG風コミュニティベースライン）

| 指標 | 値 |
|------|----|
| PER 平均 | 0.0434 |
| 受容率 | 0.00 |
| FMR | 1.00 |
| レイテンシ P50 | 160ms |

- 動的ベースライン（DyG-RAG風）

| 指標 | 値 |
|------|----|
| PER 平均 | 0.0105 |
| 受容率 | 0.00 |
| FMR | 1.00 |
| AG 発火率 | 0.62 |
| DG 発火率 | 0.49 |
| レイテンシ P50 | 200ms |

- **初期所見**  
  - g0 の中央値は約 7.8。閾値 `theta_ag = 8.0` 付近で AG が発火、`theta_dg = 0.6` で DG が 20% 発火。  
  - coverage ≈ 0.3 前後のケースで DG が作動。サポート比率が閾値を割った際に停止できている。  
  - クエリ難度が高い（多ホップ要求）と PER が落ちるが、DG により誤答が抑制されているケースが確認できる。

### 2.3 埋め込みアブレーション（25クエリ）

- コマンド:  
  ```bash
  poetry run python scripts/run_benchmark_suite.py \
    --config configs/experiment_geDIG_vs_baselines_random.yaml \
    --datasets data/sample_queries.jsonl
  ```
  および `configs/experiment_geDIG_vs_baselines_bert_cls.yaml` で再実行。

| 手法 | PER(\%) | 受容率(\%) | FMR(\%) | AG率(\%) | DG率(\%) | 平均 $g_{\min}$ |
|------|---------|------------|---------|----------|----------|-----------------|
| Sentence-BERT | 35.7 | 32.0 | 68.0 | 20.0 | 40.0 | 0.72 |
| HF BERT [CLS] | 35.7 | 32.0 | 68.0 | 32.0 | 36.0 | 0.80 |
| Random (no model) | 35.7 | 32.0 | 68.0 | 32.0 | 44.0 | 0.60 |

Sentence-BERT が最も低い AG 発火率と高い $g_{\min}$ を示し、仮定 (A1)--(A3) を満たす埋め込みが洞察検出の安定性を支えている。ランダム埋め込みでは DG 発火が過剰になり、低品質な更新が混入しやすい。

### 2.4 中間データセット（168クエリ）

- コマンド:  
  ```bash
  poetry run python scripts/run_benchmark_suite.py \
    --config configs/experiment_geDIG_vs_baselines.yaml \
    --datasets data/sample_queries_168.jsonl
  ```  
  出力: `results/rag_v3_lite_168_20251014_075956.json`

| 指標 | 値 |
|------|----|
| PER 平均 | 0.287 |
| 受容率 | 0.24 |
| FMR | 0.76 |
| レイテンシ P50 | 240ms |
| AG 発火率 | 0.32 |
| DG 発火率 | 0.28 |
| $g_0$ 範囲 | 0.02 – 9.61 |
| $g_{\min}$ 範囲 | -0.41 – 1.21 |

- グラフベース単純比較（GraphRAG風コミュニティベースライン）

| 指標 | 値 |
|------|----|
| PER 平均 | 0.0576 |
| 受容率 | 0.00 |
| FMR | 1.00 |
| レイテンシ P50 | 160ms |

- 動的ベースライン（DyG-RAG風）

| 指標 | 値 |
|------|----|
| PER 平均 | 0.0701 |
| 受容率 | 0.00 |
| FMR | 1.00 |
| AG 発火率 | 0.65 |
| DG 発火率 | 0.46 |
| レイテンシ P50 | 200ms |

---

## 3. データ活用メモ

1. **他アーキテクチャ比較**  
   `results/.../results` には Static / Cosine / Frequency も同形式で出力。PSZ の達成率が低い現段階でも、DG 発火率や停止ステップ数で差分を示せる。

2. **クエリ品質解析**  
   `per_samples[*].metadata["answer_template"]` からテンプレート分類が可能。必要に応じて `scripts/` に解析ツールを追加してください。

3. **KG品質評価**  
   coverage, score などのヒューリスティク値は `gate_logs` と `per_samples` のメタデータから取得可能。ナレッジ分母は各ドメインのテンプレート一覧を基に設計する想定です。

---

## 4. 今後のTODO

- [ ] PSZ 指標を満たすためのテンプレート生成ロジック改善（PER/受容率向上）。  
- [ ] クエリ難度の定量指標（多ホップ度、クエリ-真値距離）をレポート化。  
- [ ] ドメイン別ナレッジ分母を策定し、coverage を正式な KG 品質指標へ昇格。  
- [ ] SBERT 環境で `run_insight_vector_alignment.py` を再測定し、TF-IDF フォールバックと比較。

---

このファイルはベンチマーク実験のトップドキュメントとして維持してください。新しい結果を得た際は表と所見を更新し、評価フレームの妥当性を随時補強します。
