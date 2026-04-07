# Neuro-Pruning Specification
**"Cleaning up the Neural Clutter: Structural Head Pruning"**

## 1. コンセプト (Concept)
*   **仮説**: F値が高いHeadは「構造的な推論」を担当しており、モデルの知能に不可欠。一方、F値が低い（カオスな）Headはノイズであり、削除しても無害（あるいは性能向上）である。
*   **手法**: 学習済みモデルの全HeadのF値を計測し、下位 $N$% を物理的に無効化（Pruning）する。

## 2. アルゴリズム (Algorithm)

1.  **診断フェーズ (Diagnosis)**:
    *   キャリブレーションデータ（少量の入力テキスト）を流す。
    *   `Flash-geDIG` を用いて、各層・各Headの平均Fスコアを算出する。
    *   スコア行列 `F_scores[layer, head]` を取得。

2.  **選別フェーズ (Selection)**:
    *   全HeadをFスコア順にソート。
    *   指定された `pruning_ratio` (例: 20%) に相当する、下位Headのリストを特定する。
    *   *Option*: 層ごとの役割（浅い層はカオスでも許容など）を考慮するか？ -> 今は単純なGlobal Rankingでいく。

3.  **剪定フェーズ (Pruning)**:
    *   HuggingFaceの `model.prune_heads()` APIを利用。
    *   指定されたHeadのマスクを適用し、計算グラフから削除する。

4.  **確認フェーズ (Verification)**:
    *   パラメータ数がどれだけ減ったかを確認。
    *   （簡易的に）精度への影響をテスト。

## 3. 実装計画

### ファイル構成
*   `experiments/neuro_pruning/prune_by_structure.py`: メインスクリプト

### CLI引数
```bash
python prune_by_structure.py \
  --model_name "bert-base-uncased" \
  --amount 0.2 \  # 20%削減
  --save_path "pruned_model/"
```

### 必要なもの
*   Flash-geDIG (`src/insightspike/gedig`)
*   HuggingFace Transformers
*   Datasets (glue/sst2 などで評価)

## 4. 期待される効果
*   **モデル軽量化**: 推論速度の向上。
*   **「断捨離」効果**: ノイズ除去による、分布外（OOD）汎化性能の向上（※要検証）。
