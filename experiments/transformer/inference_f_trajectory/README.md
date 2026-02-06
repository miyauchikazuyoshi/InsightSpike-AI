# 推論時F軌跡実験

**ステータス**: 🔬 初回実験完了・仮説要修正

## 概要

Transformerの推論時（1回のforward pass）において、**層を通過するごとにF値がどう変化するか**を検証する実験。

## 核心的仮説（当初）

```
Layer 0 (入力埋め込み):  F_0 = 高い（無構造・曖昧）
Layer 1:                 F_1 < F_0
  ...
Layer L (出力):          F_L = 最小（構造解決済み）
```

## 結果サマリー

**仮説は棄却された**。代わりに**U字型パターン**を発見：

```
F軌跡:

   高│  *                          *
     │   *                        *
   F │    *                      *
     │     * * * * * * * * * *
   低│
     └────────────────────────────
       0  1  2  3  4  5 ... 10 11
               Layer
```

- ✅ 浅層（0-2）: 中〜高F（入力処理）
- ✅ 中間層（3-9）: 低F（効率的な意味処理）
- ✅ 深層（10-11）: 高Fスパイク（出力準備）
- ❌ 単調減少: 0%（全サンプルで非単調）

詳細は [REPORT.md](./REPORT.md) を参照。

## 学習過程との関係

| 過程 | 観測対象 | Fパターン |
|------|----------|----------|
| **学習** | モデル（重み） | step進行で収束 |
| **推論** | ベクトル（表現） | U字型（bathtub curve） |

両者は**異なるパターン**を示す。

## ファイル構成

```
inference_f_trajectory/
├── README.md              # 本ファイル
├── SPEC.md                # 詳細仕様書
├── REPORT.md              # 実験レポート
├── gedig_hidden.py        # Hidden stateベースのgeDIG実装
├── measure_trajectory.py  # F軌跡測定スクリプト
├── visualize_trajectory.py # 結果可視化
└── results/
    ├── trajectory_bert-base-uncased.json
    ├── trajectory_gpt2.json
    ├── trajectory_visualization.png
    ├── layer_comparison.png
    └── summary.json
```

## 実行方法

```bash
# 単体テスト
poetry run python gedig_hidden.py

# 実験実行
poetry run python measure_trajectory.py --model bert-base-uncased
poetry run python measure_trajectory.py --model gpt2

# 可視化
poetry run python visualize_trajectory.py
```

## 関連実験

- **学習過程の検証**: [../pythia_checkpoints/](../pythia_checkpoints/)

---

*Last updated: 2026-02-05*
