# Pythia チェックポイント実験

**ステータス**: 🔬 実験完了・考察中

## 概要

Pythiaの公開チェックポイントを用いて、**学習過程におけるgeDIG F値の変化**を検証した実験。

現在の実装は attention ベースで、構造項を切替できる:
- `--structural-term sp`（従来）
- `--structural-term betti1`（B定義: attentionグラフのベッチ数）

## 仮説

学習が進むにつれて：
- EPC（構造変化コスト）が増加
- H（エントロピー）が減少
- F値の変化量（ΔF）が単調減少（収束）

## 結果サマリー

- ✅ H減少を確認（1.31 → 0.60）
- ✅ EPC増加を確認（5.74 → 7.43）
- ✅ |ΔF|の収束を確認
- ⚠️ SPは測定不可（Causal LMにはCLSがない）

詳細は [REPORT.md](./REPORT.md) を参照。

## 関連実験

- **推論過程の検証**: [../inference_f_trajectory/](../inference_f_trajectory/) （仕様検討中）

## 今後の課題

- [ ] Causal LM用のSP定義（Top-k占有率等）
- [ ] 先行研究との詳細比較
- [ ] 論文への組み込み方検討

## 実行例（Attention-B）

軽量チェックポイント:

```bash
python experiments/transformer/pythia_checkpoints/analyze_training_dynamics.py \
  --light \
  --samples 10 \
  --structural-term betti1 \
  --betti-k-neighbors 5
```

任意チェックポイント:

```bash
python experiments/transformer/pythia_checkpoints/analyze_training_dynamics.py \
  --checkpoints 0,64,512 \
  --samples 10 \
  --structural-term betti1
```

---

*Last updated: 2026-02-03*
