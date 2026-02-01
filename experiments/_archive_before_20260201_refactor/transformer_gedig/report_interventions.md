# Intervention Summary (DistilBERT SST-2 mini, all-layer vs single-layer attention edits)

Source: `intervene_summary.json` and `fig_intervention_summary.png` after adding single-layer (last) mid-strength interventions.

- Setup: DistilBERT fine-tuned on SST-2 (mini batch of 32 for eval). Interventions applied either
  - to **last layer only** (mid-strength), or
  - to **all layers** (strong; kept for reference).
- Baseline: accuracy 0.96875, F_mean -0.2505, conf_vs_F_corr 0.047.

## Mid-strength (last layer only)
- Sparsify top-16 / top-32 (last): acc 0.469, F_mean ≈ -0.395, conf_vs_F_corr ≈ -0.002
- Noise scale 0.05 / 0.1 (last): acc 0.469, F_mean ≈ -0.395, conf_vs_F_corr ≈ -0.002
→ 精度は大幅低下（~47%）まで落ちるが、conf_vs_F_corr はほぼ0で安定。

## Strong (all layers, reference)
- Sparsify top-4 (all): acc 0.563, F_mean -0.394, conf_vs_F_corr 0.550
- Noise scale 0.5 (all): acc 0.469, F_mean -0.362, conf_vs_F_corr -0.055

## Takeaways
- 全層介入より弱い「単一層介入」でも精度は大きく崩れる（~47%）。中強度域で精度70–90%を保つ設定は未発見。
- conf_vs_F_corr は強度・適用層で符号が揺れやすく、精度との線形関係は弱い。
- F_mean は介入強度に応じて低下し、構造破壊の検知としては安定している。

## Next Steps
- 介入強度をさらに弱める（例: top-64/128、noise 0.02）か、特定層のみ（中間層）に適用して、精度が部分的に落ちる範囲を探索する。
- 層別感度分析：層ごとに同一強度を適用し、精度/F の変化をプロファイルする。
- conf_vs_F_corr は補助指標とし、F_mean と精度低下の閾値を優先的に可視化する。 
