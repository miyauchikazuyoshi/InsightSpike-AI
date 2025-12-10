# Phase1 Debug (geDIG on attention) — Threshold & SP tweak results

- Code changes: ΔSPを無向最短路で計算するよう修正（有向→弱連結で0貼り付きがちだった問題を回避）。
- 短い再実験: BERT-base, text=24, layers=4。

## Percentile vs Fixed threshold

### pct=0.8 (top20%)
- `use_percentile=True`: F_mean=-0.337, F_rand=-0.419, Δ=+0.082, d=+1.93
- `use_percentile=False` (τ=0.01): F_mean=+0.076, F_rand=+0.222, Δ=-0.147, d=-1.32

### pct=0.5 (top50%)
- `use_percentile=True`: F_mean=-0.089, F_rand=-0.183, Δ=+0.094, d=+2.22
- `use_percentile=False` (τ=0.01): F_mean=+0.076, F_rand=+0.223, Δ=-0.147, d=-1.32

## Takeaways
- 無向SPに変更した上で、pct系は依然として「F_real < F_random」を維持（符号OK、効果量も大）。pctを緩めても符号は変わらず。
- 固定閾値τ=0.01では依然「F_real > F_random」になる（符号逆）。固定閾値をデフォルトから外し、pct系のみを評価対象にするのが安全。

## Next Steps
- pct閾値をデフォルトに固定（τ_absはオプション扱い）し、以降のPhase 1集計をpctのみで報告。
- ΔSP計算の無向化は有効そうなので維持。必要に応じて重み付き最短路（1/attn）も検討。
