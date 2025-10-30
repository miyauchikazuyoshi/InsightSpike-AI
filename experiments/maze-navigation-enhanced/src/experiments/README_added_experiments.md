# Added Experiments (Baseline, Ablation, Scaling)

Short descriptions of newly added scripts:

- `baseline_vs_simple_plot.py`: Runs Simple mode, Random, and DFS explorers across seeds for a chosen maze variant (complex or ultra) and produces a loop redundancy boxplot plus raw records.
- `ablation_geDIG.py`: Compares loop redundancy under normal geDIG weights, randomized weights, and zeroed weights to show performance degradation when geDIG signal is removed or scrambled.
- `size_scaling_experiment.py`: Approximates scaling by varying step budget across variants and plots mean loop redundancy for Simple vs Random to assess robustness with longer exploration horizons.
- `deadend_multiseed_variability.py` (previously added): Multi-seed quantile variability analysis for terminal phase geDIG values with weight noise.

Outputs land under `results/maze_report/` with timestamped folders.
