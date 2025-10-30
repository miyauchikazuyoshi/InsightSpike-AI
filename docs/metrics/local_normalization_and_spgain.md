Local Normalization and Multi-hop Shortcut Gain (ΔSP_norm)

Overview
- Single-score geDIG: F = ΔGED_norm − λ·ΔIG_norm
- Local normalization (decision-time control): override the denominator of ΔGED_norm with a local upper bound Cmax_local^(0) derived from immediate editing opportunity.
- Multi-hop shortcut gain: reduce the structural term by normalized average shortest-path shrinkage to reflect MDL-like benefits of long-range integration.

Local Normalization (Layer1-aware)
- Denominator: Cmax_local^(0) = 1 + K
  - K = size of Layer1 candidate set for the current step.
  - In maze-navigation-enhanced, K counts neighbors that pass the Layer1 norm search gate (and optional spatial gate):
    - Norm search: L2(query, episode) ≤ τ
    - Optional weighted L2: element-wise weights emphasize wall/visit dimensions.
- Effect: stabilizes argmin + single-threshold decisions by scaling ΔGED to the local action budget.
- Configuration (env):
  - MAZE_GEDIG_LOCAL_NORM=1      Enable local normalization in GeDIGCore
  - MAZE_L1_NORM_SEARCH=1        Enable Layer1 norm gate (candidates K)
  - MAZE_L1_WEIGHTED=1           Use weighted L2 for norm gate
  - MAZE_L1_NORM_TAU=0.75        Norm threshold τ (tune 0.6–0.8)
  - MAZE_L1_WEIGHTS="1,1,0,0,3,2,0,0"  8-d weights (x,y,dx,dy,wall,visits,success,goal)
  - MAZE_WIRING_WINDOW=6         Temporal window for candidates
  - MAZE_SPATIAL_GATE=4          Manhattan gate to constrain candidates

Multi-hop Shortcut Gain (ΔSP_norm)
- Definition (per hop h ≥ 1 on induced k-hop subgraphs around focal nodes):
  - L_before = mean shortest-path length before wiring
  - L_after  = mean shortest-path length after wiring
  - ΔSP_norm^(h) = max(0, L_before − L_after) / L_before ∈ [0,1]
- Effective structural term:
  - ΔGED_eff^(h) = ΔGED_norm − ΔSP_norm^(h)
  - 0-hop: edit cost dominates (FEP-like). h-hop: shortcut gain reduces structural burden (MDL-like).
- Configuration (env):
  - MAZE_GEDIG_SP_GAIN=1         Enable SP gain injection in multihop path
  - MAZE_GEDIG_SP_MODE=relative  Use relative normalization (default)
- Performance guards (built-in in GeDIGCore):
  - Sampling for large subgraphs; parameters (defaults) are conservative:
    - sp_node_cap=200, sp_pair_samples=400, sp_use_sampling=True

Recommended Starting Set (lightweight)
- export MAZE_GEDIG_LOCAL_NORM=1
- export MAZE_L1_NORM_SEARCH=1
- export MAZE_L1_WEIGHTED=1
- export MAZE_L1_NORM_TAU=0.75
- export MAZE_WIRING_WINDOW=6
- export MAZE_SPATIAL_GATE=4
- export MAZE_GEDIG_SP_GAIN=0   # begin with SP gain OFF; enable after baseline

Notes
- Reporting vs Control: control uses local normalization; reporting can retain global normalization for cross-run comparability.
- Thresholding: consider quantile-based thresholds (e.g., q5–10%) or EMA smoothing to further stabilize decisions under local normalization.
