#!/usr/bin/env bash
set -euo pipefail

# 50x50 quick runner (single/multiple seeds)
# Defaults: semantic backtrack target + index search + spatial gate.
# Faster + more local backtracks for interactive runs and debugging.
#
# Usage:
#   bash experiments/maze-navigation-enhanced/scripts/run_50x50_quick.sh 11 17 23
#
# Optional env overrides:
#   MAZE_MAX_STEPS_50=1200            # step cap (default 1200)
#   SPATIAL_GATE=18                    # Manhattan radius gate for candidate screening
#   ENABLE_INDEX=1                     # 1 to enable vector-index L1 candidate search
#   STRATEGY=gedig                     # or gedig_optimized
#   BT_TARGET=semantic                 # semantic | heuristic (default semantic)
#   FRONTIER_SRC=maze                  # maze | memory (default maze)

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(11)
fi

MAX_STEPS=${MAZE_MAX_STEPS_50:-1200}
SPATIAL_GATE=${SPATIAL_GATE:-18}
ENABLE_INDEX=${ENABLE_INDEX:-1}
STRATEGY=${STRATEGY:-gedig}
BT_TARGET=${BT_TARGET:-semantic}
FRONTIER_SRC=${FRONTIER_SRC:-maze}
# Threshold overrides (keep NA as-is; adjust BT only by default)
NA_TAU=${NA_TAU:--0.0050}
BT_TAU=${BT_TAU:-0.0045}

# Resolve frontier flag (1=memory-based frontier, 0=maze-based frontier)
if [ "$FRONTIER_SRC" = "memory" ]; then
  FRONTIER_MEM=1
else
  FRONTIER_MEM=0
fi

OUT_ROOT="experiments/maze-navigation-enhanced/results/50x50_quick_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
echo "Writing quick-run results under ${OUT_ROOT} (strategy=${STRATEGY}, steps=${MAX_STEPS})"
echo "Using thresholds: NA=${NA_TAU}, BT=${BT_TAU}"

for S in "${SEEDS[@]}"; do
  OUT_DIR="${OUT_ROOT}/seed_${S}"
  mkdir -p "${OUT_DIR}"
  echo "== 50x50 seed ${S} -> ${OUT_DIR} =="

  MAZE_NA_GE_THRESH="$NA_TAU" \
  MAZE_BT_AGG=na_min \
  MAZE_BT_USE_DA_IF_NA=1 \
  MAZE_BACKTRACK_THRESHOLD="$BT_TAU" \
  MAZE_BACKTRACK_COOLDOWN=0 \
  MAZE_BT_DYNAMIC=0 \
  MAZE_BACKTRACK_TARGET="$BT_TARGET" \
  MAZE_BT_SEM_DIST=l2_w \
  MAZE_BT_SEM_FORCE_NEAREST_L2=1 \
  MAZE_BT_SEM_SOURCE=index \
  MAZE_BT_SEM_IDX_TOPK=32 \
  MAZE_BT_SEM_IDX_VISITED_ONLY=0 \
  MAZE_BT_SEM_IDX_ALLOW_WALLS=0 \
  MAZE_BT_SEM_SPATIAL_GATE="$SPATIAL_GATE" \
  MAZE_FRONTIER_FROM_MEMORY="$FRONTIER_MEM" \
  MAZE_BT_PATH_MODE=episode_graph \
  MAZE_BT_STRICT_MEMORY=1 \
  MAZE_BT_USE_DEADEND=1 \
  MAZE_BT_SEM_POS_REP=best \
  # Combined ranking assisted BT (obs vs mem dv)
  MAZE_BT_FROM_COMBINED=${MAZE_BT_FROM_COMBINED:-1} \
  MAZE_BT_DV_MARGIN=${MAZE_BT_DV_MARGIN:-0.02} \
  MAZE_BT_REQUIRE_NA=${MAZE_BT_REQUIRE_NA:-1} \
  MAZE_GEDIG_LOCAL_NORM=1 \
  MAZE_GEDIG_SP_GAIN=1 \
  MAZE_FORCE_MULTIHOP=1 \
  MAZE_L1_NORM_SEARCH=1 \
  MAZE_L1_WEIGHTED=1 \
  MAZE_L1_UNIT_NORM=0 \
  MAZE_L1_NORM_TAU=0.75 \
  MAZE_L1_WEIGHTS=1,1,0,0,3,2,0,0 \
  MAZE_L1_CAND_TOPK=10 \
  MAZE_L1_INDEX_SEARCH="$ENABLE_INDEX" \
  # L1 distance threshold wiring (force-accept by dv)
  MAZE_WIRING_L1_ENABLE=${MAZE_WIRING_L1_ENABLE:-1} \
  MAZE_WIRING_L1_DV_TAU=${MAZE_WIRING_L1_DV_TAU:-0.66} \
  MAZE_WIRING_L1_DV_TOPK=${MAZE_WIRING_L1_DV_TOPK:-2} \
  MAZE_SPATIAL_GATE="$SPATIAL_GATE" \
  MAZE_WIRING_TOPK=1 \
  MAZE_WIRING_MIN_ACCEPT=0 \
  MAZE_WIRING_FORCE_PREV=${MAZE_WIRING_FORCE_PREV:-0} \
  MAZE_WIRING_FORCE_L1=0 \
  # Combined ranking (obs+mem) logging for analysis
  MAZE_COMBINED_RANK_ENABLE=${MAZE_COMBINED_RANK_ENABLE:-1} \
  MAZE_COMBINED_TOPK=${MAZE_COMBINED_TOPK:-64} \
  MAZE_COMBINED_LOG_TOPK=${MAZE_COMBINED_LOG_TOPK:-16} \
  MAZE_COMBINED_ALLOW_WALLS=${MAZE_COMBINED_ALLOW_WALLS:-0} \
  MAZE_COMBINED_VISITED_ONLY=${MAZE_COMBINED_VISITED_ONLY:-0} \
  MAZE_COMBINED_REQUIRE_FRONTIER=${MAZE_COMBINED_REQUIRE_FRONTIER:-0} \
  MAZE_COMBINED_SPATIAL_GATE=${MAZE_COMBINED_SPATIAL_GATE:-0} \
  MAZE_SP_GLOBAL_ENABLE=0 \
  MAZE_SP_FORCE_ON_NA=0 \
  MAZE_SP_FORCE_SAMPLES=0 \
  MAZE_SP_FORCE_BUDGET_MS=0 \
  python experiments/maze-navigation-enhanced/src/visualization/export_threshold_dfs_run.py \
    --size 50 \
    --seed "${S}" \
    --max-steps "${MAX_STEPS}" \
    --strategy "${STRATEGY}" \
    --out-dir "${OUT_DIR}" \
    || echo "seed ${S} failed"
done

python experiments/maze-navigation-enhanced/scripts/summarize_runs.py "${OUT_ROOT}"/seed_* > "${OUT_ROOT}/summary.csv" || true
echo "Summary written to ${OUT_ROOT}/summary.csv"
