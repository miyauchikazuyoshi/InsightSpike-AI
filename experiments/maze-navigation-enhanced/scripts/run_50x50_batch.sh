#!/usr/bin/env bash
set -euo pipefail

# 50x50 maze batch runner with current NA/DA two-stage gating + L1 configuration.
# Speed tweaks: enable L1 index search and apply a spatial gate to reduce candidate fan-out.
#
# Usage:
#   bash experiments/maze-navigation-enhanced/scripts/run_50x50_batch.sh [seed1 seed2 ...]
# If no seeds are provided, a default list of 20 seeds is used.
#
# Optional env overrides:
#   MAZE_MAX_STEPS_50=2000            # step cap (default 2000)
#   SPATIAL_GATE=18                    # Manhattan radius gate for candidate screening
#   ENABLE_INDEX=1                     # 1 to enable vector-index L1 candidate search

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(11 17 23 29 31 37 41 47 53 59 61 67 71 73 79 83 89 97 101 107)
fi

MAX_STEPS=${MAZE_MAX_STEPS_50:-2000}
# Spatial/L1 knobs (overridable)
SPATIAL_GATE=${SPATIAL_GATE:-18}
ENABLE_INDEX=${ENABLE_INDEX:-1}

OUT_ROOT="experiments/maze-navigation-enhanced/results/50x50_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"

echo "Writing results under ${OUT_ROOT}"

for S in "${SEEDS[@]}"; do
  OUT_DIR="${OUT_ROOT}/seed_${S}"
  mkdir -p "${OUT_DIR}"
  echo "== 50x50 seed ${S} -> ${OUT_DIR} =="

  MAZE_NA_GE_THRESH=-0.0050 \
  MAZE_BT_AGG=na_min \
  MAZE_BT_USE_DA_IF_NA=1 \
  MAZE_BACKTRACK_THRESHOLD=-0.0120 \
  MAZE_BACKTRACK_COOLDOWN=0 \
  MAZE_BT_DYNAMIC=0 \
  MAZE_BACKTRACK_TARGET=heuristic \
  MAZE_FRONTIER_FROM_MEMORY=1 \
  MAZE_BT_SEM_FORCE_NEAREST_L2=1 \
  MAZE_BT_PATH_MODE=memory_adjacent \
  MAZE_BT_STRICT_MEMORY=1 \
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
  MAZE_SPATIAL_GATE="$SPATIAL_GATE" \
  MAZE_SP_FORCE_ON_NA=1 \
  MAZE_SP_FORCE_SAMPLES=30 \
  MAZE_SP_FORCE_BUDGET_MS=3 \
  python experiments/maze-navigation-enhanced/src/visualization/export_threshold_dfs_run.py \
    --size 50 \
    --seed "${S}" \
    --max-steps "${MAX_STEPS}" \
    --strategy gedig \
    --out-dir "${OUT_DIR}" \
    || echo "seed ${S} failed"
done

# Generate summary CSV across all seeds
python experiments/maze-navigation-enhanced/scripts/summarize_runs.py "${OUT_ROOT}"/seed_* > "${OUT_ROOT}/summary.csv"
echo "Summary written to ${OUT_ROOT}/summary.csv"
