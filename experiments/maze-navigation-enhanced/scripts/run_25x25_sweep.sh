#!/usr/bin/env bash
set -euo pipefail

# 25x25 NA/BT/L1-weights sweep runner (pure-memory, pure-geDIG gate)
#
# Usage:
#   bash experiments/maze-navigation-enhanced/scripts/run_25x25_sweep.sh 21 33 57 73 88 101
#
# Env overrides (optional):
#   MAX_STEPS=400 \
#   NA_LIST="-0.0055 -0.0050 -0.0045" \
#   BT_LIST="-0.0130 -0.0120 -0.0105" \
#   W_LIST="DEF C D" \
#   L1_TOPK=10 SP_FORCE_SAMPLES=30 SP_FORCE_BUDGET_MS=3 \
#   BASE_OUT=docs/images/gedegkaisetsu

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(21 33 57 73 88 101)
fi

MAX=${MAX_STEPS:-400}
BASE_OUT=${BASE_OUT:-docs/images/gedegkaisetsu}
NA_LIST=${NA_LIST:--0.0055 -0.0050 -0.0045}
BT_LIST=${BT_LIST:--0.0130 -0.0120 -0.0105}
W_LIST=${W_LIST:-DEF C D}
L1_TOPK=${L1_TOPK:-10}
SP_FORCE_SAMPLES=${SP_FORCE_SAMPLES:-30}
SP_FORCE_BUDGET_MS=${SP_FORCE_BUDGET_MS:-3}

san() { echo "$1" | sed 's/-/m/g; s/\./p/g'; }

weights_for() {
  case "$1" in
    DEF|def|default) echo "1,1,0,0,3,2,0,0";;
    C|c)            echo "1,1,0,0,2.5,1.5,0,0";;
    D|d)            echo "1.15,1.15,0,0,2.5,1.5,0,0";;
    STRONG|strong)  echo "1,1,0,0,6,4,0,0";;
    *)              echo "$1";; # allow explicit CSV
  esac
}

OUT_DIRS=()

for WLAB in ${W_LIST}; do
  W_CSV=$(weights_for "$WLAB")
  for NA in ${NA_LIST}; do
    for BT in ${BT_LIST}; do
      TAG="w$(san "$WLAB")_na$(san "$NA")_bt$(san "$BT")"
      for S in "${SEEDS[@]}"; do
        OUT="${BASE_OUT}/threshold_dfs_live_25_s${S}_memadj_${TAG}"
        OUT_DIRS+=("$OUT")
        echo "== 25x25 seed $S -> $OUT (max-steps=$MAX) =="
        MAZE_COPY_VIEWER=1 \
        \
        MAZE_NA_GE_THRESH="$NA" MAZE_BT_AGG=na_min MAZE_BACKTRACK_THRESHOLD="$BT" \
        MAZE_BT_USE_MEMORY_TRIGGER=0 MAZE_BT_USE_DEADEND=0 MAZE_BT_DYNAMIC=0 MAZE_BACKTRACK_COOLDOWN=0 \
        \
        MAZE_FRONTIER_FROM_MEMORY=1 MAZE_BT_SEM_FORCE_NEAREST_L2=1 \
        MAZE_BT_PATH_MODE=memory_adjacent MAZE_BT_STRICT_MEMORY=1 \
        \
        MAZE_GEDIG_LOCAL_NORM=1 MAZE_GEDIG_SP_GAIN=1 \
        MAZE_FORCE_MULTIHOP=0 \
        \
        MAZE_SP_GLOBAL_ENABLE=0 MAZE_SP_FORCE_ON_NA=1 MAZE_SP_FORCE_SAMPLES="$SP_FORCE_SAMPLES" MAZE_SP_FORCE_BUDGET_MS="$SP_FORCE_BUDGET_MS" \
        \
        MAZE_L1_WEIGHTS="$W_CSV" MAZE_L1_UNIT_NORM=0 MAZE_L1_CAND_TOPK="$L1_TOPK" \
        MAZE_USE_HOP_DECISION=0 \
        python experiments/maze-navigation-enhanced/src/visualization/export_threshold_dfs_run.py \
          --size 25 --seed "$S" --max-steps "$MAX" \
          --out-dir "$OUT" || echo "seed $S run failed or timed out"
      done
    done
  done
done

# Summaries: per-tag and overall
python experiments/maze-navigation-enhanced/scripts/summarize_runs.py "${OUT_DIRS[@]}" \
  > "${BASE_OUT}/sweep_summary_25_memadj.csv"

# Optional: rebuild gallery
python experiments/maze-navigation-enhanced/scripts/build_gallery.py "$BASE_OUT" || true

echo "Done. Summary: ${BASE_OUT}/sweep_summary_25_memadj.csv"
