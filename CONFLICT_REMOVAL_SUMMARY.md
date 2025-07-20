# Conflict Term Removal from InsightSpike Reward Calculation

## Summary of Changes

The conflict term has been successfully removed from InsightSpike's reward calculation. The reward formula has been simplified from:

**Old Formula:**
```
R(w₁,w₂,w₃) = w₁ × ΔGED + w₂ × ΔIG - w₃ × ConflictScore
```

**New Formula:**
```
R = w₁ × ΔGED + w₂ × ΔIG
```

## Files Modified

### 1. Core Metrics Module (`src/insightspike/metrics/__init__.py`)
- Updated `DEFAULT_WEIGHTS` to only include `ged` and `ig` (both 0.5)
- Modified `compute_fusion_reward()` to ignore conflict_score parameter (kept for backward compatibility)
- Updated all preset configurations to use 2-term weights
- Updated documentation strings and examples
- Simplified the fusion reward formula in metadata

### 2. Algorithms Module (`src/insightspike/algorithms/__init__.py`)
- Updated `ALGORITHM_INFO` to reflect the simplified fusion scheme
- Changed default weights from `{ged: 0.4, ig: 0.3, conflict: 0.3}` to `{ged: 0.5, ig: 0.5}`

### 3. Configuration Files
- **`src/insightspike/config/models.py`**: Set `weight_conflict` default to 0.0 with deprecation note
- **`src/insightspike/config/legacy_config.py`**: Set `weight_conflict` to 0.0 with deprecation note
- **`src/insightspike/config/constants.py`**: Updated default weights and added deprecation comment

### 4. Reward Calculator (`src/insightspike/features/graph_reasoning/reward_calculator.py`)
- Removed conflict weight from initialization
- Simplified base reward calculation to exclude conflict term
- Modified novelty reward to ignore conflicts parameter

## Backward Compatibility

The changes maintain backward compatibility:
- `conflict_score` parameter is still accepted by `compute_fusion_reward()` but ignored
- `conflict_threshold` configuration remains available but unused
- `ConflictScore` class and conflict calculation remain in the codebase but don't affect rewards

## New Default Weights

All weight configurations now use two terms that sum to 1.0:
- **Default**: `ged=0.5, ig=0.5` (balanced)
- **Research**: `ged=0.6, ig=0.4` (structure-focused)
- **Education**: `ged=0.3, ig=0.7` (learning-focused)
- **Structure-focused**: `ged=0.8, ig=0.2`

## Testing

A new test file (`tests/unit/test_fusion_reward.py`) has been created to verify:
- Correct calculation with default weights
- Correct calculation with custom weights
- Backward compatibility (conflict_score parameter is ignored)
- Various reward scenarios (positive, negative, eureka spike)

## Impact

This simplification:
1. Makes the reward calculation more interpretable
2. Reduces the number of hyperparameters to tune
3. Focuses on the two core metrics that define insight detection
4. Maintains backward compatibility for existing code