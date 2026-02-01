# Structural Similarity Experiments

This directory contains experiments for validating the structural similarity feature in geDIG.

## Experiments

### 1. Analogy Detection Benchmark (`analogy_benchmark.py`)

Tests whether the structural similarity evaluator can correctly identify known analogies and reject non-analogous pairs.

**Test Cases:**
- Known Analogies (should detect):
  - Solar system ≈ Atom (hub-spoke pattern)
  - Company org ≈ Military hierarchy (tree pattern)
  - Blood vessels ≈ River delta (branching pattern)
  - Benzene ring ≈ Ouroboros (cyclic pattern)
  - City grid ≈ Circuit board (grid pattern)
  - Actors-Movies ≈ Students-Courses (bipartite pattern)
  - Bicycle wheel ≈ Router wheel (hub-ring pattern)
  - Supply chain ≈ Relay race (chain pattern)

- Non-Analogies (should NOT detect):
  - Solar system vs Chain (different structures)
  - Tree vs Clique (hierarchy vs complete)
  - Ring vs Star (cycle vs hub-spoke)
  - Grid vs Chain (2D vs 1D)
  - Branching vs Clique (sparse vs dense)
  - Bipartite vs Clique (two-part vs complete)
  - Wheel vs Chain (hub-ring vs linear)
  - Lollipop vs Ring (clique+tail vs cycle)
  - Wheel vs Ring (hub-ring vs cycle)
  - Ladder vs Chain (2-track vs linear)

- Hard Negatives (same-domain structure matches; should be rejected when cross-domain only):
  - Solar system vs Exoplanet system (astronomy)
  - Company vs Startup hierarchy (business)
  - Benzene vs Cyclohexane rings (chemistry)
  - Students-Courses vs Teachers-Classes (education)
  - Bicycle wheel vs Gear wheel (mechanics)

- Noisy Variants (robustness):
  - Edge add/remove noise applied to positive and negative pairs

**Metrics:**
- Precision, Recall, F1 Score, Accuracy, FPR (ROC)
- Threshold sweep prints PR/ROC-style comparison
- Threshold is tuned on a validation split, then reported on a holdout split
- Multi-seed runs summarize threshold stability (mean/std for holdout metrics)

**Usage:**
```bash
cd /path/to/InsightSpike-AI
.venv/bin/python -m experiments.structural_similarity.analogy_benchmark
```
Default benchmark settings: `method="motif"`, `analogy_threshold` tuned on validation (typical 0.9)
If you see import failures in a restricted environment, run with `INSIGHTSPIKE_LITE_MODE=1`.

**Tuning Options:**
- `selection_mode="fpr_min_recall"` chooses the lowest FPR while enforcing `recall_min`
- `selection_mode="f1"` maximizes F1 (fallback when recall constraint cannot be met)
- `selection_mode="precision"` maximizes precision (optionally enforces `recall_min`, then tie-breaks on FPR, F1, higher threshold)
- `seeds=[...]` runs multiple splits and reports stability

### 2. Science History Simulation (`science_history_simulation.py`)

Simulates historical scientific discoveries that were based on analogical reasoning. Tests whether geDIG can "rediscover" these insights.

**Discoveries Simulated:**

| Discovery | Year | Scientist | Analogy |
|-----------|------|-----------|---------|
| Atomic Model | 1913 | Bohr | Solar system → Atom |
| Benzene Ring | 1865 | Kekulé | Ouroboros → Ring structure |
| Natural Selection | 1859 | Darwin | Malthus population theory → Species competition |

**What it tests:**
1. Can geDIG detect structural similarity between source and target domains?
2. Does the external analogy bonus change geDIG as expected?
3. Does connecting the analogy insight affect internal SS behavior?

**Usage:**
```bash
cd /path/to/InsightSpike-AI
.venv/bin/python -m experiments.structural_similarity.science_history_simulation
```

## Configuration

Experiments use the `StructuralSimilarityConfig` pattern. The analogy benchmark
defaults to `method="motif"` and tunes `analogy_threshold` on a validation split
unless overridden. Example:

```yaml
graph:
  structural_similarity:
    enabled: true
    method: "motif"  # or "signature", "spectral"
    analogy_threshold: 0.9
    analogy_weight: 0.3
    cross_domain_only: true
```

The motif method uses a weighted cosine signature with a small penalty for
hub-ratio and square-count mismatches to reduce false positives.

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `method` | Similarity method (signature, spectral, motif) | motif |
| `analogy_threshold` | Minimum similarity to trigger analogy detection | 0.9 |
| `analogy_weight` | Weight for analogy bonus in IG | 0.3 |
| `cross_domain_only` | Only reward cross-domain analogies | true |

## Expected Results

### Analogy Benchmark
- Precision/Recall/F1 should stay > 0.7 with noisy + hard-negative cases enabled
- FPR should remain low (goal: < 0.2) on holdout split

### Science History Simulation
- All 3 historical analogies should be detected
- Structural similarity should be > 0.5 for valid analogies
- External insight bonus should shift geDIG relative to the no-SS baseline
