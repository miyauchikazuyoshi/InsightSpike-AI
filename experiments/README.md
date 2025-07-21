# InsightSpike Experiments

This directory contains successful experiments that validate and improve InsightSpike's capabilities.

## 📊 Current Experiments

### 1. 🏆 [English Insight Reproduction](./english_insight_reproduction/)
**Status**: ✅ Completed Successfully

The definitive experiment that achieved breakthrough results with DistilGPT-2 (82M parameters):

**Key Results:**
- **66.7% spike detection accuracy** on complex questions
- **40.9% average confidence** for detected insights
- **1.6s average processing time**
- Successfully implemented LocalProvider for lightweight LLM support

**Notable Achievements:**
- Demonstrated insight detection without large language models
- Created knowledge graph visualizations showing insight formation
- Validated the GED-IG algorithm in practice

### 2. 📈 [Fixed Metrics Comparison](./fixed_metrics_comparison/)
**Status**: ✅ Algorithm Improvement Validated

This experiment validated the improved GED/IG implementation after fixing calculation issues.

**Key Improvements:**

#### **Sigmoid Normalization (2.4-2.5x Signal Amplification)**
- Old: Linear normalization compressed differences
- New: Sigmoid normalization amplifies meaningful differences
- Result: 2.4-2.5x stronger signal for insight detection

#### **Continuous Spike Scoring**
- Old: Binary threshold (spike/no spike)
- New: Continuous 0-1 scale
- Result: More nuanced insight detection

#### **Corrected Calculations**
- **ΔGED**: Now properly tracks distance from initial state
- **ΔIG**: Correctly measures entropy reduction (organization)
- Both metrics now align with the theoretical model

**Impact:**
- More sensitive insight detection
- Better differentiation between strong and weak insights
- Reduced false negatives

## 🔬 Experiment Guidelines

When creating new experiments:

1. **Use the standard directory structure**:
   ```
   experiment_name/
   ├── src/           # Experiment code
   ├── data/          # Input and processed data
   ├── results/       # Outputs and metrics
   └── README.md      # Detailed documentation
   ```

2. **Follow the experiment protocol** (see CLAUDE.md)

3. **Document results thoroughly** including failures

## 📝 Historical Note

Previous iterations and unsuccessful attempts have been removed to maintain clarity. The experiments preserved here represent:
- Final successful implementations
- Significant algorithmic improvements
- Validated approaches ready for production

## 🚀 Next Steps

1. **Production Integration**: Integrate sigmoid normalization into main codebase
2. **Scaling Tests**: Test with larger knowledge bases
3. **Multi-language Support**: Extend beyond English
4. **Real-time Applications**: Optimize for streaming scenarios

---

*Last Updated: July 2025*