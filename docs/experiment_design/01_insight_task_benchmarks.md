# Experiment Design: Insight Task Benchmarks

## Overview

Evaluate InsightSpike's performance on standardized creative insight tasks from cognitive psychology, demonstrating its unique ability to generate "Aha!" moments that traditional LLMs and RAG systems cannot achieve.

## Background

The core innovation of InsightSpike is creative insight detection and generation. By testing on established psychological benchmarks for insight, we can objectively demonstrate that our system captures something fundamentally different from information retrieval or pattern matching.

## Experimental Design

### 1. Remote Associates Test (RAT)

**Task Description**: Given three seemingly unrelated words, find a fourth word that connects to all three.

**Example**:
- Input: "COTTAGE, SWISS, CAKE"
- Answer: "CHEESE" (cottage cheese, Swiss cheese, cheesecake)

**Implementation**:
```python
rat_problems = [
    {"words": ["COTTAGE", "SWISS", "CAKE"], "answer": "CHEESE"},
    {"words": ["CREAM", "SKATE", "WATER"], "answer": "ICE"},
    {"words": ["SHOW", "LIFE", "ROW"], "answer": "BOAT"},
    # ... 50-100 problems from established RAT datasets
]
```

**Evaluation Metrics**:
- Accuracy: % of correct answers
- Insight detection rate: % where spike was detected
- Response time
- Reasoning path analysis

### 2. Riddle Solving

**Task Description**: Solve riddles that require lateral thinking and conceptual bridges.

**Example**:
- "What gets wet while drying?" → "A towel"
- "What has keys but no locks?" → "A piano"

**Dataset**: 
- Classic riddles dataset (100-200 items)
- Categorized by difficulty and insight type

### 3. Compound Remote Associates (CRA)

**Task Description**: Find compound words or phrases connecting word pairs.

**Example**:
- Input: "CROSS/RAIN"
- Answer: "BOW" (crossbow, rainbow)

### 4. Insight Problem Solving

**Classic Problems**:
- Nine-dot problem
- Candle problem
- Two-string problem

**Adaptation for LLMs**: Convert spatial/physical problems to verbal descriptions and evaluate solution generation.

## Baseline Comparisons

### Systems to Compare:
1. **Direct LLM** (GPT-3.5/4, Claude)
2. **Traditional RAG** with knowledge base
3. **Chain-of-Thought prompting**
4. **InsightSpike**

### Expected Results:
- Traditional systems: 10-30% accuracy (mostly pattern matching)
- InsightSpike: 60-80% accuracy with insight detection

## Implementation Plan

### Phase 1: Data Preparation (Week 1)
- Collect standardized RAT problems
- Compile riddle datasets
- Create evaluation framework

### Phase 2: Baseline Testing (Week 2)
- Test all baseline systems
- Document failure modes
- Establish performance floor

### Phase 3: InsightSpike Testing (Week 3)
- Run InsightSpike on all tasks
- Analyze spike detection patterns
- Document reasoning paths

## Success Criteria

1. **Primary**: InsightSpike achieves >2x accuracy of best baseline
2. **Secondary**: Spike detection correlates with correct answers (r > 0.7)
3. **Qualitative**: Generated explanations show genuine conceptual bridging

## Resources Required

- RAT problem sets (available from psychology research)
- API access for baseline LLMs
- 2-3 weeks developer time
- Minimal compute resources

## Expected Impact

This experiment directly validates InsightSpike's core claim: it can generate creative insights that other systems cannot. Success here provides strong evidence that our approach captures something fundamental about human insight generation.

## Code Structure

```
experiments/insight_benchmarks/
├── src/
│   ├── rat_solver.py
│   ├── riddle_solver.py
│   └── evaluation.py
├── data/
│   ├── input/
│   │   ├── rat_problems.json
│   │   └── riddles.json
│   └── processed/
├── results/
│   ├── baseline_performance.csv
│   ├── insightspike_performance.csv
│   └── comparative_analysis.json
└── README.md
```

## References

- Bowden, E. M., & Jung-Beeman, M. (2003). Normative data for 144 compound remote associate problems.
- Mednick, S. (1962). The associative basis of the creative process.
- Psychological benchmarks for insight: [Link to resources]