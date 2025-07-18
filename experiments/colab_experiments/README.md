# Google Colab Experiments for InsightSpike-AI

This directory contains large-scale experiments designed for Google Colab, following the experimental guidelines in CLAUDE.md.

## 🚀 Quick Start

### 1. Setup Colab Environment

```python
# In Colab, run:
!git clone https://github.com/Sunwood-ai-labs/InsightSpike-AI.git
%cd InsightSpike-AI
!python experiments/colab_setup.py
```

### 2. Run Experiments

```python
# Import experiment modules
from experiments.colab_experiments.insight_benchmarks import run_rat_benchmark, run_all_benchmarks
from experiments.colab_experiments.scalability_testing import test_memory_scaling, test_all_scalability
from experiments.colab_experiments.comparative_analysis import compare_quality, compare_all_systems

# Run specific experiments
rat_results = run_rat_benchmark(n_problems=100)
scalability_results = test_memory_scaling([100, 1000, 10000])
comparison_results = compare_all_systems(n_queries=100)
```

## 📁 Experiment Structure

Following CLAUDE.md guidelines:

```
colab_experiments/
├── src/                           # Experiment programs
│   ├── colab_setup.py            # Environment setup
│   ├── insight_benchmarks.py     # Insight discovery benchmarks
│   ├── scalability_testing.py    # Performance at scale
│   └── comparative_analysis.py   # InsightSpike vs RAG comparison
├── data/
│   ├── input/                    # Input data (read-only)
│   └── processed/                # Processed data
├── results/
│   ├── metrics/                  # Evaluation metrics
│   ├── outputs/                  # JSON outputs
│   └── visualizations/           # Plots and graphs
├── data_snapshots/               # Data backups
└── README.md                     # This file
```

## 🧪 Available Experiments

### 1. Insight Task Benchmarks (`insight_benchmarks.py`)
Based on `docs/experiment_design/01_insight_task_benchmarks.md`

- **Remote Associates Test (RAT)**: 100+ problems testing creative insight
- **Scientific Discovery**: Pattern recognition in scientific data
- **Analogy Completion**: Complex analogy reasoning
- **Creative Problem Solving**: Non-obvious solution finding

```python
# Run comprehensive benchmarks
results = run_all_benchmarks(use_wandb=True)
```

### 2. Scalability Testing (`scalability_testing.py`)
Based on `docs/experiment_design/05_scalability_testing.md`

- **Memory Scaling**: Test with 100 to 100k+ episodes
- **Graph Scaling**: Performance with increasing graph sizes
- **Concurrent Users**: Load testing with multiple users
- **Model Size Comparison**: TinyLlama vs larger models

```python
# Test all scalability dimensions
results = test_all_scalability()
```

### 3. Comparative Analysis (`comparative_analysis.py`)
Based on `docs/experiment_design/04_comparative_analysis.md`

- **Quality Comparison**: InsightSpike vs baseline RAG
- **Performance Benchmarks**: Speed and throughput analysis
- **Insight Discovery**: Creative task performance
- **Domain-Specific Tests**: General, scientific, creative, analytical

```python
# Run comprehensive comparison
results = compare_all_systems(n_queries=100)
```

## 📊 Experiment Tracking

### Weights & Biases Integration

```python
# Login to W&B
import wandb
wandb.login()

# Run experiments with tracking
suite = InsightBenchmarkSuite(use_wandb=True)
results = suite.run_comprehensive_benchmark()
```

### Google Drive Integration

Results are automatically saved to:
- Local: `/content/InsightSpike-AI/experiments/colab_experiments/results/`
- Drive: `/content/drive/MyDrive/InsightSpike_Results/` (if mounted)

## 🔧 Configuration

### Colab-Optimized Settings

The `colab_config.yaml` is automatically created with:
- GPU acceleration enabled
- Larger batch sizes (128)
- Extended memory limits
- Optimized retrieval parameters

### Resource Management

```python
# Monitor GPU usage
!nvidia-smi

# Clear GPU memory between experiments
import torch
torch.cuda.empty_cache()

# Check available memory
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
```

## 📈 Expected Results

### Performance Targets
- **RAT Benchmark**: >50% accuracy with InsightSpike
- **Scalability**: Sub-linear scaling up to 100k episodes
- **Comparison**: 2-3x quality improvement over baseline RAG

### Output Files
- `comprehensive_benchmark_YYYYMMDD_HHMMSS.json`: Full benchmark results
- `scalability_report_YYYYMMDD_HHMMSS.md`: Performance analysis
- `comparative_report_YYYYMMDD_HHMMSS.md`: System comparison

## 🚨 Important Notes

Following CLAUDE.md principles:

1. **No Cheating**: Experiments use genuine processing, no mocks
2. **Data Integrity**: Original data in `/data/input/` is never modified
3. **Reproducibility**: All experiments use fixed random seeds
4. **Transparency**: Full logging and result documentation

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce batch size
   config.processing.batch_size = 32
   
   # Use smaller models
   config.llm.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

2. **Multiprocessing Errors**
   ```python
   # Already handled in setup
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   ```

3. **GPU Not Available**
   - Check runtime type: Runtime → Change runtime type → GPU

## 📚 References

- [CLAUDE.md](../../CLAUDE.md): Experiment execution guidelines
- [experiments/README.md](../README.md): General experiment structure
- [docs/experiment_design/](../../docs/experiment_design/): Detailed experiment designs

## 🤝 Contributing

When adding new experiments:
1. Follow the standard structure in CLAUDE.md
2. Include data integrity checks
3. Provide both quick and comprehensive test functions
4. Generate visualization plots
5. Create detailed reports

---

*These experiments are designed for academic evaluation of InsightSpike's capabilities at scale.*