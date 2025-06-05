# InsightSpike-AI Colab Deployment Guide

## 🚀 Production-Level Large-Scale Experiments

This guide provides comprehensive instructions for deploying InsightSpike-AI in Google Colab for production-level insight discovery testing with GPU acceleration.

## 📋 Prerequisites

- Google Colab Pro (recommended for GPU access)
- At least 16GB RAM (25GB+ for comprehensive experiments)
- CUDA-compatible GPU (T4, V100, A100)

## 🔧 Quick Setup

### 1. Clone Repository
```bash
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
```

### 2. Enhanced Environment Setup
```bash
# Run enhanced setup script
!chmod +x scripts/setup_colab.sh
!./scripts/setup_colab.sh
```

### 3. Verify Installation
```bash
# Run comprehensive validation
!PYTHONPATH=src python scripts/final_validation.py

# Test HuggingFace dataset integration
!PYTHONPATH=src python scripts/test_hf_dataset_integration.py --gpu
```

## 🧪 Experiment Modes

### Quick Validation (100 samples)
```bash
!PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode quick
```

### Standard Experiment (1,000 samples)
```bash
!PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode standard
```

### Comprehensive Analysis (5,000+ samples)
```bash
!PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode comprehensive
```

## 📊 Supported Datasets

| Dataset | Domain | Size | Description |
|---------|--------|------|-------------|
| SQuAD | Reading Comprehension | 11K questions | Stanford Question Answering Dataset |
| CosmosQA | Commonsense | 35K questions | Commonsense reading comprehension |
| Math QA | Mathematics | 37K problems | Mathematical word problems |
| ScienceQA | Science | 21K questions | Science question answering |

## 🔍 Performance Monitoring

### GPU Utilization
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.memory_allocated() // 1024**2}MB")
```

### System Resources
```bash
# Check memory and disk usage
!free -h
!df -h
```

## 📈 Expected Performance

### Hardware Recommendations

| Hardware | Quick Mode | Standard Mode | Comprehensive Mode |
|----------|------------|---------------|-------------------|
| T4 GPU | 5-10 min | 30-60 min | 2-4 hours |
| V100 GPU | 2-5 min | 15-30 min | 1-2 hours |
| A100 GPU | 1-3 min | 10-20 min | 30-60 min |

### Throughput Benchmarks

- **Text Processing**: 100-500 samples/sec (GPU-accelerated)
- **Insight Detection**: 50-200 insights/sec  
- **Graph Analysis**: 10-50 graphs/sec

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
!PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode quick --batch-size 8
```

#### 2. Dataset Loading Timeout
```bash
# Use smaller dataset subset
!PYTHONPATH=src python scripts/test_hf_dataset_integration.py --max-samples 100
```

#### 3. Package Installation Errors
```bash
# Reinstall with force
!pip install --force-reinstall torch torchvision torchaudio
```

### Memory Optimization

```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()
```

## 📊 Results Analysis

### Experiment Outputs

- **`experiment_results/`**: JSON results and metrics
- **`experiment_results/visualizations/`**: Generated plots and dashboards
- **`logs/`**: Detailed execution logs

### Key Metrics

1. **Insight Discovery Rate**: Insights found per minute
2. **Processing Throughput**: Samples processed per second  
3. **GPU Utilization**: Memory and compute usage
4. **Accuracy Metrics**: Precision, recall, F1 scores

## 🔬 Advanced Usage

### Custom Dataset Integration
```python
from scripts.colab_large_scale_experiment import LargeScaleDataLoader

loader = LargeScaleDataLoader()
custom_data = loader.load_custom_dataset("your_dataset_name")
```

### Parameter Tuning
```bash
# Custom configuration
!PYTHONPATH=src python scripts/colab_large_scale_experiment.py \
  --mode standard \
  --max-questions 2000 \
  --batch-size 16 \
  --output-dir custom_results
```

### Multi-GPU Setup (Pro+ only)
```python
# Enable multi-GPU processing
config = ExperimentConfig(
    use_gpu=True,
    multi_gpu=True,
    devices=["cuda:0", "cuda:1"]
)
```

## 📋 Validation Checklist

- [ ] GPU acceleration enabled
- [ ] All dependencies installed
- [ ] Dataset loading successful  
- [ ] Memory usage within limits
- [ ] Output directories created
- [ ] Baseline performance validated

## 🆘 Support

### Error Reporting
- Include GPU model and memory size
- Attach relevant log files
- Specify experiment mode and parameters

### Performance Issues
- Monitor GPU utilization with `nvidia-smi`
- Check memory usage with system tools
- Consider reducing batch size or sample count

## 📚 Additional Resources

- [Colab Pro Documentation](https://colab.research.google.com/signup)
- [PyTorch GPU Guide](https://pytorch.org/get-started/locally/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [InsightSpike-AI Documentation](README.md)

---

*This guide is optimized for production-level experiments. For development setup, see the main README.md.*
