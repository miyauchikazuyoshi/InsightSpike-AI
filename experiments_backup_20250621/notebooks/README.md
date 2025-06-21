# � InsightSpike-AI Notebooks

**Google Colab & Jupyter Notebooks for InsightSpike-AI v0.8.0**

このディレクトリには、InsightSpike-AIを効率的に使用するためのnotebooksが含まれています。

## 🚀 Getting Started

### ⚡ Quick Start (推奨)
**`InsightSpike_Quick_Start_v0.8.0.ipynb`**
- 🔥 **3ステップ**でInsightSpike-AIを動作
- 📦 **最軽量**セットアップ
- 🧪 **基本実験**付き
- ⏱️ **5分以内**で完了

### 🔧 完全セットアップ & デモ
**`InsightSpike_Colab_Demo.ipynb`**
- 🛠️ **詳細な診断**・修復機能
- 📊 **パフォーマンス分析**
- 🔬 **高度な実験**例
- 🚨 **トラブルシューティング**

## 📂 Notebook Contents

### Official Setup & Experiments (2025対応)
- `InsightSpike_Colab_Setup_2025_fixed.ipynb` - **Official setup guide** with comprehensive dependency management
- `InsightSpike_Colab_Experiments_2025_fixed.ipynb` - **Detailed experiment framework** with statistical analysis

## 🚀 Large-Scale Usage

### For Large-Scale Google Colab Experiments
1. **Recommended Runtime**: A100 GPU for maximum performance
2. **Memory**: High-RAM runtime for 1M+ vectors
3. **Fallback**: V100/T4 GPU for development and testing
4. Open notebook in Google Colab
5. Run cells sequentially - automatic resource monitoring included
6. Follow notebook instructions for large-scale configuration

### Performance Specifications
- **Document Processing**: 100K+ documents with full analysis
- **Vector Operations**: 1M+ embeddings with FAISS optimization
- **Batch Processing**: Automatic batching with checkpointing
- **Memory Management**: Intelligent cleanup and monitoring
- **Fault Tolerance**: Automatic recovery from interruptions

## 🔧 Requirements for Large-Scale Operations

- **Python**: 3.8+ with performance optimizations
- **GPU**: A100 recommended, V100/T4 supported
- **Memory**: High-RAM runtime (25GB+) for large datasets
- **Runtime**: Multi-hour sessions with automatic checkpointing
- **Network**: Stable connection for package management

## 📊 Performance Targets

| Workload | Small Scale | Large Scale | Performance |
|----------|-------------|-------------|-------------|
| Documents | 1K-10K | 100K-1M+ | 10x faster processing |
| Vectors | 10K-100K | 1M+ | GPU-accelerated FAISS |
| Processing Time | Minutes | Hours | Checkpointed |
| Memory Usage | Standard | High-RAM | Optimized |

## 💡 Large-Scale Features

- **Batch Processing**: Configurable batch sizes (1K-10K items)
- **Checkpointing**: Automatic saves every 10K processed items
- **Resource Monitoring**: Real-time CPU, memory, and GPU tracking
- **Error Recovery**: Resume from last checkpoint on interruption
- **Memory Optimization**: Automatic cleanup and garbage collection
- **Progress Tracking**: ETA and throughput monitoring

## 📝 Production Notes

- Notebooks are optimized for **production-scale workloads**
- Automatic fallback methods ensure reliability at scale
- Built-in monitoring prevents resource exhaustion
- Checkpoint system enables multi-session processing
- Safe mode allows testing without external dependencies
- All notebooks support both development and production use cases

## 🎯 Target Applications

- **Research**: Large corpus analysis and insight extraction
- **Production**: Scalable document processing pipelines
- **Experimentation**: A/B testing with large datasets
- **Prototyping**: Rapid development of production-ready solutions
