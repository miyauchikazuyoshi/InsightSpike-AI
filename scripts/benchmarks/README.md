# InsightSpike Benchmarks

Performance benchmarking suite for InsightSpike algorithms and operations.

## 📁 Benchmark Categories

### 1. Algorithm Benchmarks (`performance_suite.py`)
- Tests computational complexity (O(n²), O(n³))
- Measures GED and IG algorithm scaling
- Suitable for CI/CD performance regression testing

### 2. Production Benchmarks (`production/`)
- Real-world performance testing
- DataStore and Agent throughput
- Memory usage and scalability
- Database efficiency metrics

## 🚀 Running Benchmarks

### Algorithm Performance
```bash
poetry run python scripts/benchmarks/performance_suite.py
```

### Production Performance
```bash
# Comprehensive production benchmark
poetry run python scripts/benchmarks/production/production_benchmark.py

# Lightweight DataStore benchmark
poetry run python scripts/benchmarks/production/simple_production_benchmark.py
```

## 📊 Results

Results are saved in:
- Algorithm benchmarks: `scripts/benchmarks/results/`
- Production benchmarks: `scripts/benchmarks/production/results/`