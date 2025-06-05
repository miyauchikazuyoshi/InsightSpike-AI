# InsightSpike-AI API Reference

This document provides comprehensive API reference for all modules and classes in InsightSpike-AI.

## Table of Contents

- [Core Modules](#core-modules)
  - [InsightDetector](#insightdetector)
  - [GraphEditDistance](#grapheditdistance)
  - [InformationGain](#informationgain)
- [Detection Module](#detection-module)
- [Metrics Module](#metrics-module)
- [Processing Module](#processing-module)
- [Utilities](#utilities)

---

## Core Modules

### InsightDetector

**Location**: `src/insightspike/core/insight_detector.py`

The main class for detecting insights in educational and research contexts.

#### Class Definition

```python
class InsightDetector:
    """
    Core insight detection engine that combines graph-based analysis 
    with information-theoretic measures.
    
    This class integrates Graph Edit Distance (GED) and Information Gain (IG)
    calculations to identify moments of conceptual understanding in learning
    processes.
    """
```

#### Constructor

```python
def __init__(self, 
             config: Optional[Dict[str, Any]] = None,
             ged_calculator: Optional[GraphEditDistance] = None,
             ig_calculator: Optional[InformationGain] = None):
    """
    Initialize the InsightDetector.
    
    Args:
        config: Configuration dictionary for detection parameters
        ged_calculator: Custom GED calculator instance
        ig_calculator: Custom IG calculator instance
    """
```

#### Methods

##### `detect_insights(query: str, context: str) -> Dict[str, Any]`

Primary method for insight detection.

**Parameters:**
- `query` (str): The question or problem statement
- `context` (str): Additional context or learning material

**Returns:**
- `Dict[str, Any]`: Detection results containing:
  - `ged_score`: Graph Edit Distance score
  - `ig_score`: Information Gain score
  - `insight_detected`: Boolean indicating insight presence
  - `confidence`: Confidence score (0.0 to 1.0)
  - `metadata`: Additional analysis data

**Example:**
```python
from insightspike.core.insight_detector import InsightDetector

detector = InsightDetector()
result = detector.detect_insights(
    query="What is the Monty Hall problem?",
    context="probability_theory"
)

print(f"Insight detected: {result['insight_detected']}")
print(f"Confidence: {result['confidence']:.2f}")
```

##### `analyze_learning_progression(data_sequence: List[Dict]) -> Dict[str, Any]`

Analyze a sequence of learning interactions.

**Parameters:**
- `data_sequence` (List[Dict]): Sequence of learning data points

**Returns:**
- `Dict[str, Any]`: Progression analysis including trend detection and insight moments

---

### GraphEditDistance

**Location**: `src/insightspike/algorithms/graph_edit_distance.py`

Implements graph edit distance calculation for conceptual graph comparison.

#### Class Definition

```python
class GraphEditDistance:
    """
    Graph Edit Distance calculator for measuring conceptual changes.
    
    This implementation focuses on educational concept graphs where nodes
    represent concepts and edges represent relationships between concepts.
    """
```

#### Constructor

```python
def __init__(self, 
             node_cost: float = 1.0,
             edge_cost: float = 1.0,
             optimization_level: str = "standard"):
    """
    Initialize GED calculator with cost parameters.
    
    Args:
        node_cost: Cost of node insertion/deletion operations
        edge_cost: Cost of edge insertion/deletion operations
        optimization_level: "fast", "standard", or "precise"
    """
```

#### Methods

##### `calculate_distance(graph1: nx.Graph, graph2: nx.Graph) -> float`

Calculate the edit distance between two graphs.

**Parameters:**
- `graph1` (networkx.Graph): Source graph
- `graph2` (networkx.Graph): Target graph

**Returns:**
- `float`: Edit distance value (lower indicates more similarity)

**Time Complexity:** O(n³) where n is the number of nodes

**Example:**
```python
import networkx as nx
from insightspike.algorithms.graph_edit_distance import GraphEditDistance

# Create test graphs
G1 = nx.Graph()
G1.add_edges_from([('A', 'B'), ('B', 'C')])

G2 = nx.Graph()
G2.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

# Calculate distance
ged = GraphEditDistance()
distance = ged.calculate_distance(G1, G2)
print(f"Edit distance: {distance}")
```

##### `calculate_normalized_distance(graph1: nx.Graph, graph2: nx.Graph) -> float`

Calculate normalized edit distance (0.0 to 1.0 scale).

**Returns:**
- `float`: Normalized distance where 0.0 = identical, 1.0 = completely different

---

### InformationGain

**Location**: `src/insightspike/algorithms/information_gain.py`

Implements information gain calculation for measuring learning progress.

#### Class Definition

```python
class InformationGain:
    """
    Information Gain calculator for educational insight detection.
    
    Measures the reduction in entropy when moving from one state of
    understanding to another, indicating learning progress.
    """
```

#### Methods

##### `calculate_gain(before_state: np.ndarray, after_state: np.ndarray) -> float`

Calculate information gain between two knowledge states.

**Parameters:**
- `before_state` (np.ndarray): Feature vector representing initial state
- `after_state` (np.ndarray): Feature vector representing final state

**Returns:**
- `float`: Information gain value (higher indicates more learning)

**Example:**
```python
import numpy as np
from insightspike.algorithms.information_gain import InformationGain

# Example knowledge states
before = np.array([0.2, 0.3, 0.1, 0.4])  # Initial understanding
after = np.array([0.1, 0.1, 0.7, 0.1])   # After learning

ig = InformationGain()
gain = ig.calculate_gain(before, after)
print(f"Information gain: {gain:.3f}")
```

##### `calculate_entropy(probabilities: np.ndarray) -> float`

Calculate Shannon entropy of a probability distribution.

**Parameters:**
- `probabilities` (np.ndarray): Probability distribution (must sum to 1.0)

**Returns:**
- `float`: Entropy value in bits

---

## Detection Module

### InsightClassifier

**Location**: `src/insightspike/detection/classifier.py`

Machine learning classifier for insight detection patterns.

#### Methods

##### `train(training_data: List[Dict], labels: List[bool]) -> None`

Train the classifier on labeled insight data.

##### `predict(features: np.ndarray) -> Tuple[bool, float]`

Predict insight presence and confidence.

**Returns:**
- `Tuple[bool, float]`: (insight_detected, confidence_score)

---

## Metrics Module

### PerformanceMetrics

**Location**: `src/insightspike/metrics/performance.py`

Performance evaluation and benchmarking utilities.

#### Methods

##### `measure_response_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]`

Measure function execution time.

**Returns:**
- `Tuple[Any, float]`: (function_result, execution_time_seconds)

##### `evaluate_accuracy(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]`

Calculate classification metrics.

**Returns:**
- `Dict[str, float]`: Dictionary containing precision, recall, f1_score, accuracy

---

## Processing Module

### DataPreprocessor

**Location**: `src/insightspike/processing/preprocessor.py`

Data preprocessing utilities for insight detection.

#### Methods

##### `preprocess_text(text: str) -> List[str]`

Preprocess text input for analysis.

##### `extract_features(text: str, context: str) -> np.ndarray`

Extract feature vector from text and context.

##### `build_concept_graph(text: str) -> nx.Graph`

Build concept graph from text input.

---

## Utilities

### Configuration

**Location**: `src/insightspike/utils/config.py`

Configuration management utilities.

#### Functions

##### `load_config(config_path: str) -> Dict[str, Any]`

Load configuration from file.

##### `validate_config(config: Dict[str, Any]) -> bool`

Validate configuration parameters.

### Logging

**Location**: `src/insightspike/utils/logging.py`

Logging utilities for debugging and monitoring.

#### Functions

##### `setup_logger(name: str, level: str = "INFO") -> logging.Logger`

Set up structured logging.

##### `log_performance(func_name: str, execution_time: float) -> None`

Log performance metrics.

---

## Usage Examples

### Basic Insight Detection

```python
from insightspike import InsightDetector

# Initialize detector
detector = InsightDetector()

# Detect insights in a learning scenario
result = detector.detect_insights(
    query="How does backpropagation work in neural networks?",
    context="machine_learning_course"
)

# Check results
if result['insight_detected']:
    print(f"Insight detected with {result['confidence']:.1%} confidence")
    print(f"GED score: {result['ged_score']:.3f}")
    print(f"IG score: {result['ig_score']:.3f}")
```

### Advanced Configuration

```python
from insightspike import InsightDetector
from insightspike.algorithms import GraphEditDistance, InformationGain

# Custom algorithm configuration
ged_config = GraphEditDistance(
    node_cost=0.8,
    edge_cost=1.2,
    optimization_level="precise"
)

ig_config = InformationGain()

# Custom detector configuration
config = {
    'ged_threshold': -0.5,
    'ig_threshold': 1.5,
    'confidence_method': 'bayesian'
}

detector = InsightDetector(
    config=config,
    ged_calculator=ged_config,
    ig_calculator=ig_config
)
```

### Batch Processing

```python
from insightspike import InsightDetector

detector = InsightDetector()

# Process multiple queries
queries = [
    "What is the derivative of x²?",
    "How do photosynthesis and cellular respiration relate?",
    "What causes economic inflation?"
]

results = []
for query in queries:
    result = detector.detect_insights(query, "educational_assessment")
    results.append(result)

# Analyze results
insights_found = sum(1 for r in results if r['insight_detected'])
print(f"Insights detected in {insights_found}/{len(queries)} queries")
```

---

## Error Handling

### Common Exceptions

#### `InsightDetectionError`

Raised when insight detection fails due to invalid input or processing errors.

```python
try:
    result = detector.detect_insights("", "")  # Empty inputs
except InsightDetectionError as e:
    print(f"Detection failed: {e}")
```

#### `GraphProcessingError`

Raised when graph operations fail.

#### `ConfigurationError`

Raised when configuration parameters are invalid.

---

## Performance Considerations

### Computational Complexity

- **Graph Edit Distance**: O(n³) for n nodes
- **Information Gain**: O(n log n) for n samples
- **Overall Detection**: O(n³) dominated by GED calculation

### Memory Usage

- **Graph Storage**: O(n²) for dense graphs
- **Feature Vectors**: O(d) for d dimensions
- **Result Caching**: Configurable cache size

### Optimization Tips

1. **Use smaller graphs** for real-time applications
2. **Enable caching** for repeated queries
3. **Choose appropriate optimization levels** based on accuracy/speed trade-offs
4. **Batch process** multiple queries when possible

---

## Version Compatibility

This API reference is for InsightSpike-AI version 0.7.0 and later.

### Breaking Changes from Previous Versions

- v0.6.x: `detect_insights()` return format changed
- v0.5.x: Configuration parameter names updated

### Deprecation Warnings

- `legacy_detect()` method will be removed in v0.8.0
- Old configuration format supported until v0.9.0

---

For more examples and tutorials, see the [Quick Start Guide](../guides/QUICK_START.md) and the [examples directory](../../examples/).
