# Insight Discovery CLI Features Plan

## Overview

This plan details the development of InsightSpike's unique CLI features that leverage its core geDIG (Graph Edit Distance + Information Gain) algorithm for insight discovery, concept bridging, and knowledge graph analysis.

## Goals

1. Expose InsightSpike's unique insight discovery capabilities through intuitive CLI commands
2. Differentiate from standard RAG systems by focusing on emergent knowledge discovery
3. Enable researchers to find hidden connections in their knowledge bases
4. Provide visual and analytical tools for understanding knowledge structures

## Core Features

### 1. Insight Discovery Command

**Command: `spike discover`**

Analyze a corpus to find unexpected connections and insights using spike detection.

```bash
# Basic discovery
spike discover --corpus docs/ --min-spike 0.7

# Advanced options
spike discover \
  --corpus research-papers/ \
  --min-spike 0.8 \
  --max-insights 20 \
  --categories "causal,structural,analogical" \
  --export insights.json
```

**Output Format:**
```
🔍 Discovering insights in 847 documents...

⚡ High-value insights discovered:

1. [Spike: 0.92] Unexpected connection:
   "Quantum entanglement" ←→ "Neural synchronization"
   Bridge concepts: information transfer, non-locality, coherence
   Confidence: 87%

2. [Spike: 0.85] Emergent pattern:
   "Market volatility" exhibits similar dynamics to "Ecosystem collapse"
   Common factors: cascade effects, tipping points, resilience
   Confidence: 78%

📊 Summary:
- Documents analyzed: 847
- Insights discovered: 12
- Average spike value: 0.81
- Processing time: 3.2s
```

**Implementation Details:**
```python
class InsightDiscoveryCLI:
    def discover_insights(self, corpus_path: str, min_spike: float):
        # 1. Load and process corpus
        documents = self.load_corpus(corpus_path)
        
        # 2. Build knowledge graph
        graph = self.build_graph(documents)
        
        # 3. Detect insight spikes
        insights = []
        for edge in graph.analyze_edges():
            if edge.spike_value > min_spike:
                insight = self.extract_insight(edge)
                insights.append(insight)
        
        # 4. Rank and filter insights
        return self.rank_insights(insights)
```

### 2. Concept Bridging Explorer

**Command: `spike bridge`**

Find conceptual paths between seemingly unrelated ideas.

```bash
# Find bridges between concepts
spike bridge "machine learning" "biological evolution"

# Multi-hop bridging with constraints
spike bridge "quantum computing" "consciousness" \
  --max-hops 4 \
  --min-confidence 0.6 \
  --avoid "philosophy"
```

**Output Format:**
```
🌉 Finding bridges between "machine learning" and "biological evolution"...

Path 1 (Confidence: 85%):
  machine learning
    ↓ [optimization algorithms]
  genetic algorithms
    ↓ [population-based methods]
  natural selection
    ↓ [adaptation mechanisms]
  biological evolution

Path 2 (Confidence: 72%):
  machine learning
    ↓ [pattern recognition]
  neural networks
    ↓ [biological inspiration]
  brain evolution
    ↓ [cognitive development]
  biological evolution

💡 Discovered bridge concepts:
- "Adaptive algorithms" (appears in 3 paths)
- "Fitness functions" (high centrality)
- "Emergent complexity" (novel connection)
```

**Implementation:**
```python
class ConceptBridgeExplorer:
    def find_bridges(self, concept1: str, concept2: str, max_hops: int):
        # Use modified Dijkstra with semantic distances
        paths = self.semantic_pathfinding(
            self.graph, 
            concept1, 
            concept2,
            max_hops
        )
        
        # Identify bridge concepts
        bridge_concepts = self.extract_bridge_concepts(paths)
        
        # Calculate path confidence
        for path in paths:
            path.confidence = self.calculate_path_confidence(path)
        
        return paths, bridge_concepts
```

### 3. Knowledge Graph Analytics

**Command: `spike graph`**

Analyze and visualize the knowledge structure.

```bash
# Analyze graph structure
spike graph analyze --metrics all

# Find influential concepts
spike graph centrality --top 20 --method "betweenness"

# Detect knowledge clusters
spike graph clusters --algorithm louvain --min-size 5

# Interactive visualization
spike graph visualize --output graph.html --layout force
```

**Analytics Output:**
```
📊 Knowledge Graph Analysis

Structure Metrics:
- Nodes (concepts): 3,847
- Edges (connections): 12,439
- Average degree: 6.46
- Clustering coefficient: 0.342
- Graph density: 0.0017

Insight Metrics:
- Total insights discovered: 156
- Average spike value: 0.73
- High-spike edges (>0.8): 34
- Insight categories:
  * Causal: 45 (28.8%)
  * Structural: 67 (42.9%)
  * Analogical: 31 (19.9%)
  * Synthetic: 13 (8.3%)

Top Central Concepts:
1. "Information" (betweenness: 0.89)
2. "Energy" (betweenness: 0.84)
3. "System" (betweenness: 0.81)
```

### 4. Cumulative Learning Sessions

**Command: `spike session`**

Maintain context across multiple interactions for deeper exploration.

```bash
# Start a research session
spike session start "quantum-biology-research"

# Continue exploration with context
spike query "How do quantum effects influence photosynthesis?"
spike feedback "Focus more on coherence"
spike explore-deeper

# Save session state
spike session save

# Resume later
spike session resume "quantum-biology-research"
spike session history --last 10
```

**Session Features:**
```python
class ResearchSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.query_history = []
        self.discovered_insights = []
        self.exploration_paths = []
        self.feedback_adjustments = []
    
    def process_with_context(self, query: str):
        # Use previous queries to bias search
        context_vector = self.compute_session_context()
        
        # Adjust retrieval based on feedback
        retrieval_params = self.adapt_from_feedback()
        
        # Track exploration trajectory
        self.update_exploration_path(query)
        
        return enhanced_response
```

### 5. Insight Monitoring & Alerts

**Command: `spike monitor`**

Track emergence of new insights over time.

```bash
# Monitor for new insights
spike monitor start \
  --corpus papers/ \
  --check-interval 1h \
  --notify-webhook https://slack.webhook

# Set up insight alerts
spike monitor alert \
  --condition "spike > 0.9" \
  --keywords "quantum,consciousness" \
  --email researcher@univ.edu
```

## Implementation Timeline

### Phase 1: Core Discovery (Weeks 1-3)
- [ ] Implement `spike discover` command
- [ ] Create insight ranking algorithm
- [ ] Design output formatters
- [ ] Add JSON/CSV export options

### Phase 2: Bridging & Pathfinding (Weeks 4-6)
- [ ] Implement semantic pathfinding
- [ ] Create bridge concept detection
- [ ] Add path confidence scoring
- [ ] Build interactive path visualization

### Phase 3: Graph Analytics (Weeks 7-9)
- [ ] Implement graph metrics calculation
- [ ] Add centrality algorithms
- [ ] Create clustering detection
- [ ] Build D3.js visualization export

### Phase 4: Advanced Features (Weeks 10-12)
- [ ] Implement session management
- [ ] Add monitoring capabilities
- [ ] Create feedback loop system
- [ ] Build alert mechanisms

## Technical Requirements

### Algorithm Enhancements

1. **Spike Detection Refinement**
```python
def calculate_spike_value(self, edge, context):
    # Base geDIG calculation
    base_spike = self.calculate_gedig(edge)
    
    # Context amplification
    context_factor = self.analyze_context_relevance(edge, context)
    
    # Novelty bonus
    novelty = self.calculate_novelty(edge)
    
    return base_spike * context_factor * (1 + novelty)
```

2. **Semantic Pathfinding**
```python
def semantic_dijkstra(self, start, end, max_hops):
    # Modified Dijkstra using semantic distances
    # Instead of edge weights, use semantic similarity
    # Penalize common/trivial connections
    # Reward novel pathways
```

## User Experience Design

### Progressive Disclosure
```bash
# Simple mode (default)
spike discover

# Intermediate (common flags)
spike discover --min-spike 0.8 --categories causal

# Advanced (full control)
spike discover --corpus docs/ \
  --algorithm enhanced-gedig \
  --spike-threshold 0.75 \
  --context-window 5 \
  --parallel-workers 4 \
  --cache-embeddings
```

### Interactive Mode
```bash
spike interactive

InsightSpike> discover
Analyzing current knowledge base...
Found 12 potential insights. Show details? [Y/n]

InsightSpike> bridge "concept1" to "concept2"
Exploring conceptual space...
3 paths found. Visualize? [Y/n]
```

## Success Metrics

1. **Unique Value Proposition**
   - Users discover non-obvious connections
   - Insights lead to new research directions
   - Time to insight < 5 seconds for most queries

2. **Usability**
   - Commands intuitive without documentation
   - Output immediately actionable
   - Visual representations clear and informative

3. **Performance**
   - Process 10k documents in < 1 minute
   - Real-time graph updates
   - Interactive visualizations responsive

## Integration with Core System

```python
# CLI commands map to core functionality
class InsightSpikeCLI:
    def __init__(self):
        self.agent = MainAgent()
        self.graph_analyzer = GraphAnalyzer()
        self.insight_detector = InsightDetector()
    
    def execute_discover(self, args):
        # Load documents
        docs = self.load_corpus(args.corpus)
        
        # Use MainAgent for processing
        for doc in docs:
            self.agent.add_knowledge(doc)
        
        # Extract insights using core geDIG
        insights = self.insight_detector.detect_spikes(
            self.agent.get_knowledge_graph(),
            min_spike=args.min_spike
        )
        
        return self.format_insights(insights)
```

## Conclusion

These CLI features showcase InsightSpike's unique value proposition: going beyond simple retrieval to discover hidden connections and emergent insights. By focusing on these differentiating features, InsightSpike becomes an invaluable tool for researchers, analysts, and anyone seeking deeper understanding from their knowledge bases.