# Adaptive Loop Clean Architecture Design

**Date**: 2025-01-25  
**Status**: Design Phase  
**Goal**: Implement adaptive loop without creating spaghetti code

## Architecture Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Dependency Injection**: Components don't create their dependencies
3. **Interface Segregation**: Small, focused interfaces
4. **Open/Closed**: Extend behavior without modifying existing code

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MainAgent                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AdaptiveProcessor                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │   │
│  │  │ Exploration│  │   TopK     │  │   Pattern    │ │   │
│  │  │ Strategy   │  │ Calculator │  │   Learner    │ │   │
│  │  └────────────┘  └────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ExplorationLoop                         │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────────────┐ │   │
│  │  │  L1  │→ │  L2  │→ │  L3  │→ │ Spike Detector │ │   │
│  │  └──────┘  └──────┘  └──────┘  └────────────────┘ │   │
│  │      ↑                              ↓ (no spike)     │   │
│  │      └──────────────────────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓ (spike detected)                  │
│                    ┌─────────┐                              │
│                    │   L4    │                              │
│                    │  (LLM)  │                              │
│                    └─────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/insightspike/
├── adaptive/                      # New adaptive processing module
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── adaptive_processor.py  # Main adaptive processor
│   │   ├── exploration_loop.py   # L1-L2-L3 loop manager
│   │   └── interfaces.py          # Clean interfaces
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract strategy
│   │   ├── narrowing.py          # Narrowing strategy
│   │   ├── expanding.py          # Expanding strategy
│   │   └── alternating.py        # Alternating strategy
│   │
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── adaptive_topk.py      # Ported from old implementation
│   │   └── chain_reaction.py     # Chain reaction potential
│   │
│   └── learning/
│       ├── __init__.py
│       ├── pattern_tracker.py    # Track successful patterns
│       └── temperature_manager.py # Manage exploration temperature
│
└── implementations/
    └── agents/
        └── main_agent.py         # Modified to use adaptive processor

```

## Clean Interfaces

### 1. Core Interfaces

```python
# adaptive/core/interfaces.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ExplorationParams:
    """Parameters for exploration attempt"""
    radius: float
    topk_l1: int
    topk_l2: int
    topk_l3: int
    temperature: float
    
@dataclass
class ExplorationResult:
    """Result of single exploration attempt"""
    spike_detected: bool
    confidence: float
    retrieved_docs: List[Dict]
    graph_analysis: Dict
    metrics: Dict

class ExplorationStrategy(ABC):
    """Abstract base for exploration strategies"""
    
    @abstractmethod
    def get_initial_params(self) -> ExplorationParams:
        """Get initial exploration parameters"""
        pass
    
    @abstractmethod
    def adjust_params(self, attempt: int, prev_result: ExplorationResult) -> ExplorationParams:
        """Adjust parameters based on previous result"""
        pass
    
    @abstractmethod
    def should_continue(self, results: List[ExplorationResult]) -> bool:
        """Decide whether to continue exploration"""
        pass

class TopKCalculator(ABC):
    """Abstract base for topK calculation"""
    
    @abstractmethod
    def calculate(self, l1_analysis: Dict) -> Dict[str, int]:
        """Calculate adaptive topK values"""
        pass

class PatternLearner(ABC):
    """Abstract base for pattern learning"""
    
    @abstractmethod
    def record_success(self, query: str, exploration_path: List[ExplorationParams], result: Dict):
        """Record successful exploration"""
        pass
    
    @abstractmethod
    def suggest_params(self, query: str) -> Optional[ExplorationParams]:
        """Suggest parameters based on learned patterns"""
        pass
```

### 2. Adaptive Processor

```python
# adaptive/core/adaptive_processor.py

class AdaptiveProcessor:
    """Main adaptive processing coordinator"""
    
    def __init__(
        self,
        exploration_loop: 'ExplorationLoop',
        strategy: ExplorationStrategy,
        topk_calculator: TopKCalculator,
        pattern_learner: Optional[PatternLearner] = None
    ):
        self.exploration_loop = exploration_loop
        self.strategy = strategy
        self.topk_calculator = topk_calculator
        self.pattern_learner = pattern_learner
        
    def process(self, question: str, max_attempts: int = 5) -> Dict:
        """Process question with adaptive exploration"""
        
        # Check learned patterns
        if self.pattern_learner:
            suggested_params = self.pattern_learner.suggest_params(question)
            if suggested_params:
                # Try learned parameters first
                result = self.exploration_loop.explore_once(question, suggested_params)
                if result.spike_detected:
                    return self._generate_response(question, result)
        
        # Run adaptive exploration
        exploration_results = []
        params = self.strategy.get_initial_params()
        
        for attempt in range(max_attempts):
            # Single exploration attempt
            result = self.exploration_loop.explore_once(question, params)
            exploration_results.append(result)
            
            # Check for spike
            if result.spike_detected:
                # Record success
                if self.pattern_learner:
                    self.pattern_learner.record_success(
                        question, 
                        [r.params for r in exploration_results],
                        result
                    )
                # Generate response with LLM
                return self._generate_response(question, result)
            
            # Check if should continue
            if not self.strategy.should_continue(exploration_results):
                break
                
            # Adjust parameters
            params = self.strategy.adjust_params(attempt + 1, result)
        
        # No spike found - use best result
        best_result = max(exploration_results, key=lambda r: r.confidence)
        return self._generate_response(question, best_result)
```

### 3. Exploration Loop

```python
# adaptive/core/exploration_loop.py

class ExplorationLoop:
    """Manages L1-L2-L3 exploration loop"""
    
    def __init__(self, l1_monitor, l2_memory, l3_graph):
        self.l1_monitor = l1_monitor
        self.l2_memory = l2_memory
        self.l3_graph = l3_graph
        
    def explore_once(self, question: str, params: ExplorationParams) -> ExplorationResult:
        """Single L1-L2-L3 exploration attempt"""
        
        # L1: Analyze with exploration radius
        l1_analysis = self.l1_monitor.analyze_uncertainty(
            question,
            exploration_radius=params.radius
        )
        
        # L2: Retrieve with adaptive topK
        retrieved_docs = self.l2_memory.retrieve(
            question,
            k=params.topk_l2,
            similarity_threshold=params.radius
        )
        
        # L3: Graph analysis
        if retrieved_docs:
            graph_analysis = self.l3_graph.analyze_documents(
                retrieved_docs,
                context={'l1_analysis': l1_analysis}
            )
            
            spike_detected = self._check_spike(graph_analysis)
            confidence = graph_analysis.get('reasoning_quality', 0.0)
        else:
            graph_analysis = {}
            spike_detected = False
            confidence = 0.0
            
        return ExplorationResult(
            spike_detected=spike_detected,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            graph_analysis=graph_analysis,
            metrics={
                'l1_uncertainty': l1_analysis.get('uncertainty', 1.0),
                'l2_retrieval_count': len(retrieved_docs),
                'l3_ged': graph_analysis.get('metrics', {}).get('delta_ged', 0),
                'l3_ig': graph_analysis.get('metrics', {}).get('delta_ig', 0)
            }
        )
```

## Integration with MainAgent

```python
# implementations/agents/main_agent.py

class MainAgent:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Initialize adaptive processor if enabled
        if config.processing.enable_adaptive_loop:
            self._init_adaptive_processor(config.processing.adaptive_loop)
    
    def _init_adaptive_processor(self, adaptive_config):
        """Initialize adaptive processing components"""
        
        # Create exploration loop
        exploration_loop = ExplorationLoop(
            self.l1_error_monitor,
            self.l2_memory,
            self.l3_graph
        )
        
        # Create strategy
        strategy_class = {
            'narrowing': NarrowingStrategy,
            'expanding': ExpandingStrategy,
            'alternating': AlternatingStrategy
        }[adaptive_config.exploration_strategy]
        
        strategy = strategy_class(
            initial_radius=adaptive_config.initial_exploration_radius,
            decay_factor=adaptive_config.radius_decay_factor
        )
        
        # Create calculators
        topk_calculator = AdaptiveTopKCalculator()
        
        # Create pattern learner if enabled
        pattern_learner = None
        if adaptive_config.enable_pattern_learning:
            pattern_learner = PatternTracker()
        
        # Create processor
        self.adaptive_processor = AdaptiveProcessor(
            exploration_loop,
            strategy,
            topk_calculator,
            pattern_learner
        )
    
    def process_question(self, question: str, **kwargs):
        """Process question with optional adaptive loop"""
        
        if self.config.processing.enable_adaptive_loop:
            return self.adaptive_processor.process(question, **kwargs)
        else:
            return self._process_standard(question, **kwargs)
```

## Configuration Schema

```python
# config/models.py

class AdaptiveLoopConfig(BaseModel):
    """Configuration for adaptive loop processing"""
    
    enable_adaptive_loop: bool = Field(default=False)
    max_exploration_attempts: int = Field(default=5, ge=1, le=10)
    initial_exploration_radius: float = Field(default=0.7, ge=0.1, le=1.0)
    radius_decay_factor: float = Field(default=0.8, ge=0.5, le=0.95)
    exploration_strategy: Literal["narrowing", "expanding", "alternating"] = "narrowing"
    
    # Pattern learning
    enable_pattern_learning: bool = Field(default=True)
    pattern_db_path: Optional[str] = None
    
    # Temperature control
    initial_temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    temperature_decay: float = Field(default=0.95, ge=0.8, le=1.0)
    
    # Spike detection overrides
    spike_ged_threshold_override: Optional[float] = None
    spike_ig_threshold_override: Optional[float] = None
```

## Implementation Phases

### Phase 1: Core Components (Week 1)
1. [ ] Create module structure
2. [ ] Implement interfaces
3. [ ] Port AdaptiveTopKCalculator
4. [ ] Implement ExplorationLoop
5. [ ] Create basic strategies

### Phase 2: Integration (Week 2)
1. [ ] Implement AdaptiveProcessor
2. [ ] Modify MainAgent
3. [ ] Update configuration
4. [ ] Create unit tests
5. [ ] Integration tests

### Phase 3: Learning & Optimization (Week 3)
1. [ ] Implement PatternTracker
2. [ ] Add temperature management
3. [ ] Performance benchmarks
4. [ ] Documentation
5. [ ] Example notebooks

## Benefits of This Architecture

1. **Modularity**: Each component has clear boundaries
2. **Testability**: Easy to unit test each component
3. **Extensibility**: New strategies can be added without modifying core
4. **Maintainability**: Clear separation of concerns
5. **Performance**: Clean interfaces enable optimization

## Anti-Patterns to Avoid

1. ❌ Direct layer manipulation from MainAgent
2. ❌ Circular dependencies between modules  
3. ❌ Hardcoded strategies in core logic
4. ❌ Mixing learning logic with exploration
5. ❌ Tight coupling to specific LLM providers

## Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Component interactions
3. **Benchmark Tests**: Performance comparison
4. **Regression Tests**: Ensure existing functionality

This clean architecture ensures the adaptive loop implementation remains maintainable and extensible.