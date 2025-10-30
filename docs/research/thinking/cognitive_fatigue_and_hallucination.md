# Cognitive Fatigue and Hallucination Model

*Created: 2025-07-28*

## 1. Core Analogy

This document models the phenomenon of "hallucinations from sleep deprivation" within the `InsightSpike-AI` framework.

- **Sleep Deprivation**: The state where the `KnowledgeGraphRefactoringSystem` (the "Sleep Cycle") has not been run for an extended period.
- **Edge Overflow**: Without the pruning and consolidation that occurs during sleep, the knowledge graph becomes overly dense and noisy, decreasing its signal-to-noise ratio.
- **Hallucinations**: When `geDIG` calculations are performed on this noisy graph, random connections can be misinterpreted as meaningful patterns, leading to the generation of "false positive" insight spikes.

## 2. Architectural Implementation

We propose a `CognitiveFatigueManager` to monitor the system's "mental health" and adjust its reasoning capabilities accordingly.

### 2.1 Cognitive Fatigue State

The `MainAgent` will maintain a `fatigue_level` state variable (0.0 to 1.0).

- **Increases**: With every new knowledge addition or complex query.
- **Decreases**: Drastically after a `perform_background_maintenance` (sleep cycle) is completed.

### 2.2 Implementation Sketch

```python
class CognitiveFatigueManager:
    """Monitors and manages the agent's cognitive fatigue."""

    def __init__(self, max_fatigue: float = 100.0, recovery_rate: float = 0.9):
        self.fatigue_level = 0.0
        self.max_fatigue = max_fatigue
        self.recovery_rate = recovery_rate

    def record_activity(self, complexity: float = 1.0):
        """Increase fatigue based on cognitive load."""
        self.fatigue_level = min(self.max_fatigue, self.fatigue_level + complexity)
        print(f"DEBUG: Fatigue increased to {self.fatigue_level:.2f}")

    def record_sleep_cycle(self):
        """Drastically reduce fatigue after a sleep/refactoring cycle."""
        self.fatigue_level *= (1 - self.recovery_rate)
        print(f"DEBUG: Fatigue recovered to {self.fatigue_level:.2f} after sleep.")

    def get_current_reliability_factor(self) -> float:
        """
        Returns a factor from 1.0 (fully rested) to 0.0 (fully exhausted).
        This factor can be used to modulate the confidence of generated insights.
        """
        return 1.0 - (self.fatigue_level / self.max_fatigue)


# In MainAgent...
class MainAgent:
    def __init__(self, ...):
        self.fatigue_manager = CognitiveFatigueManager()

    def add_knowledge(self, text: str):
        # ... existing logic ...
        self.fatigue_manager.record_activity(complexity=0.5)

    def process_question(self, question: str) -> 'CycleResult':
        # ... existing logic ...
        self.fatigue_manager.record_activity(complexity=1.0)

        # Modulate insight confidence based on fatigue
        reliability = self.fatigue_manager.get_current_reliability_factor()
        if result.has_spike:
            result.confidence *= reliability
            print(f"INFO: Insight confidence adjusted by fatigue. Reliability: {reliability:.2f}")
        
        return result
```

## 3. Implications

This model introduces a concept of "AI mental health." A system that is overworked without proper "rest" (optimization and refactoring) will become less reliable and more prone to "hallucinations." This provides a strong, built-in incentive to run the maintenance cycles, ensuring the long-term health and reliability of the knowledge graph.

It elegantly explains why simply accumulating information is not enough; consolidation and forgetting are crucial components of true intelligence.