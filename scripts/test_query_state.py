#!/usr/bin/env python3
"""
Test Query State directly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insightspike.core.query_transformation import QueryState, QueryTransformationHistory
import torch

# Test QueryState
print("Testing QueryState...")

# Create initial state
state1 = QueryState(
    text="What is entropy?",
    embedding=torch.randn(384),
    stage="initial"
)

print(f"Initial state:")
print(f"  Text: {state1.text}")
print(f"  Stage: {state1.stage}")
print(f"  Color: {state1.color}")
print(f"  Confidence: {state1.confidence}")

# Add some concepts
state1.absorb_concept("thermodynamics")
state1.absorb_concept("information theory")

print(f"\nAfter absorbing concepts:")
print(f"  Absorbed: {state1.absorbed_concepts}")
print(f"  Confidence: {state1.confidence}")
print(f"  Color: {state1.color}")

# Add insight
state1.add_insight("Entropy appears in both physics and information theory")

print(f"\nAfter insight:")
print(f"  Insights: {state1.insights}")
print(f"  Confidence: {state1.confidence}")
print(f"  Color: {state1.color}")

# Test transformation history
print("\n\nTesting QueryTransformationHistory...")

history = QueryTransformationHistory("What is entropy?")
history.add_state(state1)

# Create evolved state
state2 = QueryState(
    text=state1.text,
    embedding=state1.embedding + torch.randn(384) * 0.1,
    stage="transforming",
    confidence=0.6,
    absorbed_concepts=state1.absorbed_concepts.copy(),
    insights=state1.insights.copy()
)
state2.add_insight("Both entropies measure disorder/uncertainty")
history.add_state(state2)

# Final state
state3 = QueryState(
    text=state1.text,
    embedding=state2.embedding + torch.randn(384) * 0.1,
    stage="insight",
    confidence=0.8,
    absorbed_concepts=state2.absorbed_concepts.copy(),
    insights=state2.insights.copy()
)
state3.add_insight("S = k log W connects both concepts!")
history.add_state(state3)

print(f"Transformation path: {history.get_transformation_path()}")
print(f"Total insights: {history.get_total_insights()}")
print(f"Confidence trajectory: {history.get_confidence_trajectory()}")
print(f"Reached insight? {history.reached_insight()}")

print("\n✅ Query transformation components working!")