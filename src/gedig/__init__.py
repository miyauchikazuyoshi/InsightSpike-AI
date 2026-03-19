"""geDIG Unified Core — Generalized Differential Information Gain.

Provides a common F-eval framework across maze, RAG, and transformer experiments.

F = ΔEPC - λ(ΔH + γΔB)

Components:
- EPC: Graph edit cost (structural change)
- H: Entropy (information ordering)
- B: Structure potential (SP path efficiency or β₁ Betti number)
"""

__version__ = "0.1.0"
