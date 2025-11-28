"""Allow `python -m insightspike.implementations.layers.layer3` for a quick stub check."""

from . import L3GraphReasoner

if __name__ == "__main__":  # pragma: no cover
    inst = L3GraphReasoner()
    print(f"L3GraphReasoner created (enabled={getattr(inst, 'enabled', True)})")
