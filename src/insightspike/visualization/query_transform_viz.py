"""Lightweight visualizer for Query Transformation.

Keeps dependencies optional and falls back to printing/snapshots when
matplotlib is unavailable. Intended as a minimal Phase 4 starting point.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def animate_transformation(transformation_history: Iterable[Dict[str, Any]], *, show: bool = False) -> None:
    """Animate the transformation process if matplotlib is available.

    Parameters:
        transformation_history: Iterable of state dicts (or objects with to_dict()).
        show: Whether to call plt.show() at the end when matplotlib is present.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore

        magnitudes = []
        confidences = []
        steps = []
        for i, s in enumerate(transformation_history):
            sd = s.to_dict() if hasattr(s, "to_dict") else s
            steps.append(i)
            magnitudes.append(sd.get("transformation_magnitude", 0.0))
            confidences.append(sd.get("confidence", 0.0))

        plt.figure(figsize=(6, 3))
        plt.plot(steps, magnitudes, label="|Δquery|", marker="o")
        plt.plot(steps, confidences, label="confidence", marker="x")
        plt.title("Query Transformation Progress")
        plt.xlabel("step")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
    except Exception:
        # Fallback: print a brief summary only
        _print_summary(transformation_history)


def snapshot(transformation_history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a serializable snapshot with simple progress metrics."""
    steps = 0
    last_conf = 0.0
    last_mag = 0.0
    insights = 0
    for steps, s in enumerate(transformation_history, start=1):
        sd = s.to_dict() if hasattr(s, "to_dict") else s
        last_conf = float(sd.get("confidence", 0.0) or 0.0)
        last_mag = float(sd.get("transformation_magnitude", 0.0) or 0.0)
        if sd.get("insights"):
            try:
                insights += len(sd["insights"])  # type: ignore[index]
            except Exception:
                insights += 1
    return {
        "steps": steps,
        "last_confidence": last_conf,
        "last_magnitude": last_mag,
        "total_insights": insights,
    }


def _print_summary(transformation_history: Iterable[Dict[str, Any]]) -> None:
    snap = snapshot(transformation_history)
    print(
        f"[QueryTransformViz] steps={snap['steps']} last_conf={snap['last_confidence']:.3f} "
        f"last_mag={snap['last_magnitude']:.3f} insights={snap['total_insights']}"
    )


__all__ = ["animate_transformation", "snapshot"]

