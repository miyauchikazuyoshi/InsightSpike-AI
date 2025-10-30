"""Minimal example for Query Transformation visualization.

This example avoids heavy dependencies and uses a synthetic transformation
history to demonstrate the visualization helper API.
"""

from insightspike.visualization.query_transform_viz import (
    animate_transformation,
    snapshot,
)


def main():
    history = [
        {"confidence": 0.2, "transformation_magnitude": 0.1, "insights": []},
        {"confidence": 0.4, "transformation_magnitude": 0.3, "insights": ["bridge"]},
        {"confidence": 0.65, "transformation_magnitude": 0.55, "insights": ["connection", "pattern"]},
    ]
    print("Snapshot:", snapshot(history))
    # This will plot when matplotlib is available; otherwise it prints a summary
    animate_transformation(history, show=False)


if __name__ == "__main__":
    main()

