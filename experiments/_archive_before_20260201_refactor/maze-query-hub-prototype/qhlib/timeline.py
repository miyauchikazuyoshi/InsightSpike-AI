from __future__ import annotations

from typing import List, Tuple

Node = Tuple[int, int, int]


def build_current_step_timeline(
    last_query: Node,
    direction_node: Node,
    current_query: Node,
    *,
    include_dir_to_next: bool = False,
    include_pair: bool = False,
) -> List[List[List[int]]]:
    """Return timeline edges for the current step.

    By default this returns only Q_prev -> dir to avoid leaking the next-Q edge into
    the step-end visualization. Optional flags can enable additional edges when needed.
    """

    edges: List[List[List[int]]] = []
    edges.append([
        [int(last_query[0]), int(last_query[1]), int(last_query[2])],
        [int(direction_node[0]), int(direction_node[1]), int(direction_node[2])],
    ])
    if include_dir_to_next:
        edges.append([
            [int(direction_node[0]), int(direction_node[1]), int(direction_node[2])],
            [int(current_query[0]), int(current_query[1]), int(current_query[2])],
        ])
    if include_pair:
        edges.append([
            [int(last_query[0]), int(last_query[1]), int(last_query[2])],
            [int(current_query[0]), int(current_query[1]), int(current_query[2])],
        ])
    return edges

