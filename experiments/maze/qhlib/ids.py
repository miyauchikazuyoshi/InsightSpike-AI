from __future__ import annotations

from typing import Tuple

Node = Tuple[int, int, int]


def canonical_node_key(node: Node) -> str:
    return f"{int(node[0])},{int(node[1])},{int(node[2])}"


def is_query_node(node: Node) -> bool:
    try:
        return int(node[2]) == -1
    except Exception:
        return False


def same_cell(a: Node, b: Node) -> bool:
    return int(a[0]) == int(b[0]) and int(a[1]) == int(b[1])

