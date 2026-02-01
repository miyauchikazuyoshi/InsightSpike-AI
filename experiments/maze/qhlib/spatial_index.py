from __future__ import annotations

"""
Spatial grid index for fast neighborhood lookup around (row, col) cells.

Used by maze QueryHub runner to prefilter memory-origin direction nodes by
nearby cells before applying full weighted distance.

Design:
- Simple dict[(r, c)] -> set(node_id)
- Add-only (no delete path needed for current experiments)
- Methods to iterate nodes in rectangular window; ellipse check optional
"""

from typing import Dict, Iterable, Iterator, Optional, Sequence, Set, Tuple

Node = Tuple[int, int, int]


class SpatialGridIndex:
    def __init__(self, maze_shape: Tuple[int, int]):
        self._buckets: Dict[Tuple[int, int], Set[Node]] = {}
        self._h = int(maze_shape[0])
        self._w = int(maze_shape[1])

    @property
    def empty(self) -> bool:
        return not self._buckets

    def add(self, anchor: Tuple[int, int], node_id: Node) -> None:
        r, c = int(anchor[0]), int(anchor[1])
        if r < 0 or c < 0 or r >= self._h or c >= self._w:
            return
        key = (r, c)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = set()
            self._buckets[key] = bucket
        bucket.add(node_id)

    def iter_cells_rect(self, center: Tuple[int, int], Rr: int, Rc: int) -> Iterator[Tuple[int, int, Set[Node]]]:
        cr, cc = int(center[0]), int(center[1])
        Rr = max(0, int(Rr)); Rc = max(0, int(Rc))
        for dr in range(-Rr, Rr + 1):
            rr = cr + dr
            if rr < 0 or rr >= self._h:
                continue
            for dc in range(-Rc, Rc + 1):
                cc2 = cc + dc
                if cc2 < 0 or cc2 >= self._w:
                    continue
                b = self._buckets.get((rr, cc2))
                if not b:
                    continue
                yield rr, cc2, b

    def iter_nodes_rect(self, center: Tuple[int, int], Rr: int, Rc: int) -> Iterator[Node]:
        for _, _, bucket in self.iter_cells_rect(center, Rr, Rc):
            for nid in bucket:
                yield nid

    def iter_nodes_ellipse(self, center: Tuple[int, int], Rr: int, Rc: int) -> Iterator[Node]:
        # Ellipse in normalized grid: (dr/Rr)^2 + (dc/Rc)^2 <= 1
        cr, cc = int(center[0]), int(center[1])
        if Rr <= 0 and Rc <= 0:
            # treat as single-cell
            for nid in self._buckets.get((cr, cc), set()):
                yield nid
            return
        Rr = max(1, int(Rr)); Rc = max(1, int(Rc))
        for dr in range(-Rr, Rr + 1):
            rr = cr + dr
            if rr < 0 or rr >= self._h:
                continue
            for dc in range(-Rc, Rc + 1):
                cc2 = cc + dc
                if cc2 < 0 or cc2 >= self._w:
                    continue
                # ellipse check
                if (dr * dr) * (Rc * Rc) + (dc * dc) * (Rr * Rr) > (Rr * Rr) * (Rc * Rc):
                    continue
                b = self._buckets.get((rr, cc2))
                if not b:
                    continue
                for nid in b:
                    yield nid
