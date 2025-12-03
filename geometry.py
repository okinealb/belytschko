# Purpose: lightweight, robust 2D segment intersection utilities.
from typing import Sequence, Tuple
import numpy as np


Point = Tuple[float, float]


def _to_point(p: Sequence[float]) -> Point:
    """Normalize input to an (x, y) tuple of floats."""
    return float(p[0]), float(p[1])


def _orientation(p: Point, q: Point, r: Point) -> int:
    """
    Return orientation of triplet (p, q, r).
    0 = colinear, 1 = clockwise, 2 = counterclockwise.
    """
    (px, py), (qx, qy), (rx, ry) = p, q, r
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Return True if q lies on the closed segment pr (assumes colinear)."""
    (px, py), (qx, qy), (rx, ry) = p, q, r
    return (
        min(px, rx) - 1e-12 <= qx <= max(px, rx) + 1e-12
        and min(py, ry) - 1e-12 <= qy <= max(py, ry) + 1e-12
    )


def segments_intersect(
    a1: Sequence[float], a2: Sequence[float], b1: Sequence[float], b2: Sequence[float]
) -> bool:
    """
    Return True if the closed segments [a1,a2] and [b1,b2] intersect.

    Handles proper intersections and colinear overlap.
    """
    A1 = _to_point(a1)
    A2 = _to_point(a2)
    B1 = _to_point(b1)
    B2 = _to_point(b2)

    o1 = _orientation(A1, A2, B1)
    o2 = _orientation(A1, A2, B2)
    o3 = _orientation(B1, B2, A1)
    o4 = _orientation(B1, B2, A2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(A1, B1, A2):
        return True
    if o2 == 0 and _on_segment(A1, B2, A2):
        return True
    if o3 == 0 and _on_segment(B1, A1, B2):
        return True
    if o4 == 0 and _on_segment(B1, A2, B2):
        return True

    return False
