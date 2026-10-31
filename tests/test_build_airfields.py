"""Airfield geometry helpers (dashboard/build_airfields.py): line simplification and multipolygon ring stitching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import build_airfields as B  # noqa: E402


def test_simplify_drops_collinear_points_and_keeps_corners():
    assert B.simplify([(0, 0), (1, 0.1), (2, 0), (2, 5)], 0.5) == [(0, 0), (2, 0), (2, 5)]
    assert B.simplify([(0, 0), (5, 5), (0, 0.1)], 1) == [(0, 0), (5, 5), (0, 0.1)]   # near-loop keeps its far point


def test_stitch_joins_split_outer_ways_in_either_direction():
    a, b, c = [(0, 0), (1, 0)], [(1, 1), (1, 0)], [(1, 1), (0, 0)]
    (ring,) = B.stitch([a, b, c])
    assert ring[0] == ring[-1] and set(ring) == {(0, 0), (1, 0), (1, 1)}
