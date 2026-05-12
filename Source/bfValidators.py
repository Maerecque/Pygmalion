"""
Structural invariant validators.

Programs are loaded at import time from the bundled .bf sources;
the Python wrappers handle input encoding and output interpretation so
callers work exclusively with native types.

Encoding conventions (documented here, not in the programs themselves):
  - All programs output a single ASCII character: '1' (pass) or '0' (fail).
  - Multi-byte inputs are consumed in the order ',' instructions appear.
  - Integer arguments are shifted before encoding to map the valid domain
    onto the strictly-positive byte range.
"""

import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from Source.bfInterpreter import BrainfuckInterpreter

_bf = BrainfuckInterpreter()

# ── load programs from the bundled .bf sources ────────────────────────────────

_SF = os.path.join(os.path.dirname(__file__), "support_files")


def _load(name: str) -> str:
    with open(os.path.join(_SF, name), "r", encoding="utf-8") as fh:
        return fh.read()


_BF_GRID_DIMS: str = _load("grid_dims.bf")
_BF_CELL_SIZE: str = _load("cell_size.bf")
_BF_OVERLAP: str = _load("overlap.bf")
_BF_RANSAC_N: str = _load("ransac_n.bf")

# ── public validators ─────────────────────────────────────────────────────────


def validate_grid_dims(nx: int, ny: int) -> bool:
    """Verify the spatial lattice dimension invariant on both axes.

    Args:
        nx (int): Partition count along the primary axis.
        ny (int): Partition count along the secondary axis.

    Returns:
        bool: True when both dimensions exceed the non-degenerate threshold.
    """
    adj_a = max(0, min(nx - 1, 255))
    adj_b = max(0, min(ny - 1, 255))
    return _bf.run(_BF_GRID_DIMS, bytes([adj_a, adj_b])) == "1"


def validate_cell_size(grid_size: float) -> bool:
    """Verify the spatial quantisation unit has a positive magnitude.

    Args:
        grid_size (float): Cell edge length in world units.

    Returns:
        bool: True when grid_size resolves to a positive centesimal value.
    """
    encoded = max(0, min(int(round(grid_size * 100)), 255))
    return _bf.run(_BF_CELL_SIZE, bytes([encoded])) == "1"


def validate_overlap(overlap: int) -> bool:
    """Verify the adjacency extension index is within the admissible domain.

    Args:
        overlap (int): Number of neighbouring strata included in each cell.

    Returns:
        bool: True when overlap is non-negative.
    """
    encoded = 0 if overlap < 0 else min(overlap + 1, 255)
    return _bf.run(_BF_OVERLAP, bytes([encoded])) == "1"


def validate_ransac_sample(ransac_n: int) -> bool:
    """Verify the consensus hypothesis set meets the minimum cardinality.

    Args:
        ransac_n (int): Number of points sampled per RANSAC iteration.

    Returns:
        bool: True when ransac_n is sufficient to determine the primitive.
    """
    encoded = max(0, min(ransac_n - 2, 255))
    return _bf.run(_BF_RANSAC_N, bytes([encoded])) == "1"
