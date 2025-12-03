# Purpose: grid utilities and small helpers to convert between index,
# serial index, and physical coordinates. Also provides an optional
# routine to assemble node positions and a dense adjacency matrix.
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class GridConfig:
    """Grid geometry and spacing configuration."""

    length: float
    width: float
    dx: float
    dy: float
    delta: float

    @property
    def nx(self) -> int:
        """Number of grid points in x direction (columns)."""
        return int(round(self.length / self.dx)) + 1

    @property
    def ny(self) -> int:
        """Number of grid points in y direction (rows)."""
        return int(round(self.width / self.dy)) + 1

    @property
    def n_nodes(self) -> int:
        """Total number of grid nodes."""
        return self.nx * self.ny


def ind2ser(i: int, j: int, cfg: GridConfig) -> int:
    """Map 2D grid indices (i,j) to a 0-based serial index s."""
    nx, ny = cfg.nx, cfg.ny
    if not (0 <= i < ny and 0 <= j < nx):
        raise ValueError(f"Bad (i, j)=({i}, {j}) for grid {ny}x{nx}")
    return i * nx + j


def ser2ind(s: int, cfg: GridConfig) -> Tuple[int, int]:
    """Map 0-based serial index s to 2D indices (i, j)."""
    nx, ny = cfg.nx, cfg.ny
    if not (0 <= s < nx * ny):
        raise ValueError(f"Bad serial index s={s} for grid with {nx * ny} nodes")
    i, j = divmod(s, nx)
    return i, j


def ind2pos(i: int, j: int, cfg: GridConfig) -> np.ndarray:
    """Return physical coordinates (x, y) for a grid index (i, j)."""
    x = j * cfg.dx
    y = i * cfg.dy
    return np.array([x, y], dtype=float)


def ser2pos(s: int, cfg: GridConfig) -> np.ndarray:
    """Return physical coordinates (x, y) for a serial index s."""
    i, j = ser2ind(s, cfg)
    return ind2pos(i, j, cfg)


def build_positions_and_adjacency(
    Nbd: np.ndarray, cfg: GridConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build node positions and a dense adjacency (0/1) matrix from Nbd.

    Note: the adjacency matrix is dense and intended for visualization or
    small problems only. For large grids prefer edge lists / sparse structures.
    """
    n_nodes = cfg.n_nodes
    if Nbd.shape[0] != n_nodes:
        raise ValueError(f"Nbd has shape {Nbd.shape}, but grid has n_nodes={n_nodes}")

    pos = np.zeros((n_nodes, 2), dtype=float)
    A = np.zeros((n_nodes, n_nodes), dtype=int)

    for s in range(n_nodes):
        pos[s, :] = ser2pos(s, cfg)
        neighbors_mask = Nbd[s, :] >= 0
        neighbor_serials = Nbd[s, neighbors_mask].astype(int)
        for nbd in neighbor_serials:
            if 0 <= nbd < n_nodes:
                A[s, nbd] = 1
            else:
                raise ValueError(f"Neighbor index {nbd} out of range for node {s}")

    return pos, A
