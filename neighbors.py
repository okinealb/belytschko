# Purpose: construct per-node neighbor lists and geometric bond vectors
# for the peridynamic-style mesh used by the simulation.
from typing import Tuple, List
import numpy as np
from tqdm import tqdm

from grid import GridConfig, ind2ser, ser2pos
from geometry import segments_intersect


def build_neighbors(
    cfg: GridConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build neighbor indices and bond geometry arrays.

    Returns:
      Nbd        : (n_nodes, max_neighbors) int array with 0-based neighbor indices (or -1)
      Xdiff1     : per-bond Δx
      Xdiff2     : per-bond Δy
      Xdiffnorm  : per-bond rest length (norm of [Δx, Δy])
    """
    nx = cfg.nx
    ny = cfg.ny
    dx = cfg.dx
    dy = cfg.dy
    delta = cfg.delta
    n_nodes = cfg.n_nodes

    # Integer index radius for local search window
    max_i_offset = int(delta / dy)
    max_j_offset = int(delta / dx)

    neighbors_list: List[List[int]] = [[] for _ in range(n_nodes)]
    mid_p1 = (0.0, cfg.width / 2.0)
    mid_p2 = (cfg.length / 2.0, cfg.width / 2.0)

    # Collect neighbor serial indices for each node using local window + circle test.
    for i in tqdm(range(ny), desc="Building neighbors (rows)", unit="row"):
        for j in range(nx):
            s = ind2ser(i, j, cfg)
            i_low = max(0, i - max_i_offset)
            i_high = min(ny - 1, i + max_i_offset)
            j_low = max(0, j - max_j_offset)
            j_high = min(nx - 1, j + max_j_offset)
            x_ij = ser2pos(s, cfg)

            for k in range(i_low, i_high + 1):
                for l in range(j_low, j_high + 1):
                    if k == i and l == j:
                        continue
                    dx_kl = (j - l) * dx
                    dy_kl = (k - i) * dy
                    if dx_kl * dx_kl + dy_kl * dy_kl > delta * delta:
                        continue
                    s_nbd = ind2ser(k, l, cfg)
                    x_kl = ser2pos(s_nbd, cfg)
                    # Exclude bonds that cross the fixed midline segment.
                    if segments_intersect(mid_p1, mid_p2, x_ij, x_kl):
                        continue
                    neighbors_list[s].append(s_nbd)

    # Allocate arrays sized to the maximum neighbor count found.
    max_neighbors = max((len(lst) for lst in neighbors_list), default=0)
    if max_neighbors == 0:
        # Degenerate: no neighbors found.
        Nbd = -np.ones((n_nodes, 1), dtype=int)
        Xdiff1 = np.ones((n_nodes, 1), dtype=float)
        Xdiff2 = np.ones((n_nodes, 1), dtype=float)
        Xdiffnorm = np.ones((n_nodes, 1), dtype=float)
        return Nbd, Xdiff1, Xdiff2, Xdiffnorm

    Nbd = -np.ones((n_nodes, max_neighbors), dtype=int)
    Xdiff1 = np.ones((n_nodes, max_neighbors), dtype=float)
    Xdiff2 = np.ones((n_nodes, max_neighbors), dtype=float)
    Xdiffnorm = np.ones((n_nodes, max_neighbors), dtype=float)

    # Fill arrays with neighbor indices and rest geometry.
    for s in range(n_nodes):
        neighs = neighbors_list[s]
        count = len(neighs)
        if count == 0:
            continue
        Nbd[s, :count] = neighs
        x = ser2pos(s, cfg)
        for idx, nb in enumerate(neighs):
            xcap = ser2pos(nb, cfg)
            xdiff = xcap - x
            xdiffnorm = float(np.linalg.norm(xdiff))
            Xdiff1[s, idx] = xdiff[0]
            Xdiff2[s, idx] = xdiff[1]
            Xdiffnorm[s, idx] = xdiffnorm

    return Nbd, Xdiff1, Xdiff2, Xdiffnorm
