# Purpose: simple visualization helpers for mesh, neighbor counts, and damage.
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# keep interactive mode enabled so non-blocking shows work
plt.ion()

from grid import GridConfig, ind2ser, ser2pos
from simulation import compute_damage_ratio


def plot_damage(
    initial: np.ndarray, final: np.ndarray, cfg: GridConfig, *, block: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build and display a damage ratio image (ny x nx).

    Returns:
        ratio : per-node scalar damage ratio (n_nodes,)
        A     : 2D array arranged as (ny, nx) for imshow
    """
    nx, ny = cfg.nx, cfg.ny
    n_nodes = cfg.n_nodes

    ratio = compute_damage_ratio(initial, final)
    if ratio.shape[0] != n_nodes:
        raise ValueError(f"ratio length {ratio.shape[0]} != n_nodes {n_nodes}")

    A = np.zeros((ny, nx), dtype=float)
    for i in tqdm(range(ny), desc="Mapping damage rows", unit="row"):
        for j in range(nx):
            s = ind2ser(i, j, cfg)
            A[i, j] = ratio[s]

    plt.figure()
    plt.imshow(A, origin="lower")
    plt.colorbar(label="Damage ratio")
    plt.title("Damage map")
    plt.xlabel("j (x index)")
    plt.ylabel("i (y index)")
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)
    return ratio, A


def plot_nbdmap(Nbd: np.ndarray, cfg: GridConfig, *, block: bool = False) -> np.ndarray:
    """
    Build and display a neighbor-count image (ny x nx).
    Returns the 2D array used for display.
    """
    nx, ny = cfg.nx, cfg.ny
    n_nodes = cfg.n_nodes

    if Nbd.shape[0] != n_nodes:
        raise ValueError(f"Nbd rows {Nbd.shape[0]} != n_nodes {n_nodes}")

    neighbor_counts = (Nbd >= 0).sum(axis=1)
    A = np.zeros((ny, nx), dtype=int)
    for i in tqdm(range(ny), desc="Mapping neighbor rows", unit="row"):
        for j in range(nx):
            s = ind2ser(i, j, cfg)
            A[i, j] = neighbor_counts[s]

    plt.figure()
    plt.imshow(A, origin="lower")
    plt.colorbar(label="Neighbor count")
    plt.title("Neighborhood map")
    plt.xlabel("j (x index)")
    plt.ylabel("i (y index)")
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)

    return A


def draw_mesh(
    Nbd: np.ndarray, cfg: GridConfig, title: str = "mesh", *, block: bool = False
) -> None:
    """
    Draw the mesh as a LineCollection built from Nbd.

    The loop computes node positions and collects unique edges (single pass).
    """
    n_nodes = cfg.n_nodes
    pos = np.zeros((n_nodes, 2), dtype=float)
    segments_idx: list[tuple[int, int]] = []

    for s in tqdm(range(n_nodes), desc=f"Drawing {title}", unit="node"):
        pos[s, :] = ser2pos(s, cfg)
        row = Nbd[s, :]
        valid = row >= 0
        if not np.any(valid):
            continue
        neighs = row[valid].astype(int)
        for nb in neighs:
            if nb > s:
                segments_idx.append((s, nb))

    import matplotlib.collections as mcoll

    if segments_idx:
        arr = np.array(segments_idx, dtype=int)
        lc_segments = np.empty((arr.shape[0], 2, 2), dtype=float)
        lc_segments[:, 0, :] = pos[arr[:, 0]]
        lc_segments[:, 1, :] = pos[arr[:, 1]]
    else:
        lc_segments = None

    fig, ax = plt.subplots()
    if lc_segments is not None:
        lc = mcoll.LineCollection(lc_segments, linewidths=0.5, colors="k")
        ax.add_collection(lc)
        ax.scatter(pos[:, 0], pos[:, 1], s=5)
        ax.set_aspect("equal")
        ax.autoscale()
    else:
        ax.scatter(pos[:, 0], pos[:, 1], s=5)
        ax.set_aspect("equal")

    ax.set_title(f"Mesh (graph) — {title}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)
    return
