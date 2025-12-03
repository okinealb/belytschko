# Purpose: simple visualization helpers for mesh, neighbor counts, and damage.
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from grid import GridConfig, ind2ser, ser2pos
from simulation import compute_damage_ratio

# keep interactive mode enabled so non-blocking shows work
plt.ion()


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


def animate_history(
    history: np.ndarray,
    Nbd_init: np.ndarray,
    cfg: GridConfig,
    step_stride: int = 10,
    pause: float = 0.02,
) -> None:
    """
    Animate neighbor-count and damage maps using FuncAnimation for replay.
    - precomputes index map and per-frame stacks for speed
    - repeats automatically (repeat=True)
    """
    from matplotlib import animation

    if not history:
        print("No history to animate")
        return

    nx, ny = cfg.nx, cfg.ny
    n_nodes = cfg.n_nodes

    # map (i,j) -> serial index once, so per-frame mapping is a fast array-index
    idx_map = np.empty((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            idx_map[i, j] = ind2ser(i, j, cfg)

    # downsample frames and precompute stacks
    frames = history[::step_stride]
    counts_stack = np.array(
        [(h >= 0).sum(axis=1) for h in frames]
    )  # (n_frames, n_nodes)
    damage_stack = np.array(
        [compute_damage_ratio(Nbd_init, h) for h in frames]
    )  # (n_frames, n_nodes)

    # initial 2D arrays for display (use idx_map to reshape)
    initial_counts = counts_stack[0][idx_map]
    initial_damage = damage_stack[0][idx_map]

    vmax_counts = int(counts_stack.max()) if counts_stack.size else 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(
        initial_counts, origin="lower", cmap="viridis", vmin=0, vmax=vmax_counts
    )
    ax1.set_title("Neighbor count")
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Neighbor count")

    im2 = ax2.imshow(initial_damage, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("Damage ratio (surviving / initial)")
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Damage ratio")

    plt.tight_layout()

    def update(frame_idx):
        im1.set_data(counts_stack[frame_idx][idx_map])
        im2.set_data(damage_stack[frame_idx][idx_map])
        return im1, im2

    interval_ms = max(int(pause * 1000), 1)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
        repeat_delay=1000,
    )

    plt.show(block=True)
