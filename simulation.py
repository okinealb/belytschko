# Purpose: time-stepping projection logic and damage-ratio helper.
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm

from grid import GridConfig, ind2ser


@dataclass
class SimulationParams:
    """Numerical and material parameters used by the projection solver."""

    num_steps: int = 1800
    dt: float = 25e-9
    sigma0: float = 12e6
    rho: float = 2440.0
    Gnot: float = 135.0
    E: float = 72e9
    nu: float = 0.22


def compute_damage_ratio(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """
    Compute per-node damage ratio = surviving_bonds / initial_bonds.

    Both inputs are (n_nodes, max_neighbors) int arrays where a value >= 0
    marks a valid neighbor and -1 marks a broken/missing bond.
    """
    if initial.shape != final.shape:
        raise ValueError(
            f"initial and final must have same shape, got {initial.shape} vs {final.shape}"
        )

    initial_exists = initial >= 0
    final_exists = final >= 0

    num_initial = initial_exists.sum(axis=1)
    num_final = final_exists.sum(axis=1)

    ratio = np.zeros_like(num_initial, dtype=float)
    mask = num_initial > 0
    ratio[mask] = num_final[mask] / num_initial[mask]

    return ratio


def run_projection(
    Nbd_init: np.ndarray,
    Xdiff1: np.ndarray,
    Xdiff2: np.ndarray,
    Xdiffnorm: np.ndarray,
    cfg: GridConfig,
    params: SimulationParams,
    *,
    track_history: bool = False,
) -> Tuple[np.ndarray, Optional[list[np.ndarray]]]:
    """
    Time-stepping loop that updates displacements and breaks bonds based on
    a simple per-bond stretch criterion.

    Inputs:
      - Nbd_init: neighbor index matrix (n_nodes, max_neighbors), -1 = no bond
      - Xdiff*: rest geometry arrays matching Nbd_init
      - cfg: grid configuration
      - params: simulation parameters

    Returns:
      - Nbd_final: neighbor matrix after evolution
      - history: optional list of Nbd snapshots (if track_history=True)
    """
    dx = cfg.dx
    dy = cfg.dy
    nx = cfg.nx
    ny = cfg.ny
    totalnodes = cfg.n_nodes

    # material / solver constants
    sigma = params.sigma0 / dx
    rho = params.rho
    Gnot = params.Gnot
    E = params.E
    nu = params.nu

    delta = getattr(cfg, "delta", None)
    if delta is None:
        raise ValueError("cfg must have a 'delta' attribute (interaction horizon).")

    snot = np.sqrt(4.0 * np.pi * Gnot / (9.0 * E * delta))
    cnot = 6.0 * E / (np.pi * (delta**3) * (1.0 - nu))

    # working state
    Nbd = Nbd_init.copy()
    m, n = Nbd.shape

    Udiff1 = np.zeros_like(Xdiff1)
    Udiff2 = np.zeros_like(Xdiff2)

    # preallocated temporaries to minimize allocations inside the time loop
    Bigvect1 = np.zeros_like(Udiff1)
    Bigvect2 = np.zeros_like(Udiff2)
    Bigvectnorm = np.zeros_like(Udiff1)
    S = np.zeros_like(Udiff1)
    Multipl = np.zeros_like(Udiff1)
    Force1 = np.zeros_like(Udiff1)
    Force2 = np.zeros_like(Udiff2)
    totalintforce = np.zeros((totalnodes, 2), dtype=float)

    # external traction on top/bottom rows (y-component)
    extforce = np.zeros((totalnodes, 2), dtype=float)
    for j in range(nx):
        extforce[ind2ser(0, j, cfg), 1] = -sigma
        extforce[ind2ser(ny - 1, j, cfg), 1] = sigma

    # time integration state: displacement, velocity, acceleration
    uold = np.zeros((totalnodes, 2), dtype=float)
    uolddot = np.zeros((totalnodes, 2), dtype=float)
    uolddotdot = np.zeros((totalnodes, 2), dtype=float)

    dt = params.dt
    history: Optional[list[np.ndarray]] = [] if track_history else None

    dxdy = dx * dy
    inv_rho = 1.0 / rho
    half_dt = 0.5 * dt

    prev_Nbd = None

    # main time loop: compute internal forces, update kinematics, and break bonds
    for t in tqdm(range(params.num_steps), desc="Projecting", unit="step"):
        u0 = uold + dt * uolddot + 0.5 * (dt**2) * uolddotdot

        restrict = Nbd >= 0  # mask of existing bonds

        # vectorized neighbor displacement: u0_neighbors - u0[:,None,:]
        nb_idx = Nbd.copy()
        invalid_mask = nb_idx < 0
        if invalid_mask.any():
            nb_idx[invalid_mask] = 0
        u0_neighbors = u0[nb_idx]
        Udiff = u0_neighbors - u0[:, None, :]
        Udiff1[:, :] = Udiff[:, :, 0]
        Udiff2[:, :] = Udiff[:, :, 1]
        Udiff1[~restrict] = 0.0
        Udiff2[~restrict] = 0.0

        # assemble bond stretches and internal forces
        Bigvect1[:] = Udiff1 + Xdiff1
        Bigvect2[:] = Udiff2 + Xdiff2
        Bigvectnorm[:] = np.hypot(Bigvect1, Bigvect2)

        valid_mask = restrict & (Xdiffnorm != 0.0)
        S.fill(0.0)
        S[valid_mask] = (Bigvectnorm[valid_mask] - Xdiffnorm[valid_mask]) / Xdiffnorm[
            valid_mask
        ]

        Multipl.fill(0.0)
        mv2 = valid_mask & (Bigvectnorm != 0.0)
        Multipl[mv2] = cnot * (S[mv2] / Bigvectnorm[mv2])

        Force1[:] = Multipl * Bigvect1
        Force2[:] = Multipl * Bigvect2

        totalintforce[:, 0] = np.sum(Force1 * restrict, axis=1)
        totalintforce[:, 1] = np.sum(Force2 * restrict, axis=1)

        u0dotdot = (totalintforce * dxdy + extforce) * inv_rho

        # break bonds that exceed critical stretch
        keep_mask = (S - snot) < 0
        Nbd = np.where(keep_mask, Nbd, -1)

        u0dot = uolddot + half_dt * uolddotdot + half_dt * u0dotdot

        # advance solution state
        uold = u0
        uolddot = u0dot
        uolddotdot = u0dotdot

        if track_history:
            history.append(Nbd.copy())

        prev_Nbd = Nbd.copy()

    return Nbd, history
