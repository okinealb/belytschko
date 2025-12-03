# Purpose: minimal driver that builds the grid, computes neighbors,
# runs the projection simulation, and produces plots.
from grid import GridConfig
from neighbors import build_neighbors
from simulation import SimulationParams, run_projection
from plotting import plot_damage, plot_nbdmap, draw_mesh


def main():
    # Grid configuration
    cfg = GridConfig(
        length=0.1,
        width=0.04,
        dx=0.0005,
        dy=0.0005,
        delta=0.002,
    )

    # Build neighbor geometry and bond vectors
    Nbd_init, Xdiff1, Xdiff2, Xdiffnorm = build_neighbors(cfg)

    # Run the time-stepping / projection simulation
    params = SimulationParams()
    Nbd_final, history = run_projection(
        Nbd_init, Xdiff1, Xdiff2, Xdiffnorm, cfg, params
    )

    # Visualize results: neighbor-count map, damage map, initial/final meshes.
    # Use blocking show at end to keep windows open.
    plot_nbdmap(Nbd_final, cfg, block=False)
    ratio, A_damage = plot_damage(Nbd_init, Nbd_final, cfg, block=False)
    draw_mesh(Nbd_init, cfg, title="Initial mesh", block=False)
    draw_mesh(Nbd_final, cfg, title="Final mesh", block=False)

    # Final blocking show to keep all plots open
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
