# Purpose: minimal driver that builds the grid, computes neighbors,
# runs the projection simulation, and produces plots.
from grid import GridConfig
from neighbors import build_neighbors
from simulation import SimulationParams, run_projection
from plotting import plot_damage, plot_nbdmap, draw_mesh, animate_history


def main():
    # Grid configuration
    cfg = GridConfig(
        length=0.1,
        width=0.04,
        dx=0.001, # both double the usual
        dy=0.001,
        delta=0.002,
    )

    # Build neighbor geometry and bond vectors
    Nbd_init, Xdiff1, Xdiff2, Xdiffnorm = build_neighbors(cfg)

    # Run the time-stepping / projection simulation
    params = SimulationParams()
    Nbd_final, history = run_projection(
        Nbd_init, Xdiff1, Xdiff2, Xdiffnorm, cfg, params, track_history=True
    )

    # Visualize results: neighbor-count map, damage map, initial/final meshes.
    # Use blocking show at end to keep windows open.
    plot_nbdmap(Nbd_final, cfg, block=False)
    ratio, A_damage = plot_damage(Nbd_init, Nbd_final, cfg, block=False)
    draw_mesh(Nbd_init, cfg, title="Initial mesh", block=False)
    draw_mesh(Nbd_final, cfg, title="Final mesh", block=False)

    # Animate the recorded history (downsampled for speed)
    if history:
        animate_history(history, Nbd_init, cfg, step_stride=10, pause=0.02)

    # Final blocking show to keep all plots open
    import matplotlib.pyplot as plt

    plt.show(block=True)


if __name__ == "__main__":
    main()
