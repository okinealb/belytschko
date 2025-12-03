# belytschko

A compact Python translation of crack growth without remeshing. Adapted from a MATLAB reference implementation by Prof. Debdeep Bhattacharya, this repo provides a vectorized peridynamic-style demo (neighbor/bond model), a projection time‑stepper that breaks bonds, and visualization utilities.

## Table of Contents
- [Project purpose](#project-purpose)
- [Paper summary](#paper-summary)
  - [Original contribution](#original-contribution)
  - [Main results](#main-results)
  - [Mathematical tools](#mathematical-tools)
- [What this code does](#what-this-code-does)
- [How this maps to the paper](#how-this-maps-to-the-paper)
- [Quick start (macOS)](#quick-start-macos)
- [Visualization notes](#visualization-notes)
- [Performance tips](#performance-tips)
- [Project structure](#project-structure)
- [Acknowledgements & references](#acknowledgements--references)

## Project purpose
Provide a small, readable, and faster Python implementation of a classroom/demo code that illustrates crack-like growth without remeshing. The focus is on clarity, vectorized NumPy operations, and useful visualization for exploration and teaching.

## Paper summary
Reference: ["A finite element method for crack growth without remeshing" (Belytschko et al.)](#acknowledgements--references).

### Original contribution
Introduce enrichment of the finite-element approximation (partition of unity / XFEM) so cracks and discontinuities are represented inside elements without remeshing the mesh.

### Main results
Demonstrated accurate crack initiation/propagation and stress‑intensity estimates while avoiding mesh updates; validated on benchmark fracture examples.

### Mathematical tools
Partition‑of‑unity enrichment, level‑set geometry for crack location, special enrichment functions for discontinuity and near‑tip singularity, variational FEM formulations and fracture mechanics criteria.

## What this code does
- Build a regular 2D grid and per-node neighbor lists (bond geometry).
- Assemble per-bond forces in a projection/time-stepping loop and break bonds by stretch threshold.
- Compute per-node damage ratios and visualize neighbor maps, damage maps, and meshes.
- Vectorized NumPy implementation, Matplotlib visualization, and tqdm progress indicators.

## How this maps to the paper
This repository is a simplified demonstrator capturing the "no remeshing" idea as a bond-based / nonlocal interaction model rather than a full XFEM implementation. It is intended for experimentation and demonstration rather than a production XFEM solver.

## Quick start from terminal (macOS)
1. Create & activate virtual environment:
   - `$ python3 -m venv venv`
   - `$ source venv/bin/activate`
2. Install dependencies:
   - `$ pip install -r requirements.txt`
   - or: `$ pip install numpy matplotlib tqdm`
3. Run demo:
   - `$ python main.py`
   - The script uses a blocking final `plt.show(block=True)` so figures remain until closed.

## Visualization notes
- Plotting functions accept a `block` parameter. Use `block=False` for non-blocking incremental plots, and call `plt.show(block=True)` at the end to keep windows open.
- If running in headless or CI, ensure a GUI backend is available or save figures to files instead.

## Performance tips
- Keep `track_history=False` for large runs (history stores full Nbd snapshots).
- Avoid dense n×n adjacency for large grids; edge lists are used for plotting.
- For further speed consider numba JIT or a compiled kernel for the force assembly.

## Project structure
- `main.py`: example driver
- `grid.py`: grid and coordinate helpers
- `neighbors.py`: neighbor list and bond geometry builder
- `simulation.py`: time-stepping projection solver and damage helper
- `plotting.py`: visualization utilities
- `geometry.py`: 2D segment intersection helpers

## Acknowledgements & references
- MATLAB reference and course material: [Prof. Debdeep Bhattacharya](https://github.com/debdeepbh/numerical/tree/master/crack_old) (code translated and optimized for Python).
- Core method inspiration: Belytschko et al., ["A finite element method for crack growth without remeshing"](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0207%2819990910%2946%3A1%3C131%3A%3AAID-NME726%3E3.0.CO%3B2-J) (XFEM literature).
- Suggested reading: XFEM and partition-of-unity references, standard FEM textbooks on variational formulations.
