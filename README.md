# MemPore3D

A 3D numerical simulation framework for electroporation in lipid membranes using a diffuse-interface (phase-field) approach coupled with electrostatics and thermal fluctuations.

Based on the method described in:
> S. Seifi and D. Salac, "A Diffuse-Interface Method for Pore Dynamics in Lipid Membranes under Electric Fields," [arXiv:1611.03902](https://arxiv.org/abs/1611.03902)

![Electroporation simulation showing Vm and pore radius evolution](evolution_analysis.gif)

## Overview

MemPore3D models how nanoscale pores form and evolve in lipid membranes subjected to external electric fields. The code couples three physical components:

- **Phase-field solver** -- Stochastic Allen-Cahn equation for pore structure evolution (spectral/FFT-based)
- **3D Poisson solver** -- Electrolyte potential via spectral FFT or PETSc/GAMG (MPI-parallel)
- **2D transmembrane potential solver** -- Implicit scheme coupling bulk ionic current with membrane properties

Thermal fluctuations enable spontaneous pore nucleation without artificial initial conditions.

## Installation

```bash
conda env create -f environment.yml
conda activate mempore3d
```

Key dependencies: NumPy, SciPy, Numba, PETSc (petsc4py), mpi4py, Matplotlib.

## Usage

Run a simulation case:

```bash
python runs/electroporations/case_1.py
```

Or use one of the test entry points:

```bash
python run_electroporation_test.py      # Full coupled simulation
python run_single_pore_test.py          # Phase-field only
python run_electrostatic_test.py        # Electrostatics only
```

Results are saved as `.npz` snapshots in the corresponding `case_*/` directories.

## Project Structure

```
mempore3d/
  core.py                 # Main simulation driver
  parameters.py           # Configuration dataclasses
  plotting.py             # Visualization utilities
  solvers/
    poisson_solver.py     # 3D electrostatic solvers (Spectral & PETSc)
    phase_field_solver.py # Stochastic Allen-Cahn solver
    leaky_dielectric_solver.py  # 2D Vm solver
runs/electroporations/    # Simulation case configurations (case_1 -- case_12)
plotting/                 # Post-processing scripts and notebooks
utils/                    # Data extraction utilities
```

## License

Research code. Please cite the paper above if you use this work.
