#!/bin/bash
# Complete refactoring of mempore3d_petsc.py into package structure

echo "Creating package structure..."

# Create all __init__.py files
cat > mempore/__init__.py << 'EOF'
"""
Mempore: Membrane Electroporation Simulation Package

A coupled electrostatics and phase-field simulation framework for
modeling pore formation and evolution during membrane electroporation.

Author: Saman Seifi
"""
from mempore.parameters import (
    Domain,
    MembraneProps,
    PhaseFieldParams,
    ThermalParams,
    Electrostatics,
    SolverParams,
    EPS0,
    EPS_W
)

from mempore.simulation import simulate_membrane_charging

__version__ = "1.0.0"
__author__ = "Saman Seifi"

__all__ = [
    "Domain",
    "MembraneProps",
    "PhaseFieldParams",
    "ThermalParams",
    "Electrostatics",
    "SolverParams",
    "simulate_membrane_charging",
    "EPS0",
    "EPS_W",
]
EOF

cat > mempore/solvers/__init__.py << 'EOF'
"""Poisson solvers for 3D electrostatics."""
from mempore.solvers.spectral import SpectralPoissonSolver
from mempore.solvers.gamg import PETScPoissonGAMG
from mempore.solvers.vm_solver import ImplicitVMSolver

__all__ = ["SpectralPoissonSolver", "PETScPoissonGAMG", "ImplicitVMSolver"]
EOF

cat > mempore/physics/__init__.py << 'EOF'
"""Physics modules for membrane electroporation."""
from mempore.physics.phase_field import PhaseFieldSolver
from mempore.physics.electrostatics import (
    blend_properties,
    blend_properties_derivative,
    blend_properties_sharp,
    estimate_base_time_step
)

__all__ = [
    "PhaseFieldSolver",
    "blend_properties",
    "blend_properties_derivative",
    "blend_properties_sharp",
    "estimate_base_time_step"
]
EOF

cat > mempore/utils/__init__.py << 'EOF'
"""Utility functions for grid, visualization, and analysis."""
from mempore.utils.grid import (
    create_grid,
    initialize_phase_field,
    calculate_pore_radius,
    calculate_pore_radius_simple
)
from mempore.utils.visualization import plot_results

__all__ = [
    "create_grid",
    "initialize_phase_field",
    "calculate_pore_radius",
    "calculate_pore_radius_simple",
    "plot_results"
]
EOF

echo "✓ Created all __init__.py files"
echo "✓ Package structure complete!"
echo ""
echo "Package structure:"
echo "mempore/"
echo "├── __init__.py"
echo "├── parameters.py"
echo "├── simulation.py (to be created)"
echo "├── solvers/"
echo "│   ├── __init__.py"
echo "│   ├── spectral.py"
echo "│   ├── gamg.py"
echo "│   └── vm_solver.py"
echo "├── physics/"
echo "│   ├── __init__.py"
echo "│   ├── phase_field.py"
echo "│   └── electrostatics.py (to be created)"
echo "└── utils/"
echo "    ├── __init__.py"
echo "    ├── grid.py"
echo "    └── visualization.py (to be created)"
EOF

chmod +x complete_refactor.sh
./complete_refactor.sh
