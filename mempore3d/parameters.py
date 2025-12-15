
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Data Classes for Parameters
# -----------------------------------------------------------------------------
@dataclass
class Domain:
    """Parameters defining the simulation domain and grid."""
    Lx: float = 10000e-9
    Ly: float = 10000e-9
    Lz: float = 20000e-9
    Nx: int = 128
    Ny: int = 128
    Nz: int = 129  # Will be forced to an odd number

@dataclass
class MembraneProps:
    """Intrinsic electrical properties of the membrane components."""
    R_lipid: float = 1e7          # Lipid resistance, Ohm·m^2
    C_lipid: float = 1e-2          # Lipid capacitance, F/m^2
    R_pore: float = 1e-1           # Pore resistance (conductive), Ohm·m^2
    C_pore: float = 1e-9           # Pore capacitance (near zero), F/m^2

@dataclass
class PhaseFieldParams:
    """Parameters for the phase-field model of the pore."""
    # --- Initial State ---
    initial_state: str = 'pore'           # Can be 'pore' or 'intact'
    initial_pore_radius: float = 10e-9        # m, Used if initial_state is 'pore'
    intact_noise_level: float = 1e-4          # Used if initial_state is 'intact'
    transition_thickness: float | None = None   # If None, defaults to 0.5*dx
    # --- Dynamics ---
    line_tension: float = 1.5e-11 # J/m, aka 'gamma'
    mobility: float = 5000.0      # m^2/(J*s), aka 'M'
    sigma_area: float = 0.0       # J/m^2, base membrane tension (sigma_0)
    evolve: str = 'on'            # 'on' to evolve, 'off' for static pore

@dataclass
class ThermalParams:
    """Parameters for thermal fluctuations."""
    T: float = 300.0          # Temperature in Kelvin
    k_B: float = 1.380649e-23  # Boltzmann constant, J/K
    add_noise: bool = True    # Flag to turn noise on/off

@dataclass
class Electrostatics:
    """Parameters for the electrostatic environment."""
    sigma_e: float = 1.0      # Electrolyte conductivity, S/m
    V_applied: float = 0.5    # Applied voltage across the box, V

@dataclass
class SolverParams:
    """Parameters controlling the numerical solver."""
    surface_diffusion: bool = True
    D_V: float = 0.0                      # If 0, set adaptively
    dt_safety: float = 0.01                   # Safety factor for base time step calculation
    n_tau_total: float = 8.0                  # Total simulation time in units of lipid RC time
    save_frames: int = 40                     # Number of frames to save for visualization
    implicit_dt_multiplier: float = 100.0     # Factor to increase dt for implicit Vm solver
    rebuild_vm_solver_every: int = 20       # Rebuild the Vm implicit matrix every N steps
    poisson_solver: str = 'spectral'          # 'spectral' (fast) or 'gamg' (parallel)
