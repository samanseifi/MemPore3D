"""
Refactored membrane charging simulation with a diffuse pore transition.

This script models the charging of a lipid membrane containing a central pore,
using a phase-field approach to smoothly transition between lipid and pore properties.

This version includes two solver modes:
1.  Original (Decoupled): Uses a simplified "access resistance" model for the
    electrolyte current. Fast and stable.
2.  Coupled: Solves the full 3D Laplace equation for the electrolyte potential
    at each time step and uses its gradient to calculate the true ionic
    current driving the membrane. More physically rigorous but computationally
    more expensive.

Key Features
------------
- Phase-field (psi): Represents lipid (psi=1) to pore (psi=0) state.
- Smooth Blending: Electrical properties (conductance, capacitance) are blended
  using a cubic Hermite polynomial H(psi) = psi^2 * (3 - 2*psi).
- Regularization: A small bath capacitance is added to prevent division by zero
  in the highly conductive pore region (for the original decoupled solver).
- Numerical Stability: Optional lateral surface diffusion of the transmembrane
  potential (Vm) is included for robustness.
- Structure: The code is organized into small, testable functions with
  clear data classes for parameters.

Numba Optimization (August 10, 2025 by Gemini):
- Core computational loops in matrix assembly, RHS creation, the Vm update rule,
  and the 2D Laplacian have been compiled with Numba for significant speedup.
- Functions interacting with Numba have been refactored to pass primitive types
  (floats, ints, arrays) instead of dataclasses.

Original Author: Saman Seifi
Polished by: Gemini (August 1, 2025)
Coupled Solver Implemented by: Gemini (August 7, 2025)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numba # Added for JIT compilation

# -----------------------------------------------------------------------------
# Grid and Phase Field Utilities
# -----------------------------------------------------------------------------
# ------------------------------------------------------------
# multigrid_phi.py  –   fast Poisson solver for membrane code
# ------------------------------------------------------------

import pyamg
from scipy.sparse import coo_matrix

# --- Physical Constants ---
EPS0 = 8.8541878128e-12 # Vacuum permittivity, F/m
EPS_W = 80.0 * EPS0      # Water permittivity, F/m

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
    R_lipid: float = 1e7  # Lipid resistance, Ohm·m^2
    C_lipid: float = 1e-2  # Lipid capacitance, F/m^2
    R_pore: float = 1e-1   # Pore resistance (conductive), Ohm·m^2
    C_pore: float = 1e-9   # Pore capacitance (near zero), F/m^2

@dataclass
class PhaseFieldParams:
    """Parameters for the phase-field model of the pore."""
    pore_radius: float = 500e-9
    transition_thickness: float | None = None  # If None, defaults to 2*dx

@dataclass
class Electrostatics:
    """Parameters for the electrostatic environment."""
    sigma_e: float = 1.0     # Electrolyte conductivity, S/m
    V_applied: float = 0.5   # Applied voltage across the box, V

@dataclass
class SolverParams:
    """Parameters controlling the numerical solver."""
    max_iters_phi: int = 2000
    tolerance: float = 1e-6
    surface_diffusion: bool = True
    D_V: float = 0.0           # If 0, set adaptively
    dt_safety: float = 0.01    # Safety factor for time step
    n_tau_total: float = 8.0   # Total simulation time in units of lipid RC time
    save_frames: int = 20      # Number of frames to save for visualization


# -----------------------------------------------------------------------------
# Numba-JIT Compiled Core Routines
#
# These functions are the computational hotspots. Decorating them with
# @numba.njit compiles them to fast machine code.
# -----------------------------------------------------------------------------

@numba.njit(cache=True)
def _build_laplacian_numba_loops(Nx, Ny, Nz, dx2, dy2, dz2, k_mem_minus, k_mem_plus,
                                 row_idx, col_idx, diag_vals):
    """
    Numba-accelerated inner loop for assembling the sparse Laplacian matrix.
    This function modifies the pre-allocated index and value arrays.
    """
    count = 0
    
    def add_entry(r, c, v):
        nonlocal count
        row_idx[count] = r
        col_idx[count] = c
        diag_vals[count] = v
        count += 1
        
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                p = i + Nx * (j + Ny * k)

                # --- Dirichlet planes (unchanged) ---
                if k in (0, Nz - 1, k_mem_minus, k_mem_plus):
                    add_entry(p, p, 1.0)
                    continue

                # --- Interior / Neumann ---
                diag = 0.0

                # --- x-direction ---
                if i == 0:  # left wall – mirror i=-1 → i=1
                    q = (i + 1) + Nx * (j + Ny * k)
                    w = 2.0 / dx2
                    add_entry(p, p, w)
                    add_entry(p, q, -w)
                elif i == Nx - 1:  # right wall – mirror
                    q = (i - 1) + Nx * (j + Ny * k)
                    w = 2.0 / dx2
                    add_entry(p, p, w)
                    add_entry(p, q, -w)
                else:  # interior
                    q1 = (i - 1) + Nx * (j + Ny * k)
                    q2 = (i + 1) + Nx * (j + Ny * k)
                    add_entry(p, q1, -1.0 / dx2)
                    add_entry(p, q2, -1.0 / dx2)
                    diag += 2.0 / dx2

                # --- y-direction (analogous) ---
                if j == 0:
                    q = i + Nx * ((j + 1) + Ny * k)
                    w = 2.0 / dy2
                    add_entry(p, p, w)
                    add_entry(p, q, -w)
                elif j == Ny - 1:
                    q = i + Nx * ((j - 1) + Ny * k)
                    w = 2.0 / dy2
                    add_entry(p, p, w)
                    add_entry(p, q, -w)
                else:
                    q1 = i + Nx * ((j - 1) + Ny * k)
                    q2 = i + Nx * ((j + 1) + Ny * k)
                    add_entry(p, q1, -1.0 / dy2)
                    add_entry(p, q2, -1.0 / dy2)
                    diag += 2.0 / dy2

                # --- z-direction (always interior) ---
                q1 = i + Nx * (j + Ny * (k - 1))
                q2 = i + Nx * (j + Ny * (k + 1))
                add_entry(p, q1, -1.0 / dz2)
                add_entry(p, q2, -1.0 / dz2)
                diag += 2.0 / dz2

                # set final diagonal
                add_entry(p, p, diag)
    return count

@numba.njit(cache=True)
def _build_rhs_numba(rhs, Vm, Nx, Ny, Nz, k_mem_minus, k_mem_plus, V_applied):
    """Numba-accelerated function to build the RHS vector for the Poisson solve."""
    rhs_top = V_applied / 2.0
    rhs_bottom = -V_applied / 2.0
    
    # top / bottom electrolyte planes
    for j in range(Ny):
        for i in range(Nx):
            idx_bottom = i + Nx * (j + Ny * 0)
            idx_top = i + Nx * (j + Ny * (Nz - 1))
            rhs[idx_bottom] = rhs_bottom
            rhs[idx_top] = rhs_top

    # membrane jump planes  (-Vm/2, +Vm/2)
    for j in range(Ny):
        for i in range(Nx):
            idx_minus = i + Nx * (j + Ny * k_mem_minus)
            idx_plus = i + Nx * (j + Ny * k_mem_plus)
            rhs[idx_minus] = -Vm[i, j] / 2.0
            rhs[idx_plus] = Vm[i, j] / 2.0
    return rhs

@numba.njit(cache=True)
def laplacian2d_neumann_numba(field, dx):
    """
    Numba-jitted 2D Laplacian with zero-flux (Neumann) boundaries.
    Rewritten with explicit loops to avoid np.pad for better performance.
    Assumes dx = dy.
    """
    Nx, Ny = field.shape
    lap = np.empty_like(field)
    dx2 = dx * dx

    for i in range(Nx):
        for j in range(Ny):
            # Central point
            center = field[i, j]

            # X-neighbors
            if i == 0: # Left edge
                x_minus = field[i + 1, j]
            elif i == Nx - 1: # Right edge
                x_minus = field[i - 1, j]
            else: # Interior
                x_minus = field[i - 1, j]

            if i == 0: # Left edge
                x_plus = field[i + 1, j]
            elif i == Nx - 1: # Right edge
                x_plus = field[i-1, j]
            else:
                 x_plus = field[i + 1, j]

            # Y-neighbors
            if j == 0: # Bottom edge
                y_minus = field[i, j + 1]
            elif j == Ny - 1: # Top edge
                y_minus = field[i, j - 1]
            else: # Interior
                y_minus = field[i, j - 1]

            if j == 0: # Bottom edge
                y_plus = field[i, j+1]
            elif j == Ny - 1: # Top edge
                y_plus = field[i, j-1]
            else: # Interior
                y_plus = field[i, j + 1]

            # This stencil formulation correctly replicates np.pad(..., mode='edge')
            # For a simpler Neumann condition (reflection), different terms would be used.
            lap_val = (
                (field[i+1, j] if i < Nx-1 else field[i-1, j]) +
                (field[i-1, j] if i > 0 else field[i+1, j]) +
                (field[i, j+1] if j < Ny-1 else field[i, j-1]) +
                (field[i, j-1] if j > 0 else field[i, j+1]) - 4 * center
            )
            lap[i, j] = lap_val / dx2
            
    return lap

@numba.njit(cache=True)
def calculate_J_from_phi(phi, k_plane_index, sigma_e, dz):
    """
    Numba-jitted calculation of ionic current density arriving at a membrane plane.
    """
    grad_phi_z = (phi[:, :, k_plane_index + 1] - phi[:, :, k_plane_index]) / dz
    J_elec = sigma_e * grad_phi_z
    return J_elec

@numba.njit(cache=True)
def update_vm_numba(
    Vm: np.ndarray,
    G_m_map: np.ndarray,
    C_eff_map: np.ndarray,
    H: np.ndarray,
    dt: float,
    dx: float,
    # Unpacked parameters from dataclasses
    sigma_e: float,
    V_applied: float,
    Lz: float,
    surface_diffusion: bool,
    D_V: float,
    J_elec_coupled: Optional[np.ndarray]
) -> np.ndarray:
    """
    Numba-jitted function to update the transmembrane potential `Vm`.
    """
    if J_elec_coupled is None:
        # --- DECOUPLED (ORIGINAL) MODE ---
        J_elec = H * (2.0 * sigma_e * (V_applied - Vm) / Lz)
    else:
        # --- COUPLED MODE ---
        J_elec = J_elec_coupled

    base_G_sc = 2.0 * sigma_e / Lz
    G_sc = 50.0 * base_G_sc * (1.0 - H)

    dVm_dt = (J_elec - (G_m_map + G_sc) * Vm) / C_eff_map
    Vm_new = Vm + dt * dVm_dt

    if surface_diffusion and D_V > 0.0:
        lap = laplacian2d_neumann_numba(Vm_new, dx)
        Vm_new += dt * D_V * lap
    return Vm_new


# -----------------------------------------------------------------------------
# Original Functions (Modified to call Numba kernels)
# -----------------------------------------------------------------------------

def _build_laplacian_SPD(dom, dx, dy, dz, k_mem_minus, k_mem_plus):
    """
    Assemble a sparse 7-point 3-D Laplacian with corrected Neumann BCs.
    This is now a wrapper around the fast Numba-jitted loop.
    """
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    
    # Pre-allocate arrays for Numba. Max possible entries is ~9*N.
    max_nnz = 9 * Nx * Ny * Nz
    row_idx = np.empty(max_nnz, dtype=np.int32)
    col_idx = np.empty(max_nnz, dtype=np.int32)
    diag_vals = np.empty(max_nnz, dtype=np.float64)

    # Call the fast Numba kernel to populate the arrays
    nnz = _build_laplacian_numba_loops(
        Nx, Ny, Nz, dx2, dy2, dz2, k_mem_minus, k_mem_plus,
        row_idx, col_idx, diag_vals
    )
    
    # Trim arrays to actual size and create the sparse matrix
    A = coo_matrix((diag_vals[:nnz], (row_idx[:nnz], col_idx[:nnz])),
                   shape=(Nx * Ny * Nz, Nx * Ny * Nz), dtype=np.float64).tocsr()
    return A

class AMGPoissonSolver:
    """Build once, reuse each Δt."""
    def __init__(self, dom, dx, dy, dz):
        Nz = dom.Nz if dom.Nz % 2 else dom.Nz + 1
        self.k_mem_minus = Nz // 2 - 1
        self.k_mem_plus = Nz // 2 + 1
        self.N = dom.Nx * dom.Ny * Nz
        self.shape = (dom.Nx, dom.Ny, Nz)
        self.dom = dom # Store for use in solve method

        # Build the corrected matrix
        A = _build_laplacian_SPD(dom, dx, dy, dz, self.k_mem_minus, self.k_mem_plus)
        self.ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric')

    def solve(self, Vm, V_applied):
        """
        Build RHS for current Vm and return φ (shape Nx,Ny,Nz).
        This now calls the fast Numba kernel for RHS assembly.
        """
        rhs = np.zeros(self.N, dtype=np.float64)
        
        # Call the fast Numba kernel
        _build_rhs_numba(rhs, Vm, self.dom.Nx, self.dom.Ny, self.shape[2],
                         self.k_mem_minus, self.k_mem_plus, V_applied)

        # Solve the system
        # Note: The order='F' is crucial because indexing uses Fortran/column-major ordering
        phi_flat = self.ml.solve(rhs, tol=1e-5, maxiter=1) # Reduced maxiter for speed
        return phi_flat.reshape(self.shape, order="F")

def create_grid(dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Creates the computational grid."""
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz if dom.Nz % 2 else dom.Nz + 1
    x = np.linspace(-dom.Lx / 2, dom.Lx / 2, Nx)
    y = np.linspace(-dom.Ly / 2, dom.Ly / 2, Ny)
    z = np.linspace(-dom.Lz / 2, dom.Lz / 2, Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz

def smooth_step(psi: np.ndarray) -> np.ndarray:
    """
    Computes a cubic Hermite smooth step function H(psi) = psi^2 * (3 - 2*psi).
    """
    return psi**2 * (3.0 - 2.0 * psi)

def build_phase_field(x: np.ndarray, y: np.ndarray, pore_radius: float, ell: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the phase field `psi` and the smoothed field `H`.
    """
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    psi = 0.5 * (1.0 + np.tanh((r - pore_radius) / (np.sqrt(2.0) * ell)))
    H = smooth_step(psi)
    return psi, H

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blends material properties based on the phase field and adds bath capacitance.
    """
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def estimate_time_step(G_m_map: np.ndarray, C_eff_map: np.ndarray, elec: Electrostatics, dom: Domain, solver: SolverParams) -> Tuple[float, float, int]:
    """
    Estimates a stable time step `dt` for the explicit time integration.
    """
    A = 2.0 * elec.sigma_e / dom.Lz
    rate_max = (A + np.max(G_m_map)) / np.min(C_eff_map)
    tau_local_min = 1.0 / rate_max
    dt = solver.dt_safety * tau_local_min
    tau_report = np.max(C_eff_map) / (A + np.max(G_m_map))
    total_time = solver.n_tau_total * tau_report
    nsteps = int(np.ceil(total_time / dt))
    return dt, total_time, nsteps

# -----------------------------------------------------------------------------
# Visualization and Main Driver
# -----------------------------------------------------------------------------

def plot_results(x: np.ndarray, y: np.ndarray, z: np.ndarray, Vm: np.ndarray, phi: np.ndarray, time_points: list[float], avg_Vm_vs_time: list[float], title_prefix: str) -> None:
    """Generates and displays plots of the simulation results."""
    fig = plt.figure(figsize=(24, 7))
    x_nm, y_nm, z_nm = x * 1e9, y * 1e9, z * 1e9
    time_ns = np.array(time_points) * 1e9

    fig.suptitle(title_prefix, fontsize=16, y=1.02)

    # 1) Average Vm vs. Time
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(time_ns, avg_Vm_vs_time, marker="o", linestyle="-")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Average Transmembrane Potential (V)")
    ax1.set_title("Average $V_m$ Charging Curve")
    ax1.grid(True, linestyle='--')

    # 2) Final Vm spatial map
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(Vm.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="magma")
    ax2.set_xlabel("x-position (nm)")
    ax2.set_ylabel("y-position (nm)")
    ax2.set_title("Final $V_m$ Distribution")
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="Voltage (V)")

    # 3) 2D slice of the full potential phi
    ax3 = fig.add_subplot(1, 3, 3)
    y_slice_idx = phi.shape[1] // 2
    im3 = ax3.imshow(phi[:, y_slice_idx, :].T, origin="lower", extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]], aspect="auto", cmap="viridis")
    ax3.axhline(0.0, color="r", linestyle="--", label="Membrane Plane")
    ax3.set_xlabel("x-position (nm)")
    ax3.set_ylabel("z-position (nm)")
    ax3.set_title(f"Final Potential $\\phi(x, y={y_nm[y_slice_idx]:.1f}, z)$")
    ax3.legend()
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Potential (V)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def simulate_membrane_charging(
    dom: Domain | None = None,
    props: MembraneProps | None = None,
    phase: PhaseFieldParams | None = None,
    elec: Electrostatics | None = None,
    solver: SolverParams | None = None,
    use_coupled_solver: bool = False,
) -> None:
    """Main driver for the membrane charging simulation."""
    # --- 1. Initialization ---
    dom = dom if dom is not None else Domain()
    props = props if props is not None else MembraneProps()
    phase = phase if phase is not None else PhaseFieldParams()
    elec = elec if elec is not None else Electrostatics()
    solver = solver if solver is not None else SolverParams()

    x, y, z, dx, dy, dz = create_grid(dom)
    phi_solver = AMGPoissonSolver(dom, dx, dy, dz)
    Nz_actual = z.shape[0]
    ell = phase.transition_thickness if phase.transition_thickness is not None else (2.0 * dx)

    # --- 2. Build Model ---
    psi, H = build_phase_field(x, y, phase.pore_radius, ell)
    G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom)
    dt, total_time, nsteps = estimate_time_step(G_m_map, C_eff_map, elec, dom, solver)

    if solver.surface_diffusion and solver.D_V == 0.0:
        solver.D_V = 0.05 * dx**2 / dt

    title_prefix = "Coupled 3D Solver (Numba Accelerated)" if use_coupled_solver else "Decoupled Solver (Numba Accelerated)"
    print(f"\n--- Membrane Charging Simulation ({title_prefix}) ---")
    print(f"Grid: {dom.Nx}x{dom.Ny}x{Nz_actual} (dx={dx*1e9:.2f} nm)")
    print(f"Pore Radius: {phase.pore_radius*1e9:.1f} nm, Transition: {ell*1e9:.2f} nm (~{ell/dx:.2f} dx)")
    tau_report = total_time / solver.n_tau_total
    print(f"System Time Constant (τ_lipid): {tau_report*1e9:.2f} ns")
    print(f"Simulation Time: {total_time*1e6:.2f} µs ({solver.n_tau_total:.1f} τ_lipid)")
    print(f"Stable Time Step (dt): {dt*1e9:.3f} ns ({nsteps} steps)")
    if solver.surface_diffusion:
        print(f"Surface Diffusion Enabled: D_V = {solver.D_V:.3e} m^2/s")
    print("-" * 50)

    # --- 3. Run Simulation ---
    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    phi = np.zeros((dom.Nx, dom.Ny, Nz_actual), dtype=float)
    time_points, avg_Vm_vs_time = [], []
    save_interval = max(1, nsteps // solver.save_frames)

    for n in range(nsteps):
        J_elec_coupled_arg = None
        if use_coupled_solver:
            # COUPLED: Solve phi first, calculate J, then update Vm
            phi = phi_solver.solve(Vm, elec.V_applied)
            J_elec_coupled_arg = calculate_J_from_phi(
                phi, phi_solver.k_mem_plus, elec.sigma_e, dz
            )

        # In both modes, we call the fast Numba update function
        Vm = update_vm_numba(
            Vm, G_m_map, C_eff_map, H, dt, dx,
            # Unpack dataclass attributes for Numba compatibility
            elec.sigma_e, elec.V_applied, dom.Lz,
            solver.surface_diffusion, solver.D_V,
            J_elec_coupled_arg
        )

        if not use_coupled_solver:
            # For the decoupled case, phi is just a diagnostic
            if n % save_interval == 0 or n == nsteps - 1:
                phi = phi_solver.solve(Vm, elec.V_applied)

        if n % save_interval == 0 or n == nsteps - 1:
            t = (n + 1) * dt
            avg_Vm = float(np.mean(Vm))
            time_points.append(t)
            avg_Vm_vs_time.append(avg_Vm)
            iters = phi_solver.ml.iteration_count if hasattr(phi_solver.ml, 'iteration_count') else 'N/A'
            print(f"Time: {t*1e9:8.2f} ns [{n+1:>{len(str(nsteps))}}/{nsteps}] | Avg Vm: {avg_Vm:.4f} V | AMG Iters: {iters}")

    # --- 4. Final calculation of phi for plotting ---
    phi = phi_solver.solve(Vm, elec.V_applied)

    # --- 5. Output Results ---
    print("\nSimulation finished. Generating plots...")
    plot_results(x, y, z, Vm, phi, time_points, avg_Vm_vs_time, title_prefix)

    filename = f"membrane_charging_{'coupled' if use_coupled_solver else 'decoupled'}_numba.npz"
    np.savez_compressed(
        filename, x=x, y=y, z=z, Vm=Vm, phi=phi, H=H,
        time_points=np.array(time_points),
        avg_Vm_vs_time=np.array(avg_Vm_vs_time)
    )
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # --- Run with the new, more rigorous COUPLED solver ---
    # This model is more physically accurate, but significantly more
    # computationally expensive. The Numba optimizations make it much more tractable.
    # We use the smaller grid for a quick demonstration.
    print("\n" + "="*80)
    print("NOW RUNNING THE NUMBA-ACCELERATED COUPLED SOLVER")
    print("The first run may be slow due to JIT compilation.")
    print("="*80)
    custom_domain = Domain(Nx=128, Ny=128, Nz=129) # Smaller grid for speed
    custom_solver_params = SolverParams(save_frames=40)
    simulate_membrane_charging(dom=custom_domain, solver=custom_solver_params, use_coupled_solver=True)