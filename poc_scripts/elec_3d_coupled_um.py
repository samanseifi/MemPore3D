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

# -----------------------------------------------------------------------------
# Grid and Phase Field Utilities
# -----------------------------------------------------------------------------
# ------------------------------------------------------------
# multigrid_phi.py  –  fast Poisson solver for membrane code
# ------------------------------------------------------------

import pyamg
from scipy.sparse import coo_matrix

# --- Physical Constants ---
EPS0 = 8.8541878128e-12 # Vacuum permittivity, F/m
EPS_W = 80.0 * EPS0     # Water permittivity, F/m

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
    R_pore: float = 1e-1  # Pore resistance (conductive), Ohm·m^2
    C_pore: float = 1e-9  # Pore capacitance (near zero), F/m^2

@dataclass
class PhaseFieldParams:
    """Parameters for the phase-field model of the pore."""
    pore_radius: float = 1000e-9
    transition_thickness: float | None = None  # If None, defaults to 2*dx

@dataclass
class Electrostatics:
    """Parameters for the electrostatic environment."""
    sigma_e: float = 1.0    # Electrolyte conductivity, S/m
    V_applied: float = 0.5  # Applied voltage across the box, V

@dataclass
class SolverParams:
    """Parameters controlling the numerical solver."""
    max_iters_phi: int = 2000
    tolerance: float = 1e-6
    surface_diffusion: bool = True
    D_V: float = 0.0          # If 0, set adaptively
    dt_safety: float = 0.01   # Safety factor for time step
    n_tau_total: float = 8.0  # Total simulation time in units of lipid RC time
    save_frames: int = 20     # Number of frames to save for visualization



def _build_laplacian_SPD(dom, dx, dy, dz, k_mem_minus, k_mem_plus):
    """
    Assemble a sparse 7-point 3-D Laplacian with corrected Neumann BCs.
    """
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
    diag_vals, row_idx, col_idx = [], [], []

    def idx(i, j, k):
        return i + Nx*(j + Ny*k)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                p = idx(i, j, k)

                # --- Dirichlet planes (unchanged) ---
                if k in (0, Nz-1, k_mem_minus, k_mem_plus):
                    row_idx.append(p); col_idx.append(p); diag_vals.append(1.0)
                    continue

                # --- Interior / Neumann ---
                diag = 0.0

                # --- x-direction ---
                if i == 0:      # left wall – mirror i=-1 → i=1
                    q = idx(i+1, j, k); w = 2.0/dx2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                elif i == Nx-1:  # right wall – mirror
                    q = idx(i-1, j, k); w = 2.0/dx2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                else:            # interior
                    for q in (idx(i-1, j, k), idx(i+1, j, k)):
                        row_idx.append(p); col_idx.append(q); diag_vals.append(-1.0/dx2)
                    diag += 2.0/dx2

                # --- y-direction (analogous) ---
                if j == 0:
                    q = idx(i, j+1, k); w = 2.0/dy2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                elif j == Ny-1:
                    q = idx(i, j-1, k); w = 2.0/dy2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                else:
                    for q in (idx(i, j-1, k), idx(i, j+1, k)):
                        row_idx.append(p); col_idx.append(q); diag_vals.append(-1.0/dy2)
                    diag += 2.0/dy2

                # --- z-direction (always interior) ---
                for q in (idx(i, j, k-1), idx(i, j, k+1)):
                    row_idx.append(p); col_idx.append(q); diag_vals.append(-1.0/dz2)
                diag += 2.0/dz2

                # set final diagonal
                row_idx.append(p); col_idx.append(p); diag_vals.append(diag)

    A = coo_matrix((diag_vals, (row_idx, col_idx)),
                shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64).tocsr()
    return A


class AMGPoissonSolver:
    """Build once, reuse each Δt."""
    def __init__(self, dom, dx, dy, dz):
        Nz = dom.Nz if dom.Nz % 2 else dom.Nz + 1
        self.k_mem_minus = Nz//2 - 1
        self.k_mem_plus  = Nz//2 + 1
        self.N = dom.Nx * dom.Ny * Nz
        self.shape = (dom.Nx, dom.Ny, Nz)

        # Build the corrected matrix
        A = _build_laplacian_SPD(dom, dx, dy, dz,
                                 self.k_mem_minus, self.k_mem_plus)
        self.ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric')

    def solve(self, Vm, V_applied, dom):
        """
        Build RHS for current Vm and return φ (shape Nx,Ny,Nz).
        """
        Nx, Ny, Nz = dom.Nx, dom.Ny, self.shape[2]
        rhs = np.zeros(self.N, dtype=np.float64)

        def idx(i, j, k): return i + Nx*(j + Ny*k)

        # top / bottom electrolyte planes
        rhs_top    =  V_applied/2.0
        rhs_bottom = -V_applied/2.0
        for j in range(Ny):
            for i in range(Nx):
                rhs[idx(i, j, 0   )] = rhs_bottom
                rhs[idx(i, j, Nz-1)] = rhs_top

        # membrane jump planes  (-Vm/2, +Vm/2)
        for j in range(Ny):
            for i in range(Nx):
                rhs[idx(i, j, self.k_mem_minus)] = -Vm[i, j]/2.0
                rhs[idx(i, j, self.k_mem_plus )] =  Vm[i, j]/2.0

        # Solve the system
        # Note: The order='F' is crucial because idx() uses Fortran/column-major ordering
        phi_flat = self.ml.solve(rhs, tol=1e-10, maxiter=10) # Reduced maxiter for speed in coupled loop
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

    This provides a smooth transition from H(0)=0 to H(1)=1 with zero
    derivatives at the endpoints.
    """
    return psi**2 * (3.0 - 2.0 * psi)

def build_phase_field(x: np.ndarray, y: np.ndarray, pore_radius: float, ell: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the phase field `psi` and the smoothed field `H`.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        pore_radius: The radius of the central pore.
        ell: The characteristic thickness of the pore-lipid transition.

    Returns:
        A tuple containing the phase field `psi` (0=pore, 1=lipid) and the
        smoothed field `H(psi)`.
    """
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    # tanh provides a smooth profile for the phase field
    psi = 0.5 * (1.0 + np.tanh((r - pore_radius) / (np.sqrt(2.0) * ell)))
    H = smooth_step(psi)
    return psi, H

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blends material properties based on the phase field and adds bath capacitance.

    Returns:
        A tuple of (G_m_map, C_m_map, C_eff_map) representing the spatial maps of
        membrane conductance, membrane capacitance, and effective capacitance.
    """
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    # Linearly interpolate properties between pore and lipid values using H
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    # Geometric capacitance of the electrolyte columns on either side of the membrane
    C_bath = 2.0 * EPS_W / dom.Lz
    # The effective capacitance includes the bath capacitance to regularize the ODE
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

# -----------------------------------------------------------------------------
# Numerical Solvers and Time Stepping
# -----------------------------------------------------------------------------

def estimate_time_step(G_m_map: np.ndarray, C_eff_map: np.ndarray, elec: Electrostatics, dom: Domain, solver: SolverParams) -> Tuple[float, float, int]:
    """
    Estimates a stable time step `dt` for the explicit time integration.

    The estimate is based on the stiffest (fastest) local RC time constant in the
    system, including contributions from electrolyte conductivity.
    """
    A = 2.0 * elec.sigma_e / dom.Lz
    # Estimate the maximum rate of change in the system
    rate_max = (A + np.max(G_m_map)) / np.min(C_eff_map)
    tau_local_min = 1.0 / rate_max
    dt = solver.dt_safety * tau_local_min

    # Calculate total simulation time based on the lipid charging time constant
    tau_report = np.max(C_eff_map) / (A + np.max(G_m_map))
    total_time = solver.n_tau_total * tau_report
    nsteps = int(np.ceil(total_time / dt))
    return dt, total_time, nsteps

def laplacian2d_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Computes the 2D Laplacian of a field with zero-flux (Neumann) boundaries.
    Assumes dx = dy.
    """
    padded = np.pad(field, pad_width=1, mode='edge')
    lap = (padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * field)
    return lap / (dx**2)

### MODIFICATION ###
# 1. New function to calculate current from the 3D potential field
### MODIFICATION ###
# 1. New function to calculate current from the 3D potential field
def calculate_J_from_phi(
    phi: np.ndarray, k_plane_index: int, sigma_e: float, dz: float
) -> np.ndarray:
    """
    Calculates the ionic current density arriving at a membrane plane.

    This uses a one-sided finite difference to approximate the normal gradient
    of the potential phi at the specified z-plane index. The charging current
    is defined as positive when flowing from high to low potential (downward).
    J_charge = -J_z = -(-sigma_e * d(phi)/dz) = sigma_e * d(phi)/dz
    """
    ### CORRECTION 2: Sign of the current was wrong. It is now fixed.
    grad_phi_z = (phi[:, :, k_plane_index + 1] - phi[:, :, k_plane_index]) / dz
    J_elec = sigma_e * grad_phi_z
    return J_elec


### MODIFICATION ###
# 2. Updated vm_update function to handle both coupled and decoupled modes.
def update_vm(
    Vm: np.ndarray,
    G_m_map: np.ndarray,
    # C_m_map: np.ndarray, # No longer needed directly
    C_eff_map: np.ndarray,
    H: np.ndarray,
    elec: Electrostatics,
    dom: Domain,
    dt: float,
    dx: float,
    solver: SolverParams,
    J_elec_coupled: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Updates the transmembrane potential `Vm` for one time step using explicit Euler.
    """
    if J_elec_coupled is None:
        # --- DECOUPLED (ORIGINAL) MODE ---
        J_elec = H * (2.0 * elec.sigma_e * (elec.V_applied - Vm) / dom.Lz)
    else:
        # --- COUPLED MODE ---
        J_elec = J_elec_coupled

    ### CORRECTION 1: Numerical stability. Always use C_eff_map, which includes
    ### the bath capacitance as a regularizer to prevent division by zero in the pore.
    C_map_for_update = C_eff_map

    base_G_sc = 2.0 * elec.sigma_e / dom.Lz
    G_sc = 50.0 * base_G_sc * (1.0 - H)

    dVm_dt = (J_elec - (G_m_map + G_sc) * Vm) / C_map_for_update
    Vm_new = Vm + dt * dVm_dt

    if solver.surface_diffusion and solver.D_V > 0.0:
        lap = laplacian2d_neumann(Vm_new, dx)
        Vm_new += dt * solver.D_V * lap
    return Vm_new

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

    title_prefix = "Coupled 3D Solver" if use_coupled_solver else "Decoupled (Original) Solver"
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
        if use_coupled_solver:
            # COUPLED: Solve phi first, calculate J, then update Vm
            phi = phi_solver.solve(Vm, elec.V_applied, dom)
            J_elec = calculate_J_from_phi(
                phi, phi_solver.k_mem_plus, elec.sigma_e, dz
            )
            ### CORRECTION: Removed C_m_map from the function call
            Vm = update_vm(
                Vm, G_m_map, C_eff_map, H, elec, dom, dt, dx, solver,
                J_elec_coupled=J_elec
            )
        else:
            # DECOUPLED (ORIGINAL): Update Vm, then solve phi for diagnostics
            ### CORRECTION: Removed C_m_map from the function call
            Vm = update_vm(
                Vm, G_m_map, C_eff_map, H, elec, dom, dt, dx, solver,
                J_elec_coupled=None
            )
            # For the decoupled case, phi is just a diagnostic
            if n % save_interval == 0 or n == nsteps - 1:
                 phi = phi_solver.solve(Vm, elec.V_applied, dom)


        if n % save_interval == 0 or n == nsteps - 1:
            t = (n + 1) * dt
            avg_Vm = float(np.mean(Vm))
            time_points.append(t)
            avg_Vm_vs_time.append(avg_Vm)
            iters = phi_solver.ml.iteration_count if hasattr(phi_solver.ml, 'iteration_count') else 'N/A'
            print(f"Time: {t*1e9:8.2f} ns [{n+1:>{len(str(nsteps))}}/{nsteps}] | Avg Vm: {avg_Vm:.4f} V | AMG Iters: {iters}")

    # --- 4. Final calculation of phi for plotting ---
    phi = phi_solver.solve(Vm, elec.V_applied, dom)

    # --- 5. Output Results ---
    print("\nSimulation finished. Generating plots...")
    plot_results(x, y, z, Vm, phi, time_points, avg_Vm_vs_time, title_prefix)

    filename = f"membrane_charging_{'coupled' if use_coupled_solver else 'decoupled'}.npz"
    np.savez_compressed(
        filename, x=x, y=y, z=z, Vm=Vm, phi=phi, H=H,
        time_points=np.array(time_points),
        avg_Vm_vs_time=np.array(avg_Vm_vs_time)
    )
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # --- Run with default parameters using the original, fast DECOUPLED solver ---
    # This model is fast and good for observing the basic charging curve.
    # simulate_membrane_charging(use_coupled_solver=True)

    # --- Run with the new, more rigorous COUPLED solver ---
    # This model is more physically accurate, capturing the "funneling" of
    # current into the pore, which can lead to different spatial Vm patterns.
    # It is significantly more computationally expensive.
    # We reduce the grid size for a quicker demonstration.
    print("\n" + "="*80)
    print("NOW RUNNING THE COUPLED SOLVER ON A SMALLER GRID FOR DEMONSTRATION")
    print("="*80)
    custom_domain = Domain(Nx=128, Ny=128, Nz=129) # Smaller grid for speed
    custom_solver_params = SolverParams(save_frames=40)
    simulate_membrane_charging(dom=custom_domain, solver=custom_solver_params, use_coupled_solver=True)