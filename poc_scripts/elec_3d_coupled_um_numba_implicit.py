"""
Refactored membrane charging simulation with a diffuse pore transition.

This script models the charging of a lipid membrane containing a central pore,
using a phase-field approach to smoothly transition between lipid and pore properties.

This version uses a fully coupled, semi-implicit solver. The electrolyte
potential (phi) is solved using a 3D Laplace equation, and the transmembrane
potential (Vm) is updated using a numerically stable semi-implicit Backward
Euler scheme. This approach allows for much larger time steps compared to
explicit methods, resulting in a significant overall speedup for long
simulations while maintaining physical accuracy.

Key Features
------------
- Phase-field (psi): Represents lipid (psi=1) to pore (psi=0) state.
- Smooth Blending: Electrical properties (conductance, capacitance) are blended
  using a cubic Hermite polynomial H(psi) = psi^2 * (3 - 2*psi).
- Coupled Solver: Solves the full 3D Laplace equation for the electrolyte
  potential and the 2D reaction-diffusion equation for the transmembrane potential.
- Implicit Vm Solver: A `ImplicitVMSolver` class handles the stable time
  evolution of Vm, allowing for large dt and fast simulations.
- Numba Optimization: Core computational loops are JIT-compiled for performance.

Original Author: Saman Seifi
Refactored by: Gemini (August 11, 2025)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import numba
import pyamg
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import factorized

# --- Physical Constants ---
EPS0 = 8.8541878128e-12  # Vacuum permittivity, F/m
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
    dt_safety: float = 0.01    # Safety factor for base time step calculation
    n_tau_total: float = 8.0   # Total simulation time in units of lipid RC time
    save_frames: int = 20      # Number of frames to save for visualization
    implicit_dt_multiplier: float = 100.0 # Factor to increase dt for implicit Vm solver

# -----------------------------------------------------------------------------
# Numba-JIT Compiled Core Routines
# -----------------------------------------------------------------------------

@numba.njit(cache=True)
def _build_laplacian_numba_loops(Nx, Ny, Nz, dx2, dy2, dz2, k_mem_minus, k_mem_plus,
                                 row_idx, col_idx, diag_vals):
    """Numba-accelerated inner loop for assembling the sparse 3D Laplacian matrix."""
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
                if k in (0, Nz - 1, k_mem_minus, k_mem_plus):
                    add_entry(p, p, 1.0)
                    continue
                diag = 0.0
                if i == 0:
                    q = (i + 1) + Nx * (j + Ny * k)
                    add_entry(p, q, -2.0 / dx2); diag += 2.0 / dx2
                elif i == Nx - 1:
                    q = (i - 1) + Nx * (j + Ny * k)
                    add_entry(p, q, -2.0 / dx2); diag += 2.0 / dx2
                else:
                    q1 = (i - 1) + Nx * (j + Ny * k); add_entry(p, q1, -1.0 / dx2)
                    q2 = (i + 1) + Nx * (j + Ny * k); add_entry(p, q2, -1.0 / dx2)
                    diag += 2.0 / dx2
                if j == 0:
                    q = i + Nx * ((j + 1) + Ny * k)
                    add_entry(p, q, -2.0 / dy2); diag += 2.0 / dy2
                elif j == Ny - 1:
                    q = i + Nx * ((j - 1) + Ny * k)
                    add_entry(p, q, -2.0 / dy2); diag += 2.0 / dy2
                else:
                    q1 = i + Nx * ((j - 1) + Ny * k); add_entry(p, q1, -1.0 / dy2)
                    q2 = i + Nx * ((j + 1) + Ny * k); add_entry(p, q2, -1.0 / dy2)
                    diag += 2.0 / dy2
                q1 = i + Nx * (j + Ny * (k - 1)); add_entry(p, q1, -1.0 / dz2)
                q2 = i + Nx * (j + Ny * (k + 1)); add_entry(p, q2, -1.0 / dz2)
                diag += 2.0 / dz2
                add_entry(p, p, diag)
    return count

@numba.njit(cache=True)
def _build_implicit_vm_operator_numba(Nx, Ny, C_eff_flat, G_total_flat, dt, dx, D_V,
                                      row_idx, col_idx, data_vals):
    """Numba-accelerated loop for building the 2D implicit Vm operator matrix."""
    count = 0
    dx2 = dx * dx
    lap_diag_term = 4.0 * dt * D_V / dx2
    lap_offdiag_term = -dt * D_V / dx2

    def add_entry(r, c, v):
        nonlocal count
        row_idx[count] = r
        col_idx[count] = c
        data_vals[count] = v
        count += 1

    for j in range(Ny):
        for i in range(Nx):
            p = i + Nx * j
            diag_val = C_eff_flat[p] + dt * G_total_flat[p]
            
            # X-neighbors
            if i > 0: add_entry(p, p - 1, lap_offdiag_term)
            else: diag_val += lap_offdiag_term # Neumann BC
            
            if i < Nx - 1: add_entry(p, p + 1, lap_offdiag_term)
            else: diag_val += lap_offdiag_term
            
            # Y-neighbors
            if j > 0: add_entry(p, p - Nx, lap_offdiag_term)
            else: diag_val += lap_offdiag_term
            
            if j < Ny - 1: add_entry(p, p + Nx, lap_offdiag_term)
            else: diag_val += lap_offdiag_term

            diag_val += lap_diag_term
            add_entry(p, p, diag_val)
    return count

@numba.njit(cache=True)
def _build_rhs_numba(rhs, Vm, Nx, Ny, Nz, k_mem_minus, k_mem_plus, V_applied):
    """Numba-accelerated function to build the RHS vector for the Poisson solve."""
    rhs_top, rhs_bottom = V_applied / 2.0, -V_applied / 2.0
    for j in range(Ny):
        for i in range(Nx):
            rhs[i + Nx * (j + Ny * 0)] = rhs_bottom
            rhs[i + Nx * (j + Ny * (Nz - 1))] = rhs_top
            rhs[i + Nx * (j + Ny * k_mem_minus)] = -Vm[i, j] / 2.0
            rhs[i + Nx * (j + Ny * k_mem_plus)] = Vm[i, j] / 2.0
    return rhs

@numba.njit(cache=True)
def calculate_J_from_phi(phi, k_plane_index, sigma_e, dz):
    """Numba-jitted calculation of ionic current density from 3D potential."""
    grad_phi_z = (phi[:, :, k_plane_index + 1] - phi[:, :, k_plane_index]) / dz
    return sigma_e * grad_phi_z

# -----------------------------------------------------------------------------
# Solvers and High-Level Routines
# -----------------------------------------------------------------------------

def _build_laplacian_SPD(dom, dx, dy, dz, k_mem_minus, k_mem_plus):
    """Wrapper to assemble the 3D Laplacian matrix using the Numba kernel."""
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    max_nnz = 9 * Nx * Ny * Nz
    row_idx, col_idx, diag_vals = np.empty(max_nnz, dtype=np.int32), np.empty(max_nnz, dtype=np.int32), np.empty(max_nnz, dtype=np.float64)
    nnz = _build_laplacian_numba_loops(Nx, Ny, Nz, dx*dx, dy*dy, dz*dz, k_mem_minus, k_mem_plus, row_idx, col_idx, diag_vals)
    A = coo_matrix((diag_vals[:nnz], (row_idx[:nnz], col_idx[:nnz])), shape=(Nx*Ny*Nz, Nx*Ny*Nz)).tocsr()
    return A

def build_vm_implicit_operator(dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, Lz):
    """Builds the sparse matrix for the implicit Vm solve using a Numba kernel."""
    Nx, Ny = dom.Nx, dom.Ny
    base_G_sc = 2.0 * sigma_e / Lz
    G_sc_map = 50.0 * base_G_sc * (1.0 - H)
    G_total_map = G_m_map + G_sc_map

    C_eff_flat = C_eff_map.flatten(order='C')
    G_total_flat = G_total_map.flatten(order='C')
    
    max_nnz = 5 * Nx * Ny
    row_idx, col_idx, data_vals = np.empty(max_nnz, dtype=np.int32), np.empty(max_nnz, dtype=np.int32), np.empty(max_nnz, dtype=np.float64)

    nnz = _build_implicit_vm_operator_numba(
        Nx, Ny, C_eff_flat, G_total_flat, dt, dx, D_V, row_idx, col_idx, data_vals
    )
    A = coo_matrix((data_vals[:nnz], (row_idx[:nnz], col_idx[:nnz])), shape=(Nx*Ny, Nx*Ny)).tocsc()
    return A

class AMGPoissonSolver:
    """AMG solver for the 3D electrolyte potential phi."""
    def __init__(self, dom, dx, dy, dz):
        self.Nz_actual = dom.Nz if dom.Nz % 2 else dom.Nz + 1
        self.k_mem_minus = self.Nz_actual // 2 - 1
        self.k_mem_plus = self.Nz_actual // 2 + 1
        self.N = dom.Nx * dom.Ny * self.Nz_actual
        self.shape = (dom.Nx, dom.Ny, self.Nz_actual)
        self.dom = dom
        A = _build_laplacian_SPD(dom, dx, dy, dz, self.k_mem_minus, self.k_mem_plus)
        self.ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric')

    def solve(self, Vm, V_applied):
        """Builds RHS and solves for phi using the pre-built AMG solver."""
        rhs = np.zeros(self.N, dtype=np.float64)
        _build_rhs_numba(rhs, Vm, self.dom.Nx, self.dom.Ny, self.shape[2],
                         self.k_mem_minus, self.k_mem_plus, V_applied)
        phi_flat = self.ml.solve(rhs, tol=1e-8, maxiter=5)
        return phi_flat.reshape(self.shape, order="F")

class ImplicitVMSolver:
    """Solver for the semi-implicit update of the 2D transmembrane potential Vm."""
    def __init__(self, dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, Lz):
        self.Nx, self.Ny = dom.Nx, dom.Ny
        print("Building and factorizing the implicit Vm operator... ", end="")
        A = build_vm_implicit_operator(dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, Lz)
        self.solver = factorized(A)
        print("Done.")
        
    def solve(self, b_flat):
        """Solves the system Ax = b using the pre-factorized operator."""
        x_flat = self.solver(b_flat)
        return x_flat.reshape((self.Nx, self.Ny), order='C')

def create_grid(dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Creates the computational grid."""
    dom.Nz = dom.Nz if dom.Nz % 2 else dom.Nz + 1
    x = np.linspace(-dom.Lx / 2, dom.Lx / 2, dom.Nx)
    y = np.linspace(-dom.Ly / 2, dom.Ly / 2, dom.Ny)
    z = np.linspace(-dom.Lz / 2, dom.Lz / 2, dom.Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz

def smooth_step(psi: np.ndarray) -> np.ndarray:
    """Computes a cubic Hermite smooth step function H(psi)."""
    return psi**2 * (3.0 - 2.0 * psi)

def build_phase_field(x: np.ndarray, y: np.ndarray, pore_radius: float, ell: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs the phase field `psi` and the smoothed field `H`."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    psi = 0.5 * (1.0 + np.tanh((r - pore_radius) / (np.sqrt(2.0) * ell)))
    return psi, smooth_step(psi)

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends material properties based on the phase field."""
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    C_bath = 0 # 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def estimate_base_time_step(G_m_map: np.ndarray, C_eff_map: np.ndarray, elec: Electrostatics, dom: Domain, solver: SolverParams) -> Tuple[float, float, int]:
    """Estimates a base time step `dt` from explicit stability criteria."""
    A = 2.0 * elec.sigma_e / dom.Lz
    rate_max = (A + np.max(G_m_map)) / np.min(C_eff_map)
    tau_local_min = 1.0 / rate_max
    dt_base = solver.dt_safety * tau_local_min
    tau_report = np.max(C_eff_map) / (A + np.max(G_m_map))
    total_time = solver.n_tau_total * tau_report
    nsteps_base = int(np.ceil(total_time / dt_base))
    return dt_base, total_time, nsteps_base

# -----------------------------------------------------------------------------
# Visualization and Main Driver
# -----------------------------------------------------------------------------

def plot_results(x, y, z, Vm, phi, time_points, avg_Vm_vs_time, title_prefix, elapsed_time):
    """Generates and displays plots of the simulation results."""
    fig = plt.figure(figsize=(24, 7))
    x_nm, y_nm, z_nm = x * 1e9, y * 1e9, z * 1e9
    time_ns = np.array(time_points) * 1e9

    fig.suptitle(f"{title_prefix}\n(Total Wall Clock Time: {elapsed_time:.2f} s)", fontsize=16, y=1.03)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(time_ns, avg_Vm_vs_time, marker="o", linestyle="-")
    ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Average Transmembrane Potential (V)")
    ax1.set_title("Average $V_m$ Charging Curve"); ax1.grid(True, linestyle='--')

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(Vm.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="magma")
    ax2.set_xlabel("x (nm)"); ax2.set_ylabel("y (nm)"); ax2.set_title("Final $V_m$ Distribution")
    cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="Voltage (V)")

    ax3 = fig.add_subplot(1, 3, 3)
    y_slice_idx = phi.shape[1] // 2
    im3 = ax3.imshow(phi[:, y_slice_idx, :].T, origin="lower", extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]], aspect="auto", cmap="viridis")
    ax3.axhline(0.0, color="r", linestyle="--", label="Membrane Plane")
    ax3.set_xlabel("x (nm)"); ax3.set_ylabel("z (nm)"); ax3.set_title(f"Final Potential $\\phi(x, y={y_nm[y_slice_idx]:.1f}, z)$")
    ax3.legend()
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Potential (V)")

    plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()


def simulate_membrane_charging(dom_in: Domain | None = None, props: MembraneProps | None = None,
                               phase: PhaseFieldParams | None = None, elec: Electrostatics | None = None,
                               solver: SolverParams | None = None) -> None:
    """Main driver for the membrane charging simulation."""
    # --- 1. Initialization ---
    start_time = time.time()
    dom = dom_in if dom_in is not None else Domain()
    props = props if props is not None else MembraneProps()
    phase = phase if phase is not None else PhaseFieldParams()
    elec = elec if elec is not None else Electrostatics()
    solver = solver if solver is not None else SolverParams()

    x, y, z, dx, dy, dz = create_grid(dom)
    ell = phase.transition_thickness if phase.transition_thickness is not None else (2.0 * dx)
    
    # --- 2. Build Model ---
    psi, H = build_phase_field(x, y, phase.pore_radius, ell)
    G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom)
    dt_base, total_time, nsteps_base = estimate_base_time_step(G_m_map, C_eff_map, elec, dom, solver)

    # --- 3. Configure Solver and Time Stepping ---
    title_prefix = "Coupled (Semi-Implicit Vm)"
    dt = dt_base * solver.implicit_dt_multiplier
    nsteps = int(np.ceil(total_time / dt))

    if solver.surface_diffusion and solver.D_V == 0.0:
        solver.D_V = 0.0 # 0.05 * dx**2 / dt

    print(f"\n--- Membrane Charging Simulation ({title_prefix}) ---")
    print(f"Grid: {dom.Nx}x{dom.Ny}x{dom.Nz} (dx={dx*1e9:.2f} nm)")
    tau_report = total_time / solver.n_tau_total
    print(f"System Time Constant (τ_lipid): {tau_report*1e9:.2f} ns")
    print(f"Simulation Time: {total_time*1e6:.2f} µs ({solver.n_tau_total:.1f} τ_lipid)")
    print(f"Implicit dt: {dt*1e9:.3f} ns | Base dt would be: {dt_base*1e9:.3f} ns")
    print(f"Total steps: {nsteps} (vs {nsteps_base} for explicit)")
    
    phi_solver = AMGPoissonSolver(dom, dx, dy, dz)
    vm_implicit_solver = ImplicitVMSolver(
        dom, C_eff_map, G_m_map, H, dt, dx, solver.D_V, elec.sigma_e, dom.Lz
    )
    print("-" * 60)

    # --- 4. Run Simulation ---
    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    time_points, avg_Vm_vs_time = [], []
    save_interval = max(1, nsteps // solver.save_frames)

    for n in range(nsteps):
        # Solve for 3D electrolyte potential
        phi = phi_solver.solve(Vm, elec.V_applied)
        J_elec_coupled = calculate_J_from_phi(phi, phi_solver.k_mem_plus, elec.sigma_e, dz)

        # Perform semi-implicit update for Vm
        b_rhs_2d = C_eff_map * Vm + dt * J_elec_coupled
        Vm = vm_implicit_solver.solve(b_rhs_2d.flatten(order='C'))

        # Save data at specified intervals
        if n % save_interval == 0 or n == nsteps - 1:
            t = (n + 1) * dt
            avg_Vm = float(np.mean(Vm))
            time_points.append(t)
            avg_Vm_vs_time.append(avg_Vm)
            print(f"Time: {t*1e9:8.2f} ns [{n+1:>{len(str(nsteps))}}/{nsteps}] | Avg Vm: {avg_Vm:.4f} V")

    # --- 5. Output Results ---
    phi = phi_solver.solve(Vm, elec.V_applied) # Final potential field
    elapsed_time = time.time() - start_time
    print(f"\nSimulation finished in {elapsed_time:.2f} seconds.")
    
    # Optional: Plot results directly
    # plot_results(x, y, z, Vm, phi, time_points, avg_Vm_vs_time, title_prefix, elapsed_time)

    # Save results to a file
    filename = "membrane_charging_coupled_implicit.npz"
    np.savez_compressed(
        filename, 
        x=x, y=y, z=z, 
        Vm=Vm, phi=phi, H=H,
        time_points=np.array(time_points),
        avg_Vm_vs_time=np.array(avg_Vm_vs_time),
        domain_params=dom_in,
        solver_params=solver
    )
    print(f"Numerical results saved to {filename}")
    
if __name__ == "__main__":
    # --- Configuration for a standard run ---
    custom_domain = Domain(Nx=64, Ny=64, Nz=64)
    # Increase the dt multiplier for a more dramatic speedup demonstration
    fast_solver = SolverParams(save_frames=40, implicit_dt_multiplier=200.0)

    print("\n" + "="*80)
    print("RUNNING THE COUPLED SOLVER (SEMI-IMPLICIT Vm UPDATE)")
    print("The first run may be slow due to JIT compilation.")
    print("="*80)
    
    simulate_membrane_charging(dom_in=custom_domain, solver=fast_solver)
