"""
Fully coupled simulation of membrane electroporation.

This script models the dynamic evolution of a lipid membrane pore under the
influence of both intrinsic line tension and an applied electric field.

It combines two physical models into a single, coupled simulation:
1.  **3D Electrostatics (from Script 1):** Solves the full 3D Laplace equation
    for the electrolyte potential at each time step. This provides an accurate
    transmembrane potential (Vm) that accounts for current funneling into the
    pore.
2.  **2D Phase-Field Dynamics (from Script 2):** Evolves a phase-field (psi)
    representing the pore geometry. The evolution is driven by:
    a) Line tension (tending to close the pore).
    b) Electroporative force from Vm (tending to expand the pore).

The coupling is two-way:
- The evolving `psi` field continuously updates the membrane's electrical properties.
- The resulting `Vm` field continuously provides the force that evolves `psi`.

Key Features
------------
- Phase-field (psi): Represents lipid (psi=1) to pore (psi=0) state.
- Allen-Cahn Equation: Governs the evolution of the phase-field.
- Spectral Solver: Uses FFTs for efficient and stable time-stepping of the
  phase-field dynamics.
- AMG Poisson Solver: Uses Algebraic Multigrid for fast solving of the 3D
  electrolyte potential.
- Electromechanical Coupling: A new term based on capacitive energy density
  (0.5 * C * Vm^2) is added to the phase-field chemical potential.

Original Electrostatics Author: Saman Seifi
Phase-Field Dynamics Author: User provided
Coupled Electroporation Model by: Gemini (August 7, 2025)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    Lx: float = 100e-9
    Ly: float = 100e-9
    Lz: float = 200e-9
    Nx: int = 128
    Ny: int = 128
    Nz: int = 129  # Will be forced to an odd number

@dataclass
class MembraneProps:
    """Intrinsic electrical properties of the membrane components."""
    R_lipid: float = 1e7   # Lipid resistance, Ohm·m^2
    C_lipid: float = 1e-2  # Lipid capacitance, F/m^2
    R_pore: float = 1e-1   # Pore resistance (conductive), Ohm·m^2
    C_pore: float = 1e-9   # Pore capacitance (near zero), F/m^2

### NEW ###
# Dataclass to hold parameters for the phase-field dynamics.
@dataclass
class PhaseFieldDynamics:
    """Parameters for the phase-field model dynamics (pore evolution)."""
    initial_pore_radius: float = 10e-9 # m
    transition_thickness: float | None = None # If None, defaults to 2*dx
    gamma: float = 1.5e-11   # J/m, line tension
    sigma: float = 0.8e-11   # J/m^2, area-dependent energy (usually 0)
    M: float = 2.0e7         # m^2/(J*s), phase-field mobility

@dataclass
class Electrostatics:
    """Parameters for the electrostatic environment."""
    sigma_e: float = 1.0     # Electrolyte conductivity, S/m
    V_applied: float = 0.5   # Applied voltage across the box, V

@dataclass
class SolverParams:
    """Parameters controlling the numerical solver."""
    dt: float | None = None   # If None, will be estimated
    total_time: float = 100e-9 # s, total simulation time
    save_frames: int = 50     # Number of frames to save for visualization
    surface_diffusion: bool = True
    D_V: float = 0.0          # If 0, set adaptively

# -----------------------------------------------------------------------------
# 3D Poisson Solver (Unchanged from original script)
# -----------------------------------------------------------------------------

def _build_laplacian_SPD(dom, dx, dy, dz, k_mem_minus, k_mem_plus):
    """Assembles a sparse 7-point 3-D Laplacian with corrected Neumann BCs."""
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
    diag_vals, row_idx, col_idx = [], [], []

    def idx(i, j, k):
        return i + Nx*(j + Ny*k)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                p = idx(i, j, k)
                if k in (0, Nz-1, k_mem_minus, k_mem_plus):
                    row_idx.append(p); col_idx.append(p); diag_vals.append(1.0)
                    continue
                diag = 0.0
                if i == 0:
                    q = idx(i+1, j, k); w = 2.0/dx2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                elif i == Nx-1:
                    q = idx(i-1, j, k); w = 2.0/dx2
                    row_idx += [p, p]; col_idx += [p, q]; diag_vals += [w, -w]
                else:
                    for q in (idx(i-1, j, k), idx(i+1, j, k)):
                        row_idx.append(p); col_idx.append(q); diag_vals.append(-1.0/dx2)
                    diag += 2.0/dx2
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
                for q in (idx(i, j, k-1), idx(i, j, k+1)):
                    row_idx.append(p); col_idx.append(q); diag_vals.append(-1.0/dz2)
                diag += 2.0/dz2
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
        A = _build_laplacian_SPD(dom, dx, dy, dz, self.k_mem_minus, self.k_mem_plus)
        self.ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric')

    def solve(self, Vm, V_applied, dom):
        Nx, Ny, Nz = dom.Nx, dom.Ny, self.shape[2]
        rhs = np.zeros(self.N, dtype=np.float64)
        def idx(i, j, k): return i + Nx*(j + Ny*k)
        rhs_top, rhs_bottom = V_applied/2.0, -V_applied/2.0
        for j in range(Ny):
            for i in range(Nx):
                rhs[idx(i, j, 0)] = rhs_bottom
                rhs[idx(i, j, Nz-1)] = rhs_top
                rhs[idx(i, j, self.k_mem_minus)] = -Vm[i, j]/2.0
                rhs[idx(i, j, self.k_mem_plus )] =  Vm[i, j]/2.0
        phi_flat = self.ml.solve(rhs, tol=1e-10, maxiter=20)
        return phi_flat.reshape(self.shape, order="F")

# -----------------------------------------------------------------------------
# Grid, Phase Field, and Property Blending
# -----------------------------------------------------------------------------

def create_grid(dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Creates the computational grid."""
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz if dom.Nz % 2 else dom.Nz + 1
    x = np.linspace(-dom.Lx / 2, dom.Lx / 2, Nx)
    y = np.linspace(-dom.Ly / 2, dom.Ly / 2, Ny)
    z = np.linspace(-dom.Lz / 2, dom.Lz / 2, Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz

def initialize_phase_field(x: np.ndarray, y: np.ndarray, radius: float, ell: float) -> np.ndarray:
    """Constructs the initial phase field `psi` (0=pore, 1=lipid)."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    psi = 0.5 * (1.0 + np.tanh((r - radius) / (np.sqrt(2.0) * ell)))
    return psi

def smooth_step(psi: np.ndarray) -> np.ndarray:
    """Cubic Hermite smooth step function H(psi) = psi^2 * (3 - 2*psi)."""
    return psi**2 * (3.0 - 2.0 * psi)

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain, elec: Electrostatics) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends material properties based on the phase field."""
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

# -----------------------------------------------------------------------------
# Numerical Solvers and Time Stepping
# -----------------------------------------------------------------------------

def estimate_rc_time_step(G_m_map: np.ndarray, C_eff_map: np.ndarray, elec: Electrostatics, dom: Domain) -> float:
    """Estimates a stable time step `dt` based on the RC time constant."""
    A = 2.0 * elec.sigma_e / dom.Lz
    rate_max = (A + np.max(G_m_map)) / np.min(C_eff_map)
    tau_local_min = 1.0 / rate_max
    return 0.1 * tau_local_min # Use a safety factor

def laplacian2d_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """Computes the 2D Laplacian of a field with zero-flux (Neumann) boundaries."""
    padded = np.pad(field, pad_width=1, mode='edge')
    lap = (padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * field)
    return lap / (dx**2)

def calculate_J_from_phi(phi: np.ndarray, k_plane_index: int, sigma_e: float, dz: float) -> np.ndarray:
    """Calculates the ionic current density arriving at a membrane plane."""
    grad_phi_z = (phi[:, :, k_plane_index + 1] - phi[:, :, k_plane_index]) / dz
    return sigma_e * grad_phi_z

def update_vm(
    Vm: np.ndarray, G_m_map: np.ndarray, C_eff_map: np.ndarray, H: np.ndarray,
    elec: Electrostatics, dom: Domain, dt: float, dx: float,
    solver: SolverParams, J_elec_coupled: np.ndarray
) -> np.ndarray:
    """Updates the transmembrane potential `Vm` for one time step."""
    base_G_sc = 2.0 * elec.sigma_e / dom.Lz
    G_sc = 0.0 # 50.0 * base_G_sc * (1.0 - H)
    dVm_dt = (J_elec_coupled - (G_m_map + G_sc) * Vm) / C_eff_map
    Vm_new = Vm + dt * dVm_dt
    if solver.surface_diffusion and solver.D_V > 0.0:
        lap = laplacian2d_neumann(Vm_new, dx)
        Vm_new += dt * solver.D_V * lap
    return Vm_new

### NEW ###
# The phase-field evolution functions from your second script, adapted to be
# part of a class for better organization.
class PhaseFieldSolver:
    """Handles the evolution of the 2D phase-field `psi`."""
    def __init__(self, dom: Domain, dyn: PhaseFieldDynamics, dx: float, dt: float):
        self.dyn = dyn
        self.Cg = np.sqrt(2.0) / 12.0
        self.dx = dx

        # Spectral (FFT) setup
        kx = 2.0 * np.pi * np.fft.fftfreq(dom.Nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(dom.Ny, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        self.denom = 1.0 + dt * self.dyn.M * (self.dyn.gamma / self.Cg) * self.dyn.transition_thickness * K2

    @staticmethod
    def g(psi): return 0.25 * psi**2 * (1.0 - psi)**2
    @staticmethod
    def gp(psi): return 0.5 * psi * (1.0 - psi) * (1.0 - 2.0 * psi)
    @staticmethod
    def dH(psi): return 6.0 * psi * (1.0 - psi)

    ### MODIFIED METHOD ###
    def update_psi(self, psi: np.ndarray, sigma_map: np.ndarray, dt: float) -> np.ndarray:
        """
        Advances the phase-field `psi` by one time step using a total
        surface tension map (mechanical + electrical).
        """
        # --- Chemical potential from line tension ---
        mu_chem = (self.dyn.gamma / self.Cg) * (self.gp(psi) / self.dyn.transition_thickness)

        # --- Chemical potential from total surface tension ---
        # This now includes the electrical contribution calculated in the main loop.
        mu_surf = sigma_map * self.dH(psi)

        # --- Combine terms and perform time step in Fourier space ---
        rhs = -self.dyn.M * (mu_chem + mu_surf)
        psi_hat = np.fft.fft2(psi)
        rhs_hat = np.fft.fft2(rhs)
        psi_hat_new = (psi_hat + dt * rhs_hat) / self.denom

        # Transform back to real space and clip for stability
        psi_new = np.fft.ifft2(psi_hat_new).real
        return np.clip(psi_new, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Visualization and Main Driver
# -----------------------------------------------------------------------------

def plot_results(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, Vm: np.ndarray, psi: np.ndarray, phi: np.ndarray,
    diag_data: dict, title_prefix: str
) -> None:
    """Generates and displays plots of the simulation results."""
    fig = plt.figure(figsize=(24, 6))
    x_nm, y_nm, z_nm = x * 1e9, y * 1e9, z * 1e9
    time_us = np.array(diag_data['time']) * 1e6

    fig.suptitle(title_prefix, fontsize=16, y=1.02)

    # 1) Pore Radius vs. Time
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(time_us, diag_data['radius_nm'], marker=".", linestyle="-")
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Effective Pore Radius (nm)")
    ax1.set_title("Pore Radius Evolution")
    ax1.grid(True, linestyle='--')

    # 2) Final Psi field
    ax2 = fig.add_subplot(1, 4, 2)
    im2 = ax2.imshow(psi.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="viridis_r")
    ax2.set_xlabel("x-position (nm)")
    ax2.set_ylabel("y-position (nm)")
    ax2.set_title("Final Phase Field $\\psi$ (Pore Shape)")
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="$\\psi$ (0=Pore, 1=Lipid)")

    # 3) Final Vm spatial map
    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(Vm.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="magma")
    ax3.set_xlabel("x-position (nm)")
    ax3.set_ylabel("y-position (nm)")
    ax3.set_title("Final $V_m$ Distribution")
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Voltage (V)")

    # 4) 2D slice of the full potential phi
    ax4 = fig.add_subplot(1, 4, 4)
    y_slice_idx = phi.shape[1] // 2
    im4 = ax4.imshow(phi[:, y_slice_idx, :].T, origin="lower", extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]], aspect="auto", cmap="viridis")
    ax4.axhline(0.0, color="r", linestyle="--", label="Membrane Plane")
    ax4.set_xlabel("x-position (nm)")
    ax4.set_ylabel("z-position (nm)")
    ax4.set_title(f"Final Potential $\\phi(x, y={y_nm[y_slice_idx]:.1f}, z)$")
    ax4.legend()
    div4 = make_axes_locatable(ax4)
    cax4 = div4.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im4, cax=cax4, label="Potential (V)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

### NEW ###
# Main driver function is now for the full coupled electroporation problem.
def simulate_electroporation(
    dom: Domain | None = None,
    props: MembraneProps | None = None,
    dyn: PhaseFieldDynamics | None = None,
    elec: Electrostatics | None = None,
    solver: SolverParams | None = None,
) -> None:
    """Main driver for the coupled electroporation simulation."""
    # --- 1. Initialization ---
    dom = dom if dom is not None else Domain()
    props = props if props is not None else MembraneProps()
    dyn = dyn if dyn is not None else PhaseFieldDynamics()
    elec = elec if elec is not None else Electrostatics()
    solver = solver if solver is not None else SolverParams()

    x, y, z, dx, dy, dz = create_grid(dom)
    Nz_actual = z.shape[0]

    if dyn.transition_thickness is None:
        dyn.transition_thickness = 0.5 * dx

    # --- 2. Build Model Components ---
    phi_solver = AMGPoissonSolver(dom, dx, dy, dz)
    psi = initialize_phase_field(x, y, dyn.initial_pore_radius, dyn.transition_thickness)
    H = smooth_step(psi)

    G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom, elec)

    dt = solver.dt if solver.dt is not None else estimate_rc_time_step(G_m_map, C_eff_map, elec, dom)
    nsteps = int(np.ceil(solver.total_time / dt))

    # Initialize the new phase-field solver
    psi_solver = PhaseFieldSolver(dom, dyn, dx, dt)
    delta_C = props.C_lipid - props.C_pore

    if solver.surface_diffusion and solver.D_V == 0.0:
        solver.D_V = 0.05 * dx**2 / dt

    title_prefix = "Coupled Electroporation Simulation"
    print(f"\n--- {title_prefix} ---")
    print(f"Grid: {dom.Nx}x{dom.Ny}x{Nz_actual} (dx={dx*1e9:.2f} nm)")
    print(f"Initial Pore Radius: {dyn.initial_pore_radius*1e9:.1f} nm, Transition: {dyn.transition_thickness*1e9:.2f} nm")
    print(f"Simulation Time: {solver.total_time*1e6:.2f} µs")
    print(f"Time Step (dt): {dt*1e9:.3f} ns ({nsteps} steps)")
    if solver.surface_diffusion:
        print(f"Surface Diffusion Enabled: D_V = {solver.D_V:.3e} m^2/s")
    print("-" * 50)

    # --- 3. Run Simulation ---
    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    diagnostics = {'time': [], 'avg_Vm': [], 'radius_nm': []}
    save_interval = max(1, nsteps // solver.save_frames)

    for n in range(nsteps):
            # --- Update Properties and Solve Electrostatics ---
            H = smooth_step(psi)
            # We now need C_m_map in the main loop, so we retrieve it here.
            G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom, elec)
            phi = phi_solver.solve(Vm, elec.V_applied, dom)
            J_elec = calculate_J_from_phi(phi, phi_solver.k_mem_plus, elec.sigma_e, dz)

            # --- Update Transmembrane Voltage (Vm) ---
            Vm = update_vm(Vm, G_m_map, C_eff_map, H, elec, dom, dt, dx, solver, J_elec_coupled=J_elec)

            # --- MODIFICATION: Calculate Surface Tension and Update Phase Field ---
            # 1. Calculate the electrical surface tension (Maxwell stress).
            # This is the energy density stored in the local membrane capacitor.
            sigma_elec = 0.5 * C_m_map * Vm**2
            
            # 2. Add it to the base mechanical tension to get the total tension.
            sigma_total_map = dyn.sigma + sigma_elec
            
            # 3. Update psi using this new total tension map.
            psi = psi_solver.update_psi(psi, sigma_total_map, dt)

            # --- Diagnostics and Saving ---
            if n % save_interval == 0 or n == nsteps - 1:
                t = (n + 1) * dt
                avg_Vm_val = float(np.mean(Vm))
                # Calculate effective pore radius
                Apore = np.sum(1.0 - H) * dx * dy
                R_eff_nm = np.sqrt(max(Apore, 0.0) / np.pi) * 1e9

                diagnostics['time'].append(t)
                diagnostics['avg_Vm'].append(avg_Vm_val)
                diagnostics['radius_nm'].append(R_eff_nm)

                print(f"Time: {t*1e9:8.2f} ns [{n+1:>{len(str(nsteps))}}/{nsteps}] | Avg Vm: {avg_Vm_val:.4f} V | Pore Radius: {R_eff_nm:5.2f} nm")
    # --- 4. Final calculation of phi for plotting ---
    phi = phi_solver.solve(Vm, elec.V_applied, dom)

    # --- 5. Output Results ---
    print("\nSimulation finished. Generating plots...")
    plot_results(x, y, z, Vm, psi, phi, diagnostics, title_prefix)

    filename = "electroporation_results.npz"
    np.savez_compressed(
        filename, x=x, y=y, z=z, Vm=Vm, psi=psi, phi=phi, H=H,
        time_points=np.array(diagnostics['time']),
        avg_Vm_vs_time=np.array(diagnostics['avg_Vm']),
        radius_vs_time=np.array(diagnostics['radius_nm'])
    )
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # --- Configure and run the coupled electroporation simulation ---
    # We use a smaller grid and shorter time for a quick demonstration.
    # The V_applied is set relatively high to ensure the pore opens.

    sim_domain = Domain(Nx=128, Ny=128, Nz=129)

    sim_mprops = MembraneProps(
        R_lipid=1e7,  # High resistance
        C_lipid=1e-2, # Standard capacitance
        R_pore=1e-1,
        C_pore=1e-9   # ~Zero capacitance
    )

    # A higher line tension will resist opening more strongly.
    sim_dynamics = PhaseFieldDynamics(
        initial_pore_radius=2e-9, # Start with a small 2 nm pore
        gamma=2.0e-11,            # Line tension
        sigma=0.0,               # No base surface tension
        M=5.0e7                   # Phase-field mobility
    )

    # A high applied voltage is needed to overcome line tension.
    sim_electrostatics = Electrostatics(
        sigma_e=1.0,
        V_applied=0.0 # Volts
    )

    # Run for 50 ns to see the initial pore dynamics.
    sim_solver = SolverParams(
        total_time=50e-9,
        save_frames=100
    )

    simulate_electroporation(
        dom=sim_domain,
        props=sim_mprops,
        dyn=sim_dynamics,
        elec=sim_electrostatics,
        solver=sim_solver,
    )