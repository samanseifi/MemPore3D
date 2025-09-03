import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# DATA CLASSES FOR PARAMETERS
# =============================================================================

@dataclass
class Domain:
    """Parameters defining the computational grid."""
    Nx: int = 128
    Ny: int = 128
    Nz: int = 129  # Will be forced to odd for a center plane
    Lx: float = 100e-9
    Ly: float = 100e-9
    Lz: float = 200e-9

@dataclass
class PhaseFieldDynamicsParams:
    """Parameters for the Cahn-Hilliard phase-field evolution."""
    gamma: float = 1.0e-11  # Surface tension (J/m) -> related to interfacial energy
    sigma: float = 0.8e-11      # Area constraint penalty (initially zero)
    M: float = 5e+8       # Mobility parameter (m^2 s / kg)
    initial_pore_radius: float = 20e-9
    transition_thickness: float | None = None  # If None -> 2*dx

@dataclass
class MembraneProps:
    """Electrical properties of the membrane components."""
    R_lipid: float = 1e7    # Ohm·m^2
    C_lipid: float = 1e-2    # F/m^2
    R_pore: float = 1e-1     # Ohm·m^2 (conductive)
    C_pore: float = 1e-9     # F/m^2 (near zero)

@dataclass
class Electrostatics:
    """Parameters for the electrostatics problem."""
    sigma_e: float = 1.0     # Electrolyte conductivity (S/m)
    V_applied: float = 0.5   # Applied voltage (V)

@dataclass
class SolverParams:
    """Parameters controlling the numerical solver."""
    dt_phase: float = 1e-9   # Time step for phase-field evolution (s)
    n_steps: int = 2000      # Total number of phase-field steps
    dt_elec_safety: float = 0.01 # Safety factor for Vm time step
    max_iter_jacobi: int = 100 # Max iterations for 3D potential solver
    tolerance_jacobi: float = 1e-6
    save_every: int = 50     # How often to save data/plots

# =============================================================================
# UTILITY AND PHYSICS FUNCTIONS
# =============================================================================

def create_grid(dom: Domain) -> Tuple[np.ndarray, ...]:
    """Creates and returns grid coordinates and spacings."""
    dom.Nz = dom.Nz if dom.Nz % 2 != 0 else dom.Nz + 1
    x = np.linspace(-dom.Lx / 2, dom.Lx / 2, dom.Nx, endpoint=False)
    y = np.linspace(-dom.Ly / 2, dom.Ly / 2, dom.Ny, endpoint=False)
    z = np.linspace(-dom.Lz / 2, dom.Lz / 2, dom.Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz

def g(phi):
    """Double-well potential for phase separation."""
    return 0.25 * phi**2 * (1.0 - phi)**2

def gp(phi):
    """Derivative of the double-well potential."""
    return 0.5 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

def smooth_step(phi):
    """Cubic Hermite smooth step H(phi) = phi^2 * (3 - 2*phi)."""
    pc = np.clip(phi, 0.0, 1.0)
    return pc**2 * (3.0 - 2.0 * pc)

def dH_dphi(phi):
    """Derivative of the smooth step function."""
    pc = np.clip(phi, 0.0, 1.0)
    return 6.0 * pc * (1.0 - pc)

def initialize_phase_field(x: np.ndarray, y: np.ndarray, radius: float, epsilon: float) -> np.ndarray:
    """Initializes the 2D phase field with a central pore."""
    XX, YY = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(XX**2 + YY**2)
    phi = 0.5 * (1.0 + np.tanh((r - radius) / (np.sqrt(2.0) * epsilon)))
    return phi

def blend_electrical_properties(H, props: MembraneProps, dom: Domain):
    """Blends electrical properties based on the phase-field H(phi)."""
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    
    eps0 = 8.8541878128e-12
    eps_w = 80.0 * eps0
    C_bath = 2.0 * eps_w / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def update_vm_subcycled(Vm, H, G_m_map, C_eff_map, elec: Electrostatics, dom: Domain,
                        dt_phase: float, dt_elec_safety: float):
    """Updates Vm by sub-cycling with a smaller, stable time step."""
    # Short-circuit conductance to force Vm->0 in the pore region (H~0)
    G_sc_factor = 50.0
    G_sc = G_sc_factor * (2.0 * elec.sigma_e / dom.Lz) * (1.0 - H)
    
    # Driving current from the external field, only acts on membrane (H~1)
    J_elec = H * (2.0 * elec.sigma_e * (elec.V_applied - Vm) / dom.Lz)
    
    # Determine the local electrical time constant and the stable dt_elec
    rate_max = ( (2*elec.sigma_e/dom.Lz) + np.max(G_m_map) + np.max(G_sc) ) / np.min(C_eff_map)
    dt_elec = dt_elec_safety / rate_max
    
    num_sub_steps = max(1, int(np.ceil(dt_phase / dt_elec)))

    # Sub-cycle loop
    for _ in range(num_sub_steps):
        dVm_dt = (J_elec - (G_m_map + G_sc) * Vm) / C_eff_map
        Vm = Vm + (dt_phase / num_sub_steps) * dVm_dt
        # Re-evaluate J_elec because it depends on Vm
        J_elec = H * (2.0 * elec.sigma_e * (elec.V_applied - Vm) / dom.Lz)
        
    return Vm

def solve_potential_3d(phi_elec, Vm, k_mem, V_applied, dx, dy, dz, tol, max_iter):
    """Solves the 3D Laplace equation for electric potential using Jacobi iteration."""
    dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
    laplace_den = 2.0/dx2 + 2.0/dy2 + 2.0/dz2

    # Set boundary conditions
    phi_elec[:, :, 0] = -V_applied / 2.0
    phi_elec[:, :, -1] = V_applied / 2.0
    phi_elec[:, :, k_mem - 1] = -Vm / 2.0
    phi_elec[:, :, k_mem + 1] = Vm / 2.0
    
    for it in range(max_iter):
        phi_old = phi_elec.copy()
        
        # Jacobi update for the interior
        phi_elec[1:-1, 1:-1, 1:-1] = (
            (phi_old[2:, 1:-1, 1:-1] + phi_old[:-2, 1:-1, 1:-1]) / dx2 +
            (phi_old[1:-1, 2:, 1:-1] + phi_old[1:-1, :-2, 1:-1]) / dy2 +
            (phi_old[1:-1, 1:-1, 2:] + phi_old[1:-1, 1:-1, :-2]) / dz2
        ) / laplace_den
        
        # Re-impose all boundary conditions
        phi_elec[:, :, 0] = -V_applied / 2.0
        phi_elec[:, :, -1] = V_applied / 2.0
        phi_elec[:, :, k_mem - 1] = -Vm / 2.0
        phi_elec[:, :, k_mem + 1] = Vm / 2.0
        phi_elec[0, :, :] = phi_elec[1, :, :]; phi_elec[-1, :, :] = phi_elec[-2, :, :]
        phi_elec[:, 0, :] = phi_elec[:, 1, :]; phi_elec[:, -1, :] = phi_elec[:, -2, :]
        
        if np.max(np.abs(phi_elec - phi_old)) < tol:
            return phi_elec, it + 1
            
    return phi_elec, max_iter

def plot_results(diagnostics: pd.DataFrame, phi_phase, Vm, x, y):
    """Generates and saves plots of the simulation results."""
    t_ns = diagnostics['t'].values * 1e9
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Pore Radius vs. Time
    axes[0, 0].plot(t_ns, diagnostics['R_eff'].values * 1e9, 'b-')
    axes[0, 0].set_title("Pore Radius vs. Time")
    axes[0, 0].set_xlabel("Time (ns)")
    axes[0, 0].set_ylabel("Effective Radius (nm)")
    axes[0, 0].grid(True)
    
    # 2. Average Vm vs. Time
    axes[0, 1].plot(t_ns, diagnostics['Vm_avg'].values, 'r-')
    axes[0, 1].set_title("Average Transmembrane Voltage vs. Time")
    axes[0, 1].set_xlabel("Time (ns)")
    axes[0, 1].set_ylabel("Average Vm (V)")
    axes[0, 1].grid(True)

    # 3. Final Phase Field
    im3 = axes[1, 0].imshow(phi_phase.T, origin='lower', cmap='viridis',
                            extent=[x[0]*1e9, x[-1]*1e9, y[0]*1e9, y[-1]*1e9])
    axes[1, 0].set_title(f"Final Phase Field (t={t_ns[-1]:.1f} ns)")
    axes[1, 0].set_xlabel("x (nm)"); axes[1, 0].set_ylabel("y (nm)")
    plt.colorbar(im3, ax=axes[1, 0], label="phi (0=pore, 1=lipid)")

    # 4. Final Vm Distribution
    im4 = axes[1, 1].imshow(Vm.T, origin='lower', cmap='magma',
                            extent=[x[0]*1e9, x[-1]*1e9, y[0]*1e9, y[-1]*1e9])
    axes[1, 1].set_title(f"Final Vm (t={t_ns[-1]:.1f} ns)")
    axes[1, 1].set_xlabel("x (nm)"); axes[1, 1].set_ylabel("y (nm)")
    plt.colorbar(im4, ax=axes[1, 1], label="Voltage (V)")
    
    plt.tight_layout()
    plt.savefig("coupled_simulation_summary.png", dpi=150)
    plt.show()

# =============================================================================
# MAIN SIMULATION DRIVER
# =============================================================================

def simulate_coupled_model():
    """Main driver for the coupled phase-field and electrophysiology simulation."""
    # --- 1. Setup Parameters and Grid ---
    dom = Domain()
    phase_params = PhaseFieldDynamicsParams()
    mem_props = MembraneProps()
    elec_params = Electrostatics()
    solver_params = SolverParams()
    
    x, y, z, dx, dy, dz = create_grid(dom)
    Cg = np.sqrt(2.0) / 12.0 # Geometric constant for Cahn-Hilliard
    
    epsilon = phase_params.transition_thickness or (0.5 * dx)
    
    # --- 2. Initialize Fields ---
    phi_phase = initialize_phase_field(x, y, phase_params.initial_pore_radius, epsilon)
    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    phi_elec = np.zeros((dom.Nx, dom.Ny, dom.Nz), dtype=float)
    k_mem = dom.Nz // 2
    
    # --- 3. Setup Spectral Solver for Phase Field ---
    kx = 2.0 * np.pi * np.fft.fftfreq(dom.Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(dom.Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    denom_fft = 1.0 + solver_params.dt_phase * phase_params.M * (phase_params.gamma / Cg) * epsilon * K2

    # --- 4. Main Time Loop ---
    diagnostics = {'t': [], 'R_eff': [], 'Vm_avg': []}
    print("Starting coupled simulation...")
    
    for n in range(solver_params.n_steps + 1):
        t = n * solver_params.dt_phase
        
        # --- A. Diagnostics and Saving ---
        if n % solver_params.save_every == 0:
            H = smooth_step(phi_phase)
            pore_area = np.sum(1.0 - H) * dx * dy
            R_eff = np.sqrt(max(pore_area, 0.0) / np.pi)
            Vm_avg = np.mean(Vm)
            
            diagnostics['t'].append(t)
            diagnostics['R_eff'].append(R_eff)
            diagnostics['Vm_avg'].append(Vm_avg)
            
            print(f"Step {n:>5}/{solver_params.n_steps} | t={t*1e9:.2f} ns | R_eff={R_eff*1e9:.2f} nm | Vm_avg={Vm_avg:.4f} V")

        if n == solver_params.n_steps:
            break
            
        # --- B. Update Phase Field (phi_phase) ---
        chemical_potential = (phase_params.gamma / Cg) * (gp(phi_phase) / epsilon) \
                             + phase_params.sigma * dH_dphi(phi_phase)
                             
        phi_hat = np.fft.fft2(phi_phase)
        rhs_hat = np.fft.fft2(-phase_params.M * chemical_potential)
        
        phi_phase_new_hat = (phi_hat + solver_params.dt_phase * rhs_hat) / denom_fft
        phi_phase = np.fft.ifft2(phi_phase_new_hat).real
        phi_phase = np.clip(phi_phase, 0.0, 1.0) # Enforce bounds
        
        # --- C. Update Electrical Properties and Vm ---
        # The new pore geometry (phi_phase) defines the electrical properties
        H = smooth_step(phi_phase)
        G_m_map, C_m_map, C_eff_map = blend_electrical_properties(H, mem_props, dom)
        
        # Sub-cycle the Vm update because it's much faster than phi_phase evolution
        Vm = update_vm_subcycled(
            Vm, H, G_m_map, C_eff_map, elec_params, dom,
            solver_params.dt_phase, solver_params.dt_elec_safety
        )
        
    # --- 5. Final Calculations and Visualization ---
    # Solve for the final 3D potential for plotting purposes
    phi_elec, iters = solve_potential_3d(
        phi_elec, Vm, k_mem, elec_params.V_applied,
        dx, dy, dz, solver_params.tolerance_jacobi, solver_params.max_iter_jacobi
    )
    print(f"Final Jacobi solve for phi_elec converged in {iters} iterations.")
    
    df = pd.DataFrame(diagnostics)
    df.to_csv("coupled_diagnostics.csv", index=False)
    
    plot_results(df, phi_phase, Vm, x, y)


if __name__ == "__main__":
    simulate_coupled_model()