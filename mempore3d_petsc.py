"""
Dynamic membrane pore simulation with coupled electrostatics and phase-field evolution.
This version uses PETSc for parallel 3D electrostatics and includes stochastic
thermal noise to model pore nucleation.

This script models the charging of a lipid membrane that can spontaneously form
and evolve pores. It couples two main physical models:

1.  **Phase-Field Dynamics:** The pore's structure is represented by a phase-field
    variable `psi` (lipid=1, pore=0). The evolution of `psi` is governed by the
    stochastic Allen-Cahn equation, driven by line tension, a dynamic,
    voltage-dependent surface tension, and thermal fluctuations. The solver uses
    a spectral method (FFT) for efficiency.

2.  **Coupled Electrostatics:** The electrical behavior is modeled using a
    semi-implicit scheme. The 3D electrolyte potential (`phi_elec`) is solved
    using a parallel PETSc/GAMG Poisson solver, which is coupled to a 2D
    implicit solver for the transmembrane potential (`Vm`). This approach
    self-consistently captures the interaction between the bulk electrolyte
    and the membrane surface.

The coupling is bidirectional: the pore's shape determines the local electrical
properties (conductance, capacitance), and in turn, the electric field adds an
electromechanical surface tension that lowers the energy barrier for thermal
nucleation and drives pore expansion.

Key Features
------------
- Stochastic Pore Nucleation: Thermodynamically consistent noise is added to the
  phase-field evolution to simulate spontaneous pore formation from an intact membrane.
- Dynamic Phase-Field: The pore evolves based on line tension and electrical stress.
- Parallel 3D Poisson Solver: The 3D Poisson equation for the electrolyte is
  solved in parallel using petsc4py and MPI, with GAMG preconditioning.
- Implicit 2D Vm Solver: A stable, semi-implicit solver handles the time
  evolution of the 2D transmembrane potential, with a Numba-accelerated
  matrix assembly.
- Bidirectional Coupling: The phase-field and electrostatics are solved in the
  same time loop, providing a tight coupling between pore structure and electrical forces.

Author: Saman Seifi

"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numba
import numpy as np
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable
from petsc4py import PETSc
from scipy import fft as spfft
from scipy.sparse import coo_matrix
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

# -----------------------------------------------------------------------------
# Numba-JIT Compiled Core Routines
# -----------------------------------------------------------------------------
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
            
            if i > 0: add_entry(p, p - 1, lap_offdiag_term)
            else: diag_val += lap_offdiag_term
            if i < Nx - 1: add_entry(p, p + 1, lap_offdiag_term)
            else: diag_val += lap_offdiag_term
            if j > 0: add_entry(p, p - Nx, lap_offdiag_term)
            else: diag_val += lap_offdiag_term
            if j < Ny - 1: add_entry(p, p + Nx, lap_offdiag_term)
            else: diag_val += lap_offdiag_term

            diag_val += lap_diag_term
            add_entry(p, p, diag_val)
    return count

@numba.njit(cache=True)
def calculate_J_from_phi(phi, k_plane_index, sigma_e, dz):
    """Numba-jitted calculation of ionic current density from 3D potential."""
    grad_phi_z = (phi[:, :, k_plane_index + 1] - phi[:, :, k_plane_index]) / dz
    return sigma_e * grad_phi_z

# -----------------------------------------------------------------------------
# Solvers and High-Level Routines
# -----------------------------------------------------------------------------
def build_vm_implicit_operator(dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, d):
    """Builds the sparse matrix for the implicit Vm solve using a Numba kernel."""
    Nx, Ny = dom.Nx, dom.Ny
    base_G_sc = 2.0 * sigma_e / d
    G_sc_map = 50.0 * base_G_sc * (1.0 - H)
    G_total_map = G_m_map + G_sc_map

    C_eff_flat = C_eff_map.flatten(order='C')
    G_total_flat = G_total_map.flatten(order='C')
    
    max_nnz = 5 * Nx * Ny
    row_idx = np.empty(max_nnz, dtype=np.int32)
    col_idx = np.empty(max_nnz, dtype=np.int32)
    data_vals = np.empty(max_nnz, dtype=np.float64)

    nnz = _build_implicit_vm_operator_numba(
        Nx, Ny, C_eff_flat, G_total_flat, dt, dx, D_V, row_idx, col_idx, data_vals
    )
    A = coo_matrix((data_vals[:nnz], (row_idx[:nnz], col_idx[:nnz])), shape=(Nx*Ny, Nx*Ny)).tocsc()
    return A

class PETScPoissonGAMG:
    def __init__(self, Nx, Ny, Nz, dx, dy, dz):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz if Nz % 2 else Nz + 1
        self.dx2, self.dy2, self.dz2 = dx*dx, dy*dy, dz*dz
        self.cx, self.cy, self.cz = 1.0/self.dx2, 1.0/self.dy2, 1.0/self.dz2

        self.k_mem_minus = self.Nz // 2 - 1
        self.k_mem_plus  = self.Nz // 2 + 1

        N = self.Nx * self.Ny * self.Nz
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setSizes(((PETSc.DECIDE, N), (PETSc.DECIDE, N)))
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setPreallocationNNZ((7,7))
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        self.b = PETSc.Vec().createMPI(N, comm=self.comm); self.b.set(0.0)
        self.x = PETSc.Vec().createMPI(N, comm=self.comm); self.x.set(0.0)

        r0, r1 = self.A.getOwnershipRange()
        for row in range(r0, r1):
            i = row % self.Nx
            t = row // self.Nx
            j = t % self.Ny
            k = t // self.Ny

            diag = 0.0; cols=[]; vals=[]
            # X-neighbors (Neumann)
            if i == 0:
                cols.append(self.gid(i+1,j,k)); vals.append(-2.0*self.cx); diag += 2.0*self.cx
            elif i == self.Nx-1:
                cols.append(self.gid(i-1,j,k)); vals.append(-2.0*self.cx); diag += 2.0*self.cx
            else:
                cols += [self.gid(i-1,j,k), self.gid(i+1,j,k)]
                vals += [-self.cx, -self.cx]; diag += 2.0*self.cx

            # Y-neighbors (Neumann)
            if j == 0:
                cols.append(self.gid(i,j+1,k)); vals.append(-2.0*self.cy); diag += 2.0*self.cy
            elif j == self.Ny-1:
                cols.append(self.gid(i,j-1,k)); vals.append(-2.0*self.cy); diag += 2.0*self.cy
            else:
                cols += [self.gid(i,j-1,k), self.gid(i,j+1,k)]
                vals += [-self.cy, -self.cy]; diag += 2.0*self.cy

            # Z-neighbors (interior stencil for now, Dirichlet BCs handled by zeroRowsColumns)
            # This robustly adds the full diagonal contribution before BCs are applied.
            if k > 0:
                cols.append(self.gid(i,j,k-1)); vals.append(-self.cz)
            if k < self.Nz-1:
                cols.append(self.gid(i,j,k+1)); vals.append(-self.cz)
            diag += 2.0*self.cz # Full diagonal term for Z

            cols.append(row); vals.append(diag)
            self.A.setValues([row], cols, vals)

        self.A.assemblyBegin(); self.A.assemblyEnd()
        self.b.assemblyBegin(); self.b.assemblyEnd()
        
        self.A_pristine = self.A.copy()

        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType('cg')
        pc = self.ksp.getPC(); pc.setType('gamg')
        self.ksp.setTolerances(rtol=1e-12, max_it=200)
        self.ksp.setFromOptions()

        self.rows_z0   = [self.gid(i,j,0) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_z1   = [self.gid(i,j,self.Nz-1) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_km   = [self.gid(i,j,self.k_mem_minus) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_kp   = [self.gid(i,j,self.k_mem_plus) for j in range(self.Ny) for i in range(self.Nx)]
        self.IS_z0 = PETSc.IS().createGeneral(self.rows_z0, comm=self.comm)
        self.IS_z1 = PETSc.IS().createGeneral(self.rows_z1, comm=self.comm)
        self.IS_km = PETSc.IS().createGeneral(self.rows_km, comm=self.comm)
        self.IS_kp = PETSc.IS().createGeneral(self.rows_kp, comm=self.comm)

    def gid(self, i, j, k):
        return i + self.Nx * (j + self.Ny * k)

    def apply_dirichlet_planes(self, Vm, V_applied):
        """
        Set boundary values into x and zero rows+cols with unit diag.
        """
        self.A.destroy()
        self.A = self.A_pristine.copy()
        self.ksp.setOperators(self.A)
        self.x.set(0.0); self.b.set(0.0)

        for r in self.rows_z0: self.x.setValue(r, -0.5 * V_applied)
        for r in self.rows_z1: self.x.setValue(r, +0.5 * V_applied)
        
        if Vm is not None:
            for j in range(self.Ny):
                for i in range(self.Nx):
                    r_minus = self.rows_km[i + self.Nx*j]
                    r_plus  = self.rows_kp[i + self.Nx*j]
                    self.x.setValue(r_minus, -0.5 * Vm[i, j])
                    self.x.setValue(r_plus,  +0.5 * Vm[i, j])

        self.x.assemble(); self.b.assemble()

        self.A.zeroRowsColumns(self.IS_z0, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_z1, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_km, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_kp, 1.0, self.x, self.b)

    def solve(self):
        self.ksp.solve(self.b, self.x)
        return self.x

    def current_slice_kplus_minus(self, dz, sigma_e):
        """
        Compute J_z at k_mem_plus and k_mem_minus. Gathers full vector to rank 0.
        Returns a tuple of (J_plus, J_minus) as (Nx, Ny) ndarrays on rank 0; returns None elsewhere.
        """
        scat, y = PETSc.Scatter.toZero(self.x)
        scat.begin(self.x, y, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.end(self.x, y, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.destroy()

        if self.rank == 0:
            phi_flat = y.getArray(readonly=True)
            phi = phi_flat.reshape(self.Nz, self.Ny, self.Nx).transpose(2, 1, 0)
            k_plus = self.k_mem_plus
            k_minus = self.k_mem_minus
            grad_plus = (phi[:, :, k_plus + 1] - phi[:, :, k_plus]) / dz
            grad_minus = (phi[:, :, k_minus] - phi[:, :, k_minus - 1]) / dz
            return sigma_e * (grad_plus + grad_minus)
        else:
            return None

class ImplicitVMSolver:
    """Solver for the semi-implicit update of the 2D transmembrane potential Vm."""
    def __init__(self, dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, d):
        self.Nx, self.Ny = dom.Nx, dom.Ny
        A = build_vm_implicit_operator(dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, d)
        self.solver = factorized(A)
        
    def solve(self, b_flat):
        """Solves the system Ax = b using the pre-factorized operator."""
        x_flat = self.solver(b_flat)
        return x_flat.reshape((self.Nx, self.Ny), order='C')

class PhaseFieldSolver:
    """
    Stochastic Allen–Cahn with spectral (FFT) time stepping using scipy.fft.
    Uses rfft2/irfft2 (real transforms) + keeps psi_hat to avoid fft(psi) each step.

    Key Features:
    - Semi-implicit spectral update for phase-field evolution.
    - Thermodynamically consistent thermal noise for stochastic pore nucleation.
    - Efficient handling of real FFTs with precomputed wavenumbers.
    - Smooth-step blending and double-well potential derivatives for phase-field dynamics.
    - Noise amplitude derived from thermal parameters.
    """
    def __init__(self, dom: Domain, phase_params: PhaseFieldParams,
                 thermal_params: ThermalParams, dx: float, dt: float, workers: Optional[int]=None):
        self.params = phase_params
        self.thermal = thermal_params
        self.dom = dom
        self.dx = dx
        self.dt = dt
        self.workers = int(workers or os.cpu_count() or 1)
        self.Cg = np.sqrt(2.0) / 12.0

        kx = 2.0 * np.pi * spfft.fftfreq(dom.Nx, d=dx)
        ky = 2.0 * np.pi * spfft.rfftfreq(dom.Ny, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = KX**2 + KY**2

        if self.params.transition_thickness is None:
            self.params.transition_thickness = 1.5 * dx

        M = self.params.mobility
        gam = self.params.line_tension
        tth = self.params.transition_thickness
        self.denom = 1.0 + self.dt * M * (gam / self.Cg) * tth * self.K2

        if self.thermal.add_noise:
            self.noise_amplitude = np.sqrt((2 * M * self.thermal.k_B * self.thermal.T) / (dx * dx * self.dt))
        else:
            self.noise_amplitude = 0.0

        self._rhs_real = np.empty((dom.Nx, dom.Ny), dtype=np.float64)
        self.psi_hat = None

    @staticmethod
    @numba.njit(cache=True)
    def _g_prime(psi):
        return 0.5 * psi * (1.0 - psi) * (1.0 - 2.0 * psi)

    @staticmethod
    @numba.njit(cache=True)
    def _dH_prime(psi):
        psi_clipped = np.clip(psi, 0.0, 1.0)
        return 6.0 * psi_clipped * (1.0 - psi_clipped)

    def evolve(self, psi_in: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
        if self.psi_hat is None:
            self.psi_hat = spfft.rfft2(psi_in, workers=self.workers)
            psi = psi_in
        else:
            psi = spfft.irfft2(self.psi_hat, s=psi_in.shape, workers=self.workers).real

        det_rhs = -self.params.mobility * (
            (self.params.line_tension / self.Cg) * (self._g_prime(psi) / self.params.transition_thickness) +
            sigma_map * self._dH_prime(psi)
        )

        if self.noise_amplitude != 0.0:
            self._rhs_real[:] = det_rhs + self.noise_amplitude * np.random.randn(*psi.shape)
        else:
            self._rhs_real[:] = det_rhs

        rhs_hat = spfft.rfft2(self._rhs_real, workers=self.workers)
        self.psi_hat = (self.psi_hat + self.dt * rhs_hat) / self.denom

        psi_new = spfft.irfft2(self.psi_hat, s=psi.shape, workers=self.workers).real
        np.clip(psi_new, 0.0, 1.0, out=psi_new)
        return psi_new


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
    psi_clipped = np.clip(psi, 0.0, 1.0)
    return psi_clipped**2 * (3.0 - 2.0 * psi_clipped)

def initialize_phase_field(params: PhaseFieldParams, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constructs the initial phase field."""
    if params.initial_state == 'pore':
        xx, yy = np.meshgrid(x, y, indexing="ij")
        r = np.sqrt(xx**2 + yy**2)
        psi = 0.5 * (1.0 - np.tanh((params.initial_pore_radius - r) /
                                 (np.sqrt(2.0) * params.transition_thickness)))
        return psi
    elif params.initial_state == 'intact':
        psi = np.ones((len(x), len(y)), dtype=float)
        noise = params.intact_noise_level * (np.random.rand(len(x), len(y)) - 0.5)
        return np.clip(psi + noise, 0.0, 1.0)
    else:
        raise ValueError(f"Invalid initial_state '{params.initial_state}'.")

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends material properties based on the phase field."""
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def blend_properties_sharp(psi: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends material properties based on a sharp threshold."""
    is_lipid_mask = (psi > 0.5).astype(float)
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    C_lipid, C_pore = props.C_lipid, props.C_pore
    G_m_map = G_pore + (G_lipid - G_pore) * is_lipid_mask
    C_m_map = C_pore + (C_lipid - C_pore) * is_lipid_mask
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def estimate_base_time_step(props: MembraneProps, elec: Electrostatics, dom: Domain, solver: SolverParams) -> Tuple[float, float, int]:
    """Estimates a base time step `dt` from explicit stability criteria."""
    G_m_max = 1.0 / props.R_pore
    C_eff_min = 2.0 * EPS_W / dom.Lz + props.C_pore
    A = 2.0 * elec.sigma_e / dom.Lz
    rate_max = (A + G_m_max) / C_eff_min
    tau_local_min = 1.0 / rate_max
    dt_base = solver.dt_safety * tau_local_min
    
    tau_report = props.C_lipid / (1.0 / props.R_lipid + A)
    total_time = solver.n_tau_total * tau_report
    nsteps_base = int(np.ceil(total_time / dt_base))
    return dt_base, total_time, nsteps_base

def calculate_pore_radius_simple(H: np.ndarray, dx: float, dy: float) -> float:
    pore_area = np.sum(1.0 - H) * dx * dy
    eff_radius = np.sqrt(max(pore_area, 0.0) / np.pi)
    return eff_radius

# -----------------------------------------------------------------------------
# Visualization and Main Driver
# -----------------------------------------------------------------------------
def plot_results(x, y, z, Vm, phi_elec, psi, time_points, avg_Vm_vs_time, pore_radius_vs_time, title_prefix, elapsed_time):
    """Generates and displays plots of the simulation results."""
    fig = plt.figure(figsize=(24, 6))
    x_nm, y_nm = x * 1e9, y * 1e9
    time_ns = np.array(time_points) * 1e9

    fig.suptitle(f"{title_prefix}\n(Total Wall Clock Time: {elapsed_time:.2f} s)", fontsize=16, y=1.05)

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(time_ns, avg_Vm_vs_time, marker=".", linestyle="-", label="Avg Vm")
    ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Average Vm (V)")
    ax1.set_title("Membrane Charging"); ax1.grid(True, linestyle='--')
    
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(time_ns, np.array(pore_radius_vs_time) * 1e9, marker='.', linestyle='-', color='crimson')
    ax2.set_xlabel("Time (ns)"); ax2.set_ylabel("Effective Pore Radius (nm)")
    ax2.set_title("Pore Dynamics"); ax2.grid(True, linestyle='--')
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(Vm.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="magma")
    ax3.set_xlabel("x (nm)"); ax3.set_ylabel("y (nm)"); ax3.set_title("Final $V_m$ Distribution")
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Voltage (V)")

    ax4 = fig.add_subplot(1, 4, 4)
    im4 = ax4.imshow(psi.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="cividis", vmin=0, vmax=1)
    ax4.set_xlabel("x (nm)"); ax4.set_ylabel("y (nm)"); ax4.set_title("Final Pore Shape ($\\psi$)")
    cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im4, cax=cax4, label="Phase Field")

    plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()


def simulate_membrane_charging(dom_in: Domain | None = None, props: MembraneProps | None = None,
                               phase: PhaseFieldParams | None = None, elec: Electrostatics | None = None,
                               solver: SolverParams | None = None, thermal: ThermalParams | None = None) -> None:
    """ Main driver for the simulation. """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    start_time = time.time()
    dom = dom_in if dom_in is not None else Domain()
    props = props if props is not None else MembraneProps()
    solver = solver if solver is not None else SolverParams()

    phase_field_on = phase.evolve == 'on' if phase else False
    electrostatics_on = elec is not None
    
    if thermal is None: thermal = ThermalParams(add_noise=False)
    if not electrostatics_on: elec = Electrostatics(V_applied=0.0)
    if not phase: phase = PhaseFieldParams(evolve='off')

    x, y, z, dx, dy, dz = create_grid(dom)
    if phase.transition_thickness is None: phase.transition_thickness = 1.5 * dx

    dt_base, total_time, _ = estimate_base_time_step(props, elec, dom, solver)
    dt = dt_base * solver.implicit_dt_multiplier
    nsteps = int(np.ceil(total_time / dt))

    if solver.surface_diffusion and solver.D_V == 0.0:
        solver.D_V = 0.0 # 0.05 * dx**2 / dt
    
    title_prefix = "Coupled Electrostatics & Phase-Field"

    if rank == 0:
        print(f"\n--- {title_prefix} ---")
        print(f"Grid: {dom.Nx}x{dom.Ny}x{dom.Nz} (dx={dx*1e9:.2f} nm)")
        tau_report = total_time / solver.n_tau_total
        print(f"System Time Constant (τ_lipid): {tau_report*1e9:.2f} ns")
        print(f"Total Sim Time: {total_time*1e6:.2f} µs ({solver.n_tau_total:.1f} τ_lipid)")
        print(f"Time step (dt): {dt*1e9:.3f} ns | Total steps: {nsteps}")
        if electrostatics_on:
            print(f"Vm solver will be rebuilt every {solver.rebuild_vm_solver_every} steps.")
        print(f"Phase-field evolution enabled: {phase_field_on}")
        if phase_field_on:
            print(f"Thermal noise enabled: {thermal.add_noise} at T = {thermal.T} K")
        print("-" * 60)

    # --- Initialize Solvers (on all ranks as needed) ---
    psi = initialize_phase_field(phase, x, y)
    phase_field_solver = PhaseFieldSolver(dom, phase, thermal, dx, dt, workers=os.cpu_count()) if phase_field_on else None
    poisson = PETScPoissonGAMG(dom.Nx, dom.Ny, dom.Nz, dx, dy, dz) if electrostatics_on else None
    vm_implicit_solver = None

    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    time_points, avg_Vm_vs_time, pore_radius_vs_time = [], [], []
    save_interval = max(1, nsteps // solver.save_frames)

    for n in range(nsteps):
        if thermal.add_noise:
            G_m_map, C_m_map, C_eff_map = blend_properties_sharp(psi, props, dom)
        else:
            H = smooth_step(psi)
            G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom)

        sigma_elec = 0.0
        if electrostatics_on:
            if rank == 0:
                if n % solver.rebuild_vm_solver_every == 0 or vm_implicit_solver is None:
                    if n > 0 and rank == 0:
                        print(f"Step {n}: Rebuilding Vm implicit solver...")
                    H_for_solver = (psi > 0.5).astype(float) if thermal.add_noise else smooth_step(psi)
                    vm_implicit_solver = ImplicitVMSolver(
                        dom, C_eff_map, G_m_map, H_for_solver, dt, dx, solver.D_V, elec.sigma_e, 2*dz
                    )
            
            Vm = comm.bcast(Vm if rank == 0 else None, root=0)
            poisson.apply_dirichlet_planes(Vm, elec.V_applied)
            poisson.solve()
            J_elec_coupled = poisson.current_slice_kplus_minus(dz, elec.sigma_e)
            
            if rank == 0:
                b_rhs_2d = C_eff_map * Vm + dt * J_elec_coupled
                Vm = vm_implicit_solver.solve(b_rhs_2d.flatten(order='C'))
                
            Vm = comm.bcast(Vm if rank == 0 else None, root=0)
            sigma_elec = 0.5 * C_m_map * (Vm**2)
            
        if phase_field_on:
            sigma_total_map = phase.sigma_area + sigma_elec
            if rank == 0:
                psi = phase_field_solver.evolve(psi, sigma_total_map)
            # Broadcast the updated phase field to all processes to ensure consistency
            psi = comm.bcast(psi, root=0)
        
        if n % save_interval == 0 or n == nsteps - 1:
            t = (n + 1) * dt
            # Ensure psi is consistent before calculating radius on rank 0
            eff_radius = calculate_pore_radius_simple(smooth_step(psi), dx, dy)
            avg_Vm = float(np.mean(Vm)) if electrostatics_on else 0.0

            if rank == 0:
                time_points.append(t)
                avg_Vm_vs_time.append(avg_Vm)
                pore_radius_vs_time.append(eff_radius)
                print(f"Time: {t*1e9:8.2f} ns [{n+1:>{len(str(nsteps))}}/{nsteps}] | Avg Vm: {avg_Vm:.4f} V | Pore R: {eff_radius*1e9:.2f} nm")

    # --- Output Results ---
    # Final state is already synchronized from the last loop iteration
    if electrostatics_on:
        scat, phi_vec = PETSc.Scatter.toZero(poisson.x)
        scat.begin(poisson.x, phi_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.end  (poisson.x, phi_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.destroy()
        if rank == 0:
            phi_arr = phi_vec.getArray(readonly=True)
            phi_elec_out = phi_arr.reshape(dom.Nz, dom.Ny, dom.Nx).transpose(2,1,0)
    else:
        phi_elec_out = np.zeros((dom.Nx, dom.Ny, dom.Nz)) if rank == 0 else None

    elapsed_time = time.time() - start_time

    if rank == 0:
        print(f"\nSimulation finished in {elapsed_time:.2f} seconds.")
        plot_results(x, y, z, Vm, phi_elec_out, psi, time_points, avg_Vm_vs_time,
                     pore_radius_vs_time, title_prefix, elapsed_time)

        filename = "membrane_simulation_results.npz"
        H = smooth_step(psi)
        np.savez_compressed(
            filename, x=x, y=y, z=z, Vm=Vm, phi_elec=phi_elec_out, psi=psi, H=H,
            time_points=np.array(time_points), avg_Vm_vs_time=np.array(avg_Vm_vs_time),
            pore_radius_vs_time=np.array(pore_radius_vs_time),
            domain_params=dom, props_params=props, phase_params=phase,
            elec_params=elec, solver_params=solver, thermal_params=thermal
        )
        print(f"Numerical results saved to {filename}")
    comm.Barrier()

