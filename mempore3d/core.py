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
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.measure import find_contours

from mempore3d.parameters import *
from mempore3d.solvers.poisson_solver import *
from mempore3d.solvers.leaky_dielectric_solver import *
from mempore3d.solvers.phase_field_solver import *
from mempore3d.plotting import plot_results

from mpi4py import MPI


# --- Physical Constants ---
EPS0 = 8.8541878128e-12  # Vacuum permittivity, F/m
EPS_W = 80.0 * EPS0      # Water permittivity, F/m


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
        solver.D_V = 0.0 

    # --- Setup Output Directory ---
    output_dir = "simulation_results"
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"\n--- Coupled Electrostatics & Phase-Field ---")
        print(f"Grid: {dom.Nx}x{dom.Ny}x{dom.Nz}")
        print(f"Saving output every {solver.save_frames} frames to '{output_dir}/'")
        print("-" * 60)

    # --- Initialize Solvers ---
    psi = initialize_phase_field(phase, x, y)
    phase_field_solver = PhaseFieldSolver(dom, phase, thermal, dx, dt, workers=os.cpu_count()) if phase_field_on else None

    if electrostatics_on:
        if solver.poisson_solver == 'spectral':
            poisson = SpectralPoissonSolver(dom.Nx, dom.Ny, dom.Nz, dx, dy, dz)
        elif solver.poisson_solver == 'gamg_petsc':
            poisson = PETScPoissonGAMG(dom.Nx, dom.Ny, dom.Nz, dx, dy, dz)
        else:
            raise ValueError(f"Unknown poisson_solver: '{solver.poisson_solver}'")
    else:
        poisson = None

    vm_implicit_solver = None
    Vm = np.zeros((dom.Nx, dom.Ny), dtype=float)
    
    # Store history for the FINAL summary plot (optional, or you can save incrementally)
    time_points, avg_Vm_vs_time, pore_radius_vs_time = [], [], []
    save_interval = max(1, nsteps // solver.save_frames)

    # --- Main Time Loop ---
    for n in range(nsteps):
        if thermal.add_noise:
            # Note: Ensure you use the "masked noise" fix discussed previously here inside the solver!
            H = (psi > 0.5).astype(float) 
            G_m_map, C_m_map, C_eff_map = blend_properties_sharp(psi, props, dom)
        else:
            H = smooth_step(psi)
            G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom)

        # 1. Update Electrostatics
        if electrostatics_on:
            if rank == 0:
                if n % solver.rebuild_vm_solver_every == 0 or vm_implicit_solver is None:
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

        # 2. Update Phase Field (Pore Growth)
        P_elec = 0.0
        if electrostatics_on and rank == 0:
            C_lipid = props.C_lipid
            maxwell_energy_density = 0.5 * C_lipid * (Vm**2)
            lipid_area = np.sum(H)
            if lipid_area > 1e-6:
                P_elec = np.sum(maxwell_energy_density * H) / lipid_area

        if phase_field_on:
            if rank == 0:
                sigma_total_map = phase.sigma_area + P_elec
                psi = phase_field_solver.evolve(psi, sigma_total_map)
            psi = comm.bcast(psi, root=0)
        
        # 3. Checkpointing / Saving Data
        if n % save_interval == 0 or n == nsteps - 1:
            t = (n + 1) * dt
            
            # --- Gather 3D Potential Field (phi) ---
            # We do this inside the loop now so we can save it to disk
            phi_elec_out = None
            if electrostatics_on:
                if isinstance(poisson, SpectralPoissonSolver):
                    phi_elec_out = poisson.phi if rank == 0 else None
                else:
                    # GAMG / PETSc Scatter
                    scat, phi_vec = PETSc.Scatter.toZero(poisson.x)
                    scat.begin(poisson.x, phi_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    scat.end(poisson.x, phi_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    if rank == 0:
                        phi_arr = phi_vec.getArray(readonly=True)
                        phi_elec_out = phi_arr.reshape(dom.Nz, dom.Ny, dom.Nx).transpose(2,1,0)
                    scat.destroy() # Clean up scatter context
            else:
                phi_elec_out = np.zeros((dom.Nx, dom.Ny, dom.Nz)) if rank == 0 else None

            # --- Save to Disk (Rank 0 only) ---
            eff_radius = calculate_pore_radius(smooth_step(psi), dx, dy)
            avg_Vm = float(np.mean(Vm)) if electrostatics_on else 0.0
            
            if rank == 0:
                time_points.append(t)
                avg_Vm_vs_time.append(avg_Vm)
                pore_radius_vs_time.append(eff_radius)
                
                print(f"Time: {t*1e9:8.2f} ns | Avg Vm: {avg_Vm:.4f} V | Pore R: {eff_radius*1e9:.2f} nm | Saving...")
                
                # Save numbered file
                filename = f"{output_dir}/step_{n:06d}.npz"
                np.savez_compressed(
                    filename, 
                    x=x, y=y, z=z, 
                    Vm=Vm, 
                    phi_elec=phi_elec_out, 
                    psi=psi, 
                    H=smooth_step(psi),
                    time=t,
                    avg_Vm=avg_Vm,
                    pore_radius=eff_radius
                )
            
            # Sync before continuing to ensure file I/O doesn't cause race conditions
            comm.Barrier()

    elapsed_time = time.time() - start_time
    if rank == 0:
        print(f"\nSimulation finished in {elapsed_time:.2f} seconds.")
        # Optional: Save a final summary file with full time histories
        np.savez_compressed(
            f"{output_dir}/summary_history.npz",
            time_points=np.array(time_points),
            avg_Vm_vs_time=np.array(avg_Vm_vs_time),
            pore_radius_vs_time=np.array(pore_radius_vs_time)
        )

def create_grid(dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Creates the computational grid."""
    dom.Nz = dom.Nz if dom.Nz % 2 else dom.Nz + 1
    x = np.linspace(-dom.Lx / 2, dom.Lx / 2, dom.Nx)
    y = np.linspace(-dom.Ly / 2, dom.Ly / 2, dom.Ny)
    z = np.linspace(-dom.Lz / 2, dom.Lz / 2, dom.Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz



# ---- single-function pore radius with smooth subcell/contour area ----
def calculate_pore_radius(H: np.ndarray, dx: float, dy: float,
                          level: float = 0.5, sub: int = 6) -> float:
    """
    r_eff = sqrt(A_pore/pi). Tries H=level contour area, else bilinear supersampling fallback.
    """
    try:
        contours = find_contours(H, level=level)
        if contours:
            A = 0.0
            for C in contours:
                x = C[:, 1] * dx
                y = C[:, 0] * dy
                A += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            return np.sqrt(max(A, 0.0) / np.pi)
    except Exception:
        pass

    ny, nx = H.shape
    xs = (np.arange(nx * sub) + 0.5) / sub - 0.5
    ys = (np.arange(ny * sub) + 0.5) / sub - 0.5
    Xs, Ys = np.meshgrid(xs, ys)

    x0 = np.floor(Xs).astype(int); y0 = np.floor(Ys).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1); y1 = np.clip(y0 + 1, 0, ny - 1)
    x0 = np.clip(x0, 0, nx - 1);     y0 = np.clip(y0, 0, ny - 1)

    Ia = H[y0, x0]; Ib = H[y0, x1]; Ic = H[y1, x0]; Id = H[y1, x1]
    wx = Xs - x0; wy = Ys - y0
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    Hs = top * (1 - wy) + bot * wy

    dA = (dx / sub) * (dy / sub)
    A = float(np.sum(1.0 - Hs) * dA)
    return np.sqrt(max(A, 0.0) / np.pi)

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

def blend_properties_derivative(psi: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends material properties and computes their derivatives w.r.t. psi."""
    H = smooth_step(psi)
    dH_dpsi = smooth_step_derivative(psi)
    
    G_lipid, G_pore = 1.0 / props.R_lipid, 1.0 / props.R_pore
    C_lipid, C_pore = props.C_lipid, props.C_pore

    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = C_pore + (C_lipid - C_pore) * H
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map

    dGm_dpsi = (G_lipid - G_pore) * dH_dpsi
    dCm_dpsi = (C_lipid - C_pore) * dH_dpsi
    dCeff_dpsi = dCm_dpsi

    return G_m_map, C_m_map, C_eff_map, dGm_dpsi, dCm_dpsi, dCeff_dpsi

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

