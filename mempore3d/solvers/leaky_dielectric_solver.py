from __future__ import annotations

import numba
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized



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
    G_sc_map = 5.0 * base_G_sc * (1.0 - H)
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