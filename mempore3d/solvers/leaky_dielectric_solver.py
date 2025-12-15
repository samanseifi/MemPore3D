from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import numba


# =============================================================================
# Data model
# =============================================================================

@dataclass(frozen=True, slots=True)
class Domain2D:
    """
    2D domain with a regular Cartesian grid.

    Conventions used throughout this file:
    - Arrays are shaped (Ny, Nx) = (y, x).
    - Linear index p = i + Nx*j (x is fastest).
    """
    Nx: int
    Ny: int


# =============================================================================
# Fast sparse operator assembly (5-point stencil, Neumann BC)
# =============================================================================

def _as_flat_c_order(a: np.ndarray) -> np.ndarray:
    """Return a contiguous float64 1D view, flattened in C-order."""
    return np.ascontiguousarray(a, dtype=np.float64).ravel(order="C")


def build_vm_implicit_operator(
    dom: Domain2D,
    C_eff_map: np.ndarray,     # (Ny, Nx)
    G_m_map: np.ndarray,       # (Ny, Nx)
    H: np.ndarray,             # (Ny, Nx)
    dt: float,
    dx: float,
    D_V: float,
    sigma_e: float,
    d: float,
) -> sp.csc_matrix:
    """
    Build A for the semi-implicit 2D transmembrane potential update:
        (C_eff + dt*G_total) Vm^{n+1} - dt*D_V * Lap(Vm^{n+1}) = RHS

    - 5-point Laplacian, uniform dx=dy=dx.
    - Neumann (zero normal derivative) at all boundaries implemented by
      reflecting the missing neighbor, which is equivalent to adding the
      missing off-diagonal weight onto the diagonal.

    Returns:
        A in CSC format for fast factorization/solves.

    Notes:
        For performance and correctness at Nx,Ny in [64,512], we avoid COO
        assembly and instead build via sparse.diags with 5 diagonals.
    """
    Nx, Ny = dom.Nx, dom.Ny
    if C_eff_map.shape != (Ny, Nx) or G_m_map.shape != (Ny, Nx) or H.shape != (Ny, Nx):
        raise ValueError(f"Expected (Ny,Nx)=({Ny},{Nx}) arrays for C_eff_map/G_m_map/H.")

    N = Nx * Ny

    # --- Conductances
    base_G_sc = 2.0 * sigma_e / d
    G_sc_map = 5.0 * base_G_sc * (1.0 - H)
    G_total_map = G_m_map + G_sc_map

    # --- Flatten (must match p = i + Nx*j)
    C = _as_flat_c_order(C_eff_map)        # length N
    G = _as_flat_c_order(G_total_map)      # length N

    dx2 = dx * dx
    w = -dt * D_V / dx2          # off-diagonal Laplacian weight
    lap_diag = -4.0 * w          # = 4*dt*D_V/dx^2

    # --- Main diagonal baseline
    diag = C + dt * G + lap_diag

    # --- Neumann boundary correction: add w for each missing neighbor
    # left boundary (i = 0)
    diag[0:N:Nx] += w
    # right boundary (i = Nx-1)
    diag[Nx - 1:N:Nx] += w
    # bottom boundary (j = 0)
    diag[0:Nx] += w
    # top boundary (j = Ny-1)
    diag[N - Nx:N] += w

    # --- Off diagonals
    # x-neighbors: +/- 1, but prevent wrap-around between rows
    off_x = np.full(N - 1, w, dtype=np.float64)
    row_breaks = np.arange(Nx - 1, N - 1, Nx)  # last col of each row
    off_x[row_breaks] = 0.0

    # y-neighbors: +/- Nx
    off_y = np.full(N - Nx, w, dtype=np.float64)

    A = sp.diags(
        diagonals=[off_y, off_x, diag, off_x, off_y],
        offsets=[-Nx, -1, 0, 1, Nx],
        shape=(N, N),
        format="csc",
        dtype=np.float64,
    )
    return A


# =============================================================================
# Prefactorized solver wrapper
# =============================================================================

class ImplicitVMSolver:
    """
    Prefactorized implicit Vm solver for:
        A Vm^{n+1} = b

    Conventions:
    - Vm is shaped (Ny, Nx).
    - b should be either (Ny, Nx) or flat length Nx*Ny.
    """

    def __init__(
        self,
        dom: Domain2D,
        C_eff_map: np.ndarray,
        G_m_map: np.ndarray,
        H: np.ndarray,
        dt: float,
        dx: float,
        D_V: float,
        sigma_e: float,
        d: float,
    ):
        self.dom = dom
        self.Nx, self.Ny = dom.Nx, dom.Ny

        A = build_vm_implicit_operator(dom, C_eff_map, G_m_map, H, dt, dx, D_V, sigma_e, d)
        self._solve_flat: Callable[[np.ndarray], np.ndarray] = factorized(A)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve A x = b.

        Parameters
        ----------
        b : np.ndarray
            Either shape (Ny, Nx) or flat shape (Nx*Ny,).

        Returns
        -------
        Vm_new : np.ndarray
            Shape (Ny, Nx).
        """
        if b.ndim == 2:
            if b.shape != (self.Ny, self.Nx):
                raise ValueError(f"b must have shape (Ny,Nx)=({self.Ny},{self.Nx})")
            b_flat = _as_flat_c_order(b)
        elif b.ndim == 1:
            if b.size != self.Nx * self.Ny:
                raise ValueError(f"b must have length Nx*Ny={self.Nx*self.Ny}")
            b_flat = np.ascontiguousarray(b, dtype=np.float64)
        else:
            raise ValueError("b must be 1D or 2D")

        x_flat = self._solve_flat(b_flat)
        return x_flat.reshape((self.Ny, self.Nx), order="C")


# =============================================================================
# Fast ionic current density evaluation (in-place Numba)
# =============================================================================

@numba.njit(cache=True, fastmath=True)
def calculate_J_from_phi_inplace(J_out, phi, k_plane_index, sigma_e, dz):
    """
    Compute ionic current density through a z-plane:
        J = sigma_e * (phi[:,:,k+1] - phi[:,:,k]) / dz

    Parameters
    ----------
    J_out : (Ny, Nx) float64 output array
    phi   : (Ny, Nx, Nz) float64 array
    k_plane_index : int, must satisfy 0 <= k < Nz-1
    sigma_e : float
    dz : float
    """
    Ny, Nx, Nz = phi.shape
    k = k_plane_index
    if k < 0 or k + 1 >= Nz:
        raise ValueError("k_plane_index out of bounds for phi.shape[2].")

    scale = sigma_e / dz
    for j in range(Ny):
        for i in range(Nx):
            J_out[j, i] = scale * (phi[j, i, k + 1] - phi[j, i, k])


def calculate_J_from_phi(phi: np.ndarray, k_plane_index: int, sigma_e: float, dz: float,
                         out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convenience wrapper around the in-place Numba kernel.
    """
    if phi.ndim != 3:
        raise ValueError("phi must be 3D (Ny, Nx, Nz).")

    Ny, Nx, _ = phi.shape
    if out is None:
        out = np.empty((Ny, Nx), dtype=np.float64)
    else:
        if out.shape != (Ny, Nx):
            raise ValueError(f"out must have shape (Ny,Nx)=({Ny},{Nx})")
        if out.dtype != np.float64:
            raise ValueError("out must be float64 for best performance")

    calculate_J_from_phi_inplace(out, np.ascontiguousarray(phi, dtype=np.float64),
                                 k_plane_index, float(sigma_e), float(dz))
    return out
