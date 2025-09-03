"""
Complete 2‑D phase‑field electroporation solver (Numba‑accelerated)
-------------------------------------------------------------------
* Conservative Cahn–Hilliard for membrane integrity φ(x,y,t)
* Self‑consistent electrostatics: solves ∇·(ε(φ)∇Φ)=0 each step
  - Periodic in x, Dirichlet in y: Φ(x,0)=0, Φ(x,Ly)=U_app
* Capacitive driving: ½ C_m U_app^2 H(φ) (default ON)
  - Optional permittivity‑mismatch drive (OFF by default)
* Semi‑implicit spectral update for CH with 2/3 de‑aliasing
* **Numba** speeds up the Poisson operator and electric‑field kernels (hot spots)
* Adaptive time‑stepping with backtracking
* Diagnostics: energies, mean(φ), A=<H(φ)>, dt history, Φ snapshots
* GIF writers for φ and (φ, Φ) evolution

Author: you (2025‑07‑25) — License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Callable
import numpy as np

# Optional Numba acceleration ---------------------------------------------------
try:
    import numba as nb
    NUMBA_OK = True
except Exception:
    nb = None  # type: ignore
    NUMBA_OK = False

# ----------------------------- small helpers ---------------------------------

def fft2(a: np.ndarray) -> np.ndarray:
    return np.fft.fftn(a)

def ifft2(ah: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(ah).real


def heaviside_smooth(phi: np.ndarray, eps: float) -> np.ndarray:
    """Smooth switch 0→pore, 1→membrane."""
    return 0.5 * (1.0 + np.tanh((phi - 0.5) / eps))

def dheaviside_smooth(phi: np.ndarray, eps: float) -> np.ndarray:
    sech = 1.0 / np.cosh((phi - 0.5) / eps)
    return 0.5 * (sech ** 2) / eps


# ----------------------------- parameters ------------------------------------
@dataclass
class CHParams:
    # CH physics
    gamma: float = 1.0        # line-tension scale
    ell: float = 0.01         # diffuse interface half-width
    kappa_b: float = 0.0      # bending-like penalty (optional)
    mobility: float = 1.0     # base mobility M0
    m_min: float = 1e-3       # residual to keep mobility bounded away from 0

    # Bias & area control (to avoid spinodal labyrinths)
    beta_bias: float = 0.02
    A_target: float = 0.995
    K_area: float = 5.0

    # Electrostatics
    solve_poisson: bool = True
    U_app: float = 0.3        # volts
    eps_mem: float = 2.5e-11  # F/m
    eps_wat: float = 7.0e-10  # F/m
    hs_eps: float = 0.02      # width of smooth switch H
    use_capacitive_term: bool = True
    use_permittivity_drive: bool = False
    C_m: float = 1.0e-2       # F/m^2
    cap_E2: Optional[float] = None  # cap on |E|^2; None disables capping
    # Electrode / geometry options
    symmetric_bc: bool = True           # if True: Φ(x,0)=-U/2, Φ(x,Ly)=+U/2; else 0..U
    y_mem_frac: float = 0.5             # membrane plane location as fraction of Ly (default mid-plane)

    # Domain/grid
    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 256
    Ny: int = 256

    # Adaptive time stepping
    dt_init: float = 2e-5
    dt_min: float = 2e-7
    dt_max: float = 2e-4
    grow: float = 1.25
    shrink: float = 0.5
    max_retries: int = 6

    # Run control
    nsteps: int = 10000
    save_every: int = 200

    # Noise
    use_noise: bool = False
    kBT: float = 0.0
    rng_seed: Optional[int] = None

    # Numerics / safety
    enforce_mean_fix: bool = True
    clip_bounds: Tuple[float, float] = (-0.2, 1.2)  # only for derivative evaluation
    blowup_threshold: float = 20.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 1000
    stab_alpha: float = 2.0   # linear stabilization added to implicit operator


# ----------------------------- Poisson operator -------------------------------
U_APP_GLOBAL: float = 0.0  # boundary value used by operator (set before each CG)

if NUMBA_OK:
    @nb.njit(cache=True, fastmath=True)
    def _apply_poisson_operator_nb(eps: np.ndarray, Phi: np.ndarray, dx: float, dy: float, U_app_val: float) -> np.ndarray:
        Nx, Ny = Phi.shape
        out = np.zeros_like(Phi)
        # x periodic flux; interior j = 1..Ny-2
        for i in range(Nx):
            ip = i + 1
            if ip == Nx:
                ip = 0
            im = i - 1
            if im < 0:
                im = Nx - 1
            for j in range(1, Ny - 1):
                eps_xp = 0.5 * (eps[i, j] + eps[ip, j])
                eps_xm = 0.5 * (eps[i, j] + eps[im, j])
                dPhix_p = (Phi[ip, j] - Phi[i, j]) / dx
                dPhix_m = (Phi[i, j] - Phi[im, j]) / dx
                # y faces (Dirichlet rows pinned at j=0 and j=Ny-1)
                eps_yp = 0.5 * (eps[i, j] + eps[i, j + 1])
                eps_ym = 0.5 * (eps[i, j] + eps[i, j - 1])
                dPhiy_p = (Phi[i, j + 1] - Phi[i, j]) / dy
                dPhiy_m = (Phi[i, j] - Phi[i, j - 1]) / dy
                out[i, j] = (eps_xp * dPhix_p - eps_xm * dPhix_m) / dx + (eps_yp * dPhiy_p - eps_ym * dPhiy_m) / dy
        # Dirichlet rows — identity rows for all i
        for i in range(Nx):
            out[i, 0] = Phi[i, 0]
            out[i, Ny - 1] = Phi[i, Ny - 1]
        return out
else:
    def _apply_poisson_operator_nb(eps: np.ndarray, Phi: np.ndarray, dx: float, dy: float, U_app_val: float) -> np.ndarray:  # type: ignore
        Nx, Ny = Phi.shape
        out = np.zeros_like(Phi)
        # x periodic
        eps_xp = 0.5 * (eps + np.roll(eps, -1, axis=0))
        eps_xm = np.roll(eps_xp, 1, axis=0)
        dPhix_p = (np.roll(Phi, -1, axis=0) - Phi) / dx
        dPhix_m = (Phi - np.roll(Phi, 1, axis=0)) / dx
        div_x = (eps_xp * dPhix_p - eps_xm * dPhix_m) / dx
        # y Dirichlet
        div_y = np.zeros_like(Phi)
        eps_yp = 0.5 * (eps[:, 1:-1] + eps[:, 2:])
        eps_ym = 0.5 * (eps[:, 1:-1] + eps[:, 0:-2])
        dPhiy_p = (Phi[:, 2:] - Phi[:, 1:-1]) / dy
        dPhiy_m = (Phi[:, 1:-1] - Phi[:, 0:-2]) / dy
        div_y[:, 1:-1] = (eps_yp * dPhiy_p - eps_ym * dPhiy_m) / dy
        out[:, 1:-1] = div_x[:, 1:-1] + div_y[:, 1:-1]
        out[:, 0] = Phi[:, 0]
        out[:, -1] = Phi[:, -1]
        return out


def apply_poisson_operator(eps: np.ndarray, Phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return _apply_poisson_operator_nb(eps, Phi, dx, dy, U_APP_GLOBAL)


def conjugate_gradient(A_mul: Callable[[np.ndarray], np.ndarray], b: np.ndarray, x0: np.ndarray,
                       tol: float, maxiter: int) -> Tuple[np.ndarray, int]:
    shape = b.shape
    b_vec = b.ravel(); x = x0.ravel().copy()

    def Amul_vec(v: np.ndarray) -> np.ndarray:
        return A_mul(v.reshape(shape)).ravel()

    r = b_vec - Amul_vec(x)
    p = r.copy(); rsold = float(np.dot(r, r))
    it = 0
    for it in range(1, maxiter + 1):
        Ap = Amul_vec(p)
        denom = float(np.dot(p, Ap))
        if abs(denom) < 1e-30:
            break
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(np.dot(r, r))
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x.reshape(shape), it


# --------- simple diagonal preconditioner for PCG ----------------------------

def build_diag_precond(eps: np.ndarray, dx: float, dy: float) -> np.ndarray:
    Nx, Ny = eps.shape
    D = np.zeros_like(eps)
    # interior
    ex_p = 0.5 * (eps + np.roll(eps, -1, axis=0))
    ex_m = np.roll(ex_p, 1, axis=0)
    # use same faces as operator; for y use slices
    ey_p = np.zeros_like(eps); ey_m = np.zeros_like(eps)
    ey_p[:, 1:-1] = 0.5 * (eps[:, 1:-1] + eps[:, 2:])
    ey_m[:, 1:-1] = 0.5 * (eps[:, 1:-1] + eps[:, 0:-2])
    D[:, 1:-1] = (ex_p[:, 1:-1] + ex_m[:, 1:-1]) / (dx * dx) + (ey_p[:, 1:-1] + ey_m[:, 1:-1]) / (dy * dy)
    # boundary rows: identity from Dirichlet
    D[:, 0] = 1.0; D[:, -1] = 1.0
    # guard
    D = np.maximum(D, 1e-12)
    return 1.0 / D


def pcg(A_mul: Callable[[np.ndarray], np.ndarray], b: np.ndarray, x0: np.ndarray,
        M_inv: np.ndarray, tol: float, maxiter: int) -> Tuple[np.ndarray, int]:
    shape = b.shape
    b_vec = b.ravel(); x = x0.ravel().copy()

    def Amul_vec(v: np.ndarray) -> np.ndarray:
        return A_mul(v.reshape(shape)).ravel()

    Minv = M_inv.ravel()
    r = b_vec - Amul_vec(x)
    z = Minv * r
    p = z.copy()
    rz_old = float(np.dot(r, z))
    it = 0
    for it in range(1, maxiter + 1):
        Ap = Amul_vec(p)
        denom = float(np.dot(p, Ap))
        if abs(denom) < 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < tol:
            break
        z = Minv * r
        rz_new = float(np.dot(r, z))
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x.reshape(shape), it

# Optional Numba electric-field kernel ----------------------------------------
if NUMBA_OK:
    @nb.njit(cache=True, fastmath=True)
    def electric_field_nb(Phi: np.ndarray, dx: float, dy: float, Ex: np.ndarray, Ey: np.ndarray) -> None:
        Nx, Ny = Phi.shape
        for i in range(Nx):
            ip = i + 1
            if ip == Nx:
                ip = 0
            im = i - 1
            if im < 0:
                im = Nx - 1
            for j in range(Ny):
                Ex[i, j] = - (Phi[ip, j] - Phi[im, j]) / (2.0 * dx)
        # y: one-sided on boundaries
        for i in range(Nx):
            Ey[i, 0] = - (Phi[i, 1] - Phi[i, 0]) / dy
            for j in range(1, Ny - 1):
                Ey[i, j] = - (Phi[i, j + 1] - Phi[i, j - 1]) / (2.0 * dy)
            Ey[i, Ny - 1] = - (Phi[i, Ny - 1] - Phi[i, Ny - 2]) / dy
else:
    def electric_field_nb(Phi: np.ndarray, dx: float, dy: float, Ex: np.ndarray, Ey: np.ndarray) -> None:  # type: ignore
        Ex[:] = - (np.roll(Phi, -1, axis=0) - np.roll(Phi, 1, axis=0)) / (2.0 * dx)
        Ey[:, 1:-1] = - (Phi[:, 2:] - Phi[:, 0:-2]) / (2.0 * dy)
        Ey[:, 0] = - (Phi[:, 1] - Phi[:, 0]) / dy
        Ey[:, -1] = - (Phi[:, -1] - Phi[:, -2]) / dy


# ----------------------------- Solver ----------------------------------------
class CahnHilliard2D:
    def __init__(self, p: CHParams):
        self.p = p
        if p.rng_seed is not None:
            np.random.seed(p.rng_seed)

        # Real-space grid
        self.x = np.linspace(0.0, p.Lx, p.Nx, endpoint=False)
        self.y = np.linspace(0.0, p.Ly, p.Ny, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Spectral k arrays for CH
        kx = 2.0 * np.pi * np.fft.fftfreq(p.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(p.Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.KX ** 2 + self.KY ** 2
        self.k4 = self.k2 ** 2
        kx_max = np.max(np.abs(kx)); ky_max = np.max(np.abs(ky))
        self.dealias = (np.abs(self.KX) <= (2.0 / 3.0) * kx_max) & (np.abs(self.KY) <= (2.0 / 3.0) * ky_max)

        # Fields
        self.phi: Optional[np.ndarray] = None
        self.Phi: Optional[np.ndarray] = None
        self.mean_phi0: Optional[float] = None

        # Adaptive dt
        self.dt = p.dt_init
        self.dt_hist: List[float] = []

        # Diagnostics
        self.energy_total: List[float] = []
        self.energy_bulk: List[float] = []
        self.energy_grad: List[float] = []
        self.energy_bend: List[float] = []
        self.energy_el: List[float] = []
        self.energy_area: List[float] = []
        self.mean_phi: List[float] = []
        self.A_hist: List[float] = []
        self.Phi_snaps: List[np.ndarray] = []
        self.Phi_mem_hist: List[np.ndarray] = []
        self.cg_iters_hist: List[int] = []
        self.cg_res_hist: List[float] = []

        # Buffers for E field (to avoid realloc in Numba path)
        self._Ex = np.zeros((p.Nx, p.Ny), dtype=np.float64)
        self._Ey = np.zeros((p.Nx, p.Ny), dtype=np.float64)

    # ------------------ initial condition -------------------------------------
    def set_initial_condition(self, kind: str = "uniform_noise", **kw) -> None:
        if kind == "uniform_noise":
            delta = kw.get("delta", 5e-4)
            phi0 = 1.0 + delta * (np.random.rand(self.p.Nx, self.p.Ny) - 0.5)
        elif kind == "seeded_pore":
            R0 = kw.get("R0", 0.02 * min(self.p.Lx, self.p.Ly))
            xc = kw.get("xc", 0.5 * self.p.Lx)
            yc = kw.get("yc", 0.5 * self.p.Ly)
            ell = kw.get("ell", self.p.ell)
            r = np.hypot(self.X - xc, self.Y - yc)
            # We want φ≈1 outside the pore (membrane) and φ≈0 inside the pore (water).
            # Use a smooth step that goes from 0 (r<R0) to 1 (r>R0).
            phi0 = 0.5 * (1.0 + np.tanh((r - R0) / (np.sqrt(2.0) * ell)))
        elif callable(kind):
            phi0 = kind(self.X, self.Y)
        else:
            raise ValueError("Unknown IC kind")
        self.phi = phi0.astype(np.float64)
        self.mean_phi0 = float(self.phi.mean())
        # Linear initial guess for Φ
        self.Phi = np.tile(np.linspace(0.0, self.p.U_app, self.p.Ny), (self.p.Nx, 1))

    # ------------------ mobility ---------------------------------------------
    def mobility_field(self, phi: np.ndarray) -> np.ndarray:
        return self.p.mobility * ((phi * (1.0 - phi)) ** 2 + self.p.m_min)

    # ------------------ μ contributions ---------------------------------------
    def dfdphi_bulk(self, phi: np.ndarray) -> np.ndarray:
        lo, hi = self.p.clip_bounds
        phi_safe = np.clip(phi, lo, hi)
        Wprime = 2.0 * phi_safe * (1.0 - phi_safe) * (1.0 - 2.0 * phi_safe)
        return (self.p.gamma / self.p.ell) * Wprime + self.p.beta_bias

    def dfdphi_area_penalty(self, phi: np.ndarray) -> np.ndarray:
        H = heaviside_smooth(phi, self.p.hs_eps)
        dH = dheaviside_smooth(phi, self.p.hs_eps)
        A = float(H.mean())
        return self.p.K_area * (A - self.p.A_target) * dH

    def dfdphi_electric(self, phi: np.ndarray, Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
        H = heaviside_smooth(phi, self.p.hs_eps)
        dH = dheaviside_smooth(phi, self.p.hs_eps)
        term = 0.0
        if self.p.use_permittivity_drive:
            deps_dphi = (self.p.eps_mem - self.p.eps_wat) * dH
            E2 = Ex ** 2 + Ey ** 2
            if self.p.cap_E2 is not None:
                E2 = np.minimum(E2, self.p.cap_E2)
            term += 0.5 * deps_dphi * E2
        if self.p.use_capacitive_term:
            term += 0.5 * self.p.C_m * dH * (self.p.U_app ** 2)
        return term

    def divergence_noise(self) -> np.ndarray:
        if not (self.p.use_noise and self.p.kBT > 0.0):
            return 0.0
        coef = np.sqrt(2.0 * self.p.kBT * self.p.mobility / self.dt)
        eta_x = coef * np.random.randn(self.p.Nx, self.p.Ny)
        eta_y = coef * np.random.randn(self.p.Nx, self.p.Ny)
        div_hat = 1j * (self.KX * fft2(eta_x) + self.KY * fft2(eta_y))
        return ifft2(div_hat)

    # ------------------ electrostatics ----------------------------------------
    def solve_poisson(self) -> None:
        if not self.p.solve_poisson:
            return
        H = heaviside_smooth(self.phi, self.p.hs_eps)
        eps_phi = self.p.eps_mem * H + self.p.eps_wat * (1.0 - H)
        Phi = self.Phi
        global U_APP_GLOBAL
        U_APP_GLOBAL = self.p.U_app

        def A_mul(P: np.ndarray) -> np.ndarray:
            return apply_poisson_operator(eps_phi, P, self.dx, self.dy)

        # Build RHS with Dirichlet values on boundary rows (keep operator linear)
        if self.p.symmetric_bc:
            low = -0.5 * self.p.U_app
            high = +0.5 * self.p.U_app
        else:
            low = 0.0
            high = self.p.U_app
        b = np.zeros_like(Phi)
        b[:, 0] = low
        b[:, -1] = high

        # build diagonal preconditioner once per step
        M_inv = build_diag_precond(eps_phi, self.dx, self.dy)
        # build diagonal preconditioner once per step
        M_inv = build_diag_precond(eps_phi, self.dx, self.dy)

        accept = False
        Phi_trial = Phi
        for attempt in range(2):
            Phi_new, iters = pcg(A_mul, b, Phi_trial, M_inv, self.p.cg_tol, self.p.cg_maxiter)
            # Enforce BCs explicitly
            Phi_new[:, 0] = low
            Phi_new[:, -1] = high
            # Residual check
            r = b - A_mul(Phi_new)
            res = float(np.linalg.norm(r))
            self.cg_iters_hist.append(iters)
            self.cg_res_hist.append(res)
            if np.isfinite(res) and res < 1e-6:
                accept = True
                break
            # fallback: use linear profile as new guess
            if attempt == 0:
                Phi_trial = np.tile(np.linspace(0.0, self.p.U_app, self.p.Ny), (self.p.Nx, 1))
        if not accept:
            raise RuntimeError(f"Poisson diverged: residual {res:.3e}; Φ range {Phi_new.min()}..{Phi_new.max()}")
        # Final sanity guard
        if Phi_new.max() > max(abs(high), abs(low)) * 10.0 or Phi_new.min() < -max(abs(high), abs(low)) * 10.0:
            raise RuntimeError(f"Poisson diverged: Φ range {Phi_new.min()}..{Phi_new.max()}")
        self.Phi = Phi_new

    def electric_field(self) -> Tuple[np.ndarray, np.ndarray]:
        Ex, Ey = self._Ex, self._Ey
        electric_field_nb(self.Phi, self.dx, self.dy, Ex, Ey)
        return Ex, Ey

    # ------------------ time stepping -----------------------------------------
    def attempt_step(self) -> Tuple[bool, np.ndarray]:
        # update Φ
        self.solve_poisson()
        Ex, Ey = self.electric_field()

        phi = self.phi
        fprime = (
            self.dfdphi_bulk(phi)
            + self.dfdphi_area_penalty(phi)
            + self.dfdphi_electric(phi, Ex, Ey)
        )

        # spectral semi-implicit with linear stabilization; use average mobility
        M_mean = float(self.mobility_field(phi).mean())
        fhat = fft2(fprime) * self.dealias
        phihat = fft2(phi) * self.dealias
        denom = 1.0 + self.dt * M_mean * self.k2 * (
            self.p.gamma * self.p.ell * self.k2 + self.p.kappa_b * self.k4 + self.p.stab_alpha
        )
        noise_hat = fft2(self.divergence_noise())
        numer = phihat - self.dt * M_mean * self.k2 * fhat + self.dt * noise_hat
        phi_new_hat = numer / denom
        phi_new_hat *= self.dealias
        phi_new = ifft2(phi_new_hat)

        # safety
        if not np.isfinite(phi_new).all():
            return False, phi
        if np.max(np.abs(phi_new)) > self.p.blowup_threshold:
            return False, phi
        if self.p.enforce_mean_fix:
            phi_new -= (phi_new.mean() - self.mean_phi0)
        return True, phi_new

    def step(self) -> None:
        for _ in range(self.p.max_retries):
            ok, phi_new = self.attempt_step()
            if ok:
                self.phi = phi_new
                self.dt = min(self.p.dt_max, self.dt * self.p.grow)
                self.dt_hist.append(self.dt)
                return
            else:
                self.dt = max(self.p.dt_min, self.dt * self.p.shrink)
        raise RuntimeError("Adaptive step failed after maximum retries.")

    # ------------------ diagnostics -------------------------------------------
    def energy_components(self) -> Tuple[float, float, float, float, float, float, float]:
        phi = self.phi
        gamma, ell = self.p.gamma, self.p.ell
        H = heaviside_smooth(phi, self.p.hs_eps)
        A = float(H.mean())

        W = phi ** 2 * (1.0 - phi) ** 2
        E_bulk = (gamma / ell) * float(W.mean()) + self.p.beta_bias * float(phi.mean())

        phihat = fft2(phi)
        E_grad = 0.5 * gamma * ell * float(np.sum(self.k2 * np.abs(phihat) ** 2)) / phi.size

        if self.p.kappa_b != 0.0:
            E_bend = 0.5 * self.p.kappa_b * float(np.sum(self.k4 * np.abs(phihat) ** 2)) / phi.size
        else:
            E_bend = 0.0

        # electric
        if self.p.solve_poisson:
            Ex, Ey = self.electric_field()
            eps_phi = self.p.eps_mem * H + self.p.eps_wat * (1.0 - H)
            E2 = Ex ** 2 + Ey ** 2
            E_el = -0.5 * float((eps_phi * E2).mean())
            if self.p.use_capacitive_term:
                E_el += 0.5 * self.p.C_m * A * (self.p.U_app ** 2)
        else:
            E_el = 0.0

        E_area = 0.5 * self.p.K_area * (A - self.p.A_target) ** 2
        E_total = (E_bulk + E_grad + E_bend + E_el + E_area) * (self.p.Lx * self.p.Ly)
        return E_total, E_bulk, E_grad, E_bend, E_el, E_area, A

    def record_diagnostics(self) -> None:
        Et, Eb, Eg, Ebend, Eel, Ea, A = self.energy_components()
        self.energy_total.append(Et)
        self.energy_bulk.append(Eb)
        self.energy_grad.append(Eg)
        self.energy_bend.append(Ebend)
        self.energy_el.append(Eel)
        self.energy_area.append(Ea)
        self.mean_phi.append(float(self.phi.mean()))
        self.A_hist.append(A)
        self.Phi_snaps.append(self.Phi.copy())
        # store membrane-plane potential line
        j_mem = int(round(self.p.y_mem_frac * (self.p.Ny - 1)))
        self.Phi_mem_hist.append(self.Phi[:, j_mem].copy())

    # ------------------ run ----------------------------------------------------
    def run(self) -> Dict[str, List]:
        assert self.phi is not None, "Initial condition not set."
        snaps: List[np.ndarray] = []
        for n in range(1, self.p.nsteps + 1):
            self.step()
            if n % self.p.save_every == 0:
                self.record_diagnostics()
                snaps.append(self.phi.copy())
                Phi_min, Phi_max = float(self.Phi.min()), float(self.Phi.max())
                j_mem = int(round(self.p.y_mem_frac * (self.p.Ny - 1)))
                Phi_mem_min = float(self.Phi[:, j_mem].min()); Phi_mem_max = float(self.Phi[:, j_mem].max())
                print(f"n={n:6d}  Φ:[{Phi_min:.3g},{Phi_max:.3g}]  Φ_mem:[{Phi_mem_min:.3g},{Phi_mem_max:.3g}]  meanφ={self.mean_phi[-1]:.6f}  A={self.A_hist[-1]:.6f}  dt={self.dt:.2e}  cg={self.cg_iters_hist[-1]}  r={self.cg_res_hist[-1]:.2e}")
        return {
            "phi": snaps,
            "Phi": self.Phi_snaps,
            "energy_total": self.energy_total,
            "energy_bulk": self.energy_bulk,
            "energy_grad": self.energy_grad,
            "energy_bend": self.energy_bend,
            "energy_el": self.energy_el,
            "energy_area": self.energy_area,
            "mean_phi": self.mean_phi,
            "A_hist": self.A_hist,
            "dt_hist": self.dt_hist,
            "cg_iters": self.cg_iters_hist,
            "cg_res": self.cg_res_hist,
            "Phi_mem": self.Phi_mem_hist,
        }


# ----------------------------- GIF utilities ---------------------------------

def save_phi_gif(snaps: List[np.ndarray], Lx: float, Ly: float, fname: str = "phi_evolution.gif", fps: int = 15) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        import io
    except Exception as exc:
        print("GIF skipped (missing packages):", exc)
        return
    frames = []
    vmin = min(float(s.min()) for s in snaps)
    vmax = max(float(s.max()) for s in snaps)
    for s in snaps:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(s.T, origin="lower", extent=[0, Lx, 0, Ly], vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(fname, frames, fps=fps, loop=0)
    print(f"Saved {fname}")


def save_phiPhi_gif(phi_snaps: List[np.ndarray], Phi_snaps: List[np.ndarray], Lx: float, Ly: float,
                     fname: str = "phi_Phi_evolution.gif", fps: int = 12) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        import io
    except Exception as exc:
        print("GIF skipped (missing packages):", exc)
        return
    frames = []
    vmin_phi = min(float(s.min()) for s in phi_snaps)
    vmax_phi = max(float(s.max()) for s in phi_snaps)
    vmin_Phi = min(float(S.min()) for S in Phi_snaps)
    vmax_Phi = max(float(S.max()) for S in Phi_snaps)
    for s, S in zip(phi_snaps, Phi_snaps):
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        axs[0].imshow(s.T, origin="lower", extent=[0, Lx, 0, Ly], vmin=vmin_phi, vmax=vmax_phi, cmap="viridis")
        axs[0].set_title("φ"); axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[1].imshow(S.T, origin="lower", extent=[0, Lx, 0, Ly], vmin=vmin_Phi, vmax=vmax_Phi, cmap="plasma")
        axs[1].set_title("Φ (V)"); axs[1].set_xticks([]); axs[1].set_yticks([])
        plt.tight_layout()
        import io as _io
        buf = _io.BytesIO()
        plt.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(fname, frames, fps=fps, loop=0)
    print(f"Saved {fname}")


# ----------------------------- example driver --------------------------------
if __name__ == "__main__":
    p = CHParams(rng_seed=0, use_noise=False, U_app=0.3,
                 use_capacitive_term=True, use_permittivity_drive=False)
    sim = CahnHilliard2D(p)
    # Start with a tiny seeded pore to probe deterministic growth
    sim.set_initial_condition("seeded_pore", R0=0.01)
    out = sim.run()

    # Try to save GIFs (optional)
    save_phi_gif(out["phi"], p.Lx, p.Ly, "phi_evolution.gif", fps=15)
    if len(out["Phi"]) == len(out["phi"]):
        save_phiPhi_gif(out["phi"], out["Phi"], p.Lx, p.Ly, "phi_Phi_evolution.gif", fps=12)

    # Quick plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.imshow(out["phi"][-1].T, origin="lower", extent=[0, p.Lx, 0, p.Ly], cmap="viridis")
        plt.colorbar(label="φ")
        plt.title("Final φ")
        plt.tight_layout()

        plt.figure(figsize=(5, 4))
        plt.imshow(out["Phi"][-1].T, origin="lower", extent=[0, p.Lx, 0, p.Ly], cmap="plasma")
        plt.colorbar(label="Φ (V)")
        plt.title("Final Φ (plates at y=0 and y=Ly)")
        plt.tight_layout()

        plt.figure(figsize=(6, 4))
        plt.plot(out["energy_total"], label="Total")
        plt.plot(out["energy_bulk"], label="Bulk")
        plt.plot(out["energy_grad"], label="Grad")
        if p.kappa_b != 0.0:
            plt.plot(out["energy_bend"], label="Bend")
        plt.plot(out["energy_el"], label="Electric")
        plt.plot(out["energy_area"], label="Area")
        plt.xlabel("Saved index")
        plt.ylabel("Energy")
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(6, 3))
        plt.plot(out["A_hist"], label="A=<H(φ)>")
        plt.plot(out["mean_phi"], label="mean(φ)")
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(6, 3))
        plt.plot(out["dt_hist"]) ; plt.ylabel("dt") ; plt.tight_layout()
        plt.show()
    except Exception as exc:
        print("Plot skipped:", exc)
