# spectral_membrane_charging_fftw.py
#
# Fast membrane-charging loop using pyFFTW (threaded) for 2D FFTs.
# Periodic in x,y; Dirichlet planes in z. We avoid solving φ everywhere:
# compute J_elec directly in spectral space from Vm (closed-form, BOTH
# electrolytes included), then update Vm (semi-implicit by default).
#
# J_hat(k) for k≠0:  J = -(1/2) * [σ κ (coth(κΔt)+coth(κΔb))] * Vm_hat
# DC mode:           J_dc = (σ/2) * (Vapp - Vm_dc) * (1/Δt + 1/Δb)

from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# -------------------------
# Optional pyFFTW backend
# -------------------------
_USE_FFTW = True
try:
    import pyfftw
except Exception:
    _USE_FFTW = False

def _get_nthreads():
    # Respect env if user set it; else use all logical cores.
    s = os.environ.get("FFTW_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if s and s.isdigit():
        return max(1, int(s))
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1

# -------------------------
# FFT wrapper (rfft2/irfft2)
# -------------------------
class RFFT2:
    """
    Real 2D FFT wrapper. Uses pyFFTW if available, otherwise NumPy.
    Keeps pre-allocated aligned arrays and FFTW plans.
    """
    def __init__(self, Nx: int, Ny: int, nthreads: int | None = None, wisdom: bool = True):
        self.Nx, self.Ny = Nx, Ny
        self.nthreads = nthreads or _get_nthreads()
        self.uses_fftw = _USE_FFTW

        if self.uses_fftw:
            if wisdom:
                try:
                    pyfftw.import_wisdom(pyfftw.export_wisdom())
                except Exception:
                    pass

            # aligned working buffers
            self._in   = pyfftw.empty_aligned((Nx, Ny), dtype='float64')
            self._out  = pyfftw.empty_aligned((Nx, Ny//2 + 1), dtype='complex128')
            self._rin  = pyfftw.empty_aligned((Nx, Ny//2 + 1), dtype='complex128')
            self._rout = pyfftw.empty_aligned((Nx, Ny), dtype='float64')

            # build plans (NOTE: no overwrite_input kw)
            self._fw = pyfftw.builders.rfft2(
                self._in, s=(Nx, Ny),
                threads=self.nthreads,
                planner_effort='FFTW_MEASURE'
            )
            self._bw = pyfftw.builders.irfft2(
                self._rin, s=(Nx, Ny),
                threads=self.nthreads,
                planner_effort='FFTW_MEASURE'
            )
        else:
            # NumPy fallback
            self._in   = np.empty((Nx, Ny), dtype=np.float64)
            self._out  = np.empty((Nx, Ny//2 + 1), dtype=np.complex128)
            self._rin  = np.empty((Nx, Ny//2 + 1), dtype=np.complex128)
            self._rout = np.empty((Nx, Ny), dtype=np.float64)

    @property
    def inarray(self):  return self._in
    @property
    def outarray(self): return self._out

    def rfft2(self, real2d: np.ndarray) -> np.ndarray:
        if self.uses_fftw:
            np.copyto(self._in, real2d)
            return self._fw()  # fills and returns self._out
        else:
            self._out[...] = np.fft.rfft2(real2d)
            return self._out

    def irfft2(self, hat2d: np.ndarray) -> np.ndarray:
        if self.uses_fftw:
            np.copyto(self._rin, hat2d)
            return self._bw()  # fills and returns self._rout
        else:
            self._rout[...] = np.fft.irfft2(hat2d, s=(self.Nx, self.Ny)).real
            return self._rout

# -------------------------
# Physical constants
# -------------------------
EPS0 = 8.8541878128e-12
EPS_W = 80.0 * EPS0

# -------------------------
# Parameters
# -------------------------
@dataclass
class Domain:
    Lx: float = 10_000e-9
    Ly: float = 10_000e-9
    Lz: float = 20_000e-9
    Nx: int = 128
    Ny: int = 128
    Nz: int = 129   # odd → centered membrane region

@dataclass
class MembraneProps:
    R_lipid: float = 1e7     # Ohm·m^2
    C_lipid: float = 1e-2    # F/m^2
    R_pore: float = 1e-1     # Ohm·m^2
    C_pore: float = 1e-9     # F/m^2

@dataclass
class PhaseFieldParams:
    pore_radius: float = 500e-9
    transition_thickness: float | None = None  # defaults to 2*dx

@dataclass
class Electrostatics:
    sigma_e: float = 1.0     # S/m
    V_applied: float = 0.5   # V

@dataclass
class SolverParams:
    surface_diffusion: bool = True
    D_V: float = 0.0              # if 0, set adaptively
    dt_safety: float = 0.01
    n_tau_total: float = 8.0
    save_frames: int = 20
    print_every: int = 25
    nthreads_fft: Optional[int] = None  # threads for FFTW

# -------------------------
# Grid & helpers
# -------------------------
def create_grid(dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    Nx, Ny = dom.Nx, dom.Ny
    Nz = dom.Nz if dom.Nz % 2 else dom.Nz + 1
    # Periodic in x,y → avoid endpoint to fit FFT grid naturally
    x = np.linspace(-dom.Lx/2, dom.Lx/2, Nx, endpoint=False)
    y = np.linspace(-dom.Ly/2, dom.Ly/2, Ny, endpoint=False)
    z = np.linspace(-dom.Lz/2, dom.Lz/2, Nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return x, y, z, dx, dy, dz

def smooth_step(psi: np.ndarray) -> np.ndarray:
    return psi**2 * (3.0 - 2.0*psi)

def build_phase_field(x: np.ndarray, y: np.ndarray, pore_radius: float, ell: float) -> Tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    psi = 0.5*(1.0 + np.tanh((r - pore_radius)/(np.sqrt(2.0)*ell)))
    H = smooth_step(psi)
    return psi, H

def blend_properties(H: np.ndarray, props: MembraneProps, dom: Domain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    G_lipid, G_pore = 1.0/props.R_lipid, 1.0/props.R_pore
    G_m_map = G_pore + (G_lipid - G_pore) * H
    C_m_map = props.C_pore + (props.C_lipid - props.C_pore) * H
    C_bath = 2.0 * EPS_W / dom.Lz
    C_eff_map = C_bath + C_m_map
    return G_m_map, C_m_map, C_eff_map

def estimate_time_step(G_m_map: np.ndarray, C_eff_map: np.ndarray, elec: Electrostatics,
                       dom: Domain, solver: SolverParams) -> Tuple[float, float, int]:
    A = 2.0 * elec.sigma_e / dom.Lz
    rate_max = (A + np.max(G_m_map)) / np.min(C_eff_map)
    tau_local_min = 1.0 / rate_max
    dt = solver.dt_safety * tau_local_min
    tau_report = np.max(C_eff_map) / (A + np.max(G_m_map))
    total_time = solver.n_tau_total * tau_report
    nsteps = int(np.ceil(total_time / dt))
    return dt, total_time, nsteps

def laplacian2d_neumann_like(field: np.ndarray, dx: float) -> np.ndarray:
    # For optional surface smoothing of Vm (acts like Neumann reflection at edges)
    Nx, Ny = field.shape
    lap = np.empty_like(field)
    dx2 = dx*dx
    for i in range(Nx):
        im1 = i-1 if i>0 else 1
        ip1 = i+1 if i<Nx-1 else Nx-2
        for j in range(Ny):
            jm1 = j-1 if j>0 else 1
            jp1 = j+1 if j<Ny-1 else Ny-2
            lap[i,j] = (field[ip1,j]+field[im1,j]+field[i,jp1]+field[i,jm1]-4*field[i,j])/dx2
    return lap

def update_vm_semiimplicit(Vm, G_m, C_eff, H, dt, dx, sigma, Vapp, Lz,
                           surface_diffusion, D_V, J_elec):
    base_G_sc = 2.0 * sigma / Lz
    # Consider reducing alpha if Vm looks clamped; alpha=50 is conservative.
    G_sc = 50.0 * base_G_sc * (1.0 - H)
    Gtot = G_m + G_sc
    num = C_eff*Vm + dt*(J_elec)           # J at n+1 (from spectral formula this step)
    den = C_eff + dt*Gtot
    Vm_new = num/den
    if surface_diffusion and D_V>0:
        Vm_new += dt * D_V * laplacian2d_neumann_like(Vm_new, dx)
    return Vm_new

# -------------------------
# Spectral J-only (pyFFTW)
# -------------------------
class SpectralJComputer:
    """
    Computes J_elec from Vm without building φ, including BOTH electrolytes:
      Non-DC: J_hat = -(1/2)*σ*κ*(coth(κΔt)+coth(κΔb)) * Vm_hat
      DC:     J_dc  =  (σ/2)*(Vapp - Vm_dc)*(1/Δt + 1/Δb)
    Vectorized with rFFT2/irFFT2 and precomputed κ and coth terms.
    """
    def __init__(self, dom: Domain, z_plus_idx: int, z_top_idx: int, nthreads_fft: Optional[int] = None):
        Nx, Ny = dom.Nx, dom.Ny
        Lx, Ly = dom.Lx, dom.Ly
        self.Nx, self.Ny = Nx, Ny
        self.z_plus_idx = z_plus_idx
        self.z_top_idx = z_top_idx

        # rFFT frequencies: full in x, half in y
        kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)            # shape (Nx,)
        ky = 2*np.pi*np.fft.rfftfreq(Ny, d=Ly/Ny)           # shape (Ny//2+1,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')         # (Nx, Ny//2+1)
        self.K = np.sqrt(KX*KX + KY*KY)

        # FFT plans/buffers
        self.fft = RFFT2(Nx, Ny, nthreads=nthreads_fft)

        # Placeholders allocated once
        self._Vm_hat = self.fft.outarray      # complex (Nx, Ny//2+1)
        self._J_hat  = np.empty_like(self._Vm_hat)

        # caches for coth terms
        self._coth_cache_valid = False
        self._Delta_top_cached = None
        self._Delta_bot_cached = None

    def compute_J(self, Vm: np.ndarray, sigma: float, V_applied: float,
                  Delta_top: float, Delta_bot: float) -> np.ndarray:
        """
        Correct membrane current J_elec including top and bottom electrolytes.
        """
        # cache coth for both distances
        if (self._Delta_top_cached != Delta_top or
            self._Delta_bot_cached != Delta_bot or
            not self._coth_cache_valid):
            K = self.K
            self._coth_t = np.zeros_like(K)
            self._coth_b = np.zeros_like(K)
            mask = (K > 0)
            self._coth_t[mask] = 1.0 / np.tanh(K[mask] * Delta_top)
            self._coth_b[mask] = 1.0 / np.tanh(K[mask] * Delta_bot)
            self._Delta_top_cached = Delta_top
            self._Delta_bot_cached = Delta_bot
            self._coth_cache_valid = True

        Vm_hat = self.fft.rfft2(Vm)  # (Nx, Ny//2+1)

        # Non-DC modes: Yacc = σ κ (coth_t + coth_b)
        Yacc = sigma * self.K * (self._coth_t + self._coth_b)
        self._J_hat[:] = -0.5 * Yacc * Vm_hat

        # DC correction (overwrite [0,0])
        Vm_dc = Vm_hat[0, 0].real
        J_dc = 0.5 * sigma * (V_applied - Vm_dc) * (1.0/Delta_top + 1.0/Delta_bot)
        self._J_hat[0, 0] = J_dc

        return self.fft.irfft2(self._J_hat)

# -------------------------
# Main driver
# -------------------------
def simulate_membrane_charging(
    dom: Domain | None = None,
    props: MembraneProps | None = None,
    phase: PhaseFieldParams | None = None,
    elec: Electrostatics | None = None,
    solver: SolverParams | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dom = dom if dom is not None else Domain()
    props = props if props is not None else MembraneProps()
    phase = phase if phase is not None else PhaseFieldParams()
    elec = elec if elec is not None else Electrostatics()
    solver = solver if solver is not None else SolverParams()

    x, y, z, dx, dy, dz = create_grid(dom)
    Nz = z.size

    # Membrane bracketing planes
    km_minus = Nz//2 - 1
    km_plus  = Nz//2 + 1

    # distances from membrane planes to electrodes
    Delta_top = z[-1]       - z[km_plus]   # meters
    Delta_bot = z[km_minus] - z[0]         # meters

    ell = phase.transition_thickness if phase.transition_thickness is not None else (2.0*dx)
    psi, H = build_phase_field(x, y, phase.pore_radius, ell)
    G_m_map, C_m_map, C_eff_map = blend_properties(H, props, dom)
    dt, total_time, nsteps = estimate_time_step(G_m_map, C_eff_map, elec, dom, solver)

    if solver.surface_diffusion and solver.D_V == 0.0:
        solver.D_V = 0.05 * dx*dx / dt

    # pyFFTW-based spectral J
    Jcomp = SpectralJComputer(dom, z_plus_idx=km_plus, z_top_idx=Nz-1, nthreads_fft=solver.nthreads_fft)

    Vm = np.zeros((dom.Nx, dom.Ny), dtype=np.float64)

    time_points = []
    avg_Vm_vs_time = []

    save_interval = max(1, nsteps // solver.save_frames)

    print(f"[FFTs] Using {'pyFFTW' if _USE_FFTW else 'NumPy'} with {Jcomp.fft.nthreads if _USE_FFTW else 1} thread(s)")
    print(f"Grid: {dom.Nx}x{dom.Ny}x{Nz} (dx={dx*1e9:.2f} nm), periodic x,y")
    print(f"Pore R={phase.pore_radius*1e9:.1f} nm, trans.={ell*1e9:.1f} nm")
    print(f"dt={dt*1e9:.3f} ns, steps={nsteps}, τ≈{(total_time/solver.n_tau_total)*1e9:.2f} ns")

    for n in range(nsteps):
        # 1) Compute J_elec directly from Vm (two-sided spectral closed form)
        J_elec = Jcomp.compute_J(Vm, elec.sigma_e, elec.V_applied, Delta_top, Delta_bot)

        # 2) Semi-implicit Vm update (unconditionally stable to the sink term)
        Vm = update_vm_semiimplicit(
            Vm, G_m_map, C_eff_map, H, dt, dx,
            elec.sigma_e, elec.V_applied, dom.Lz,
            solver.surface_diffusion, solver.D_V, J_elec
        )

        if (n % save_interval == 0) or (n == nsteps-1):
            t = (n+1)*dt
            time_points.append(t)
            avg_Vm_vs_time.append(float(np.mean(Vm)))
            if ((n % max(1, solver.print_every)) == 0) or (n == nsteps-1):
                print(f"Step {n+1:>6}/{nsteps}  t={t*1e9:8.2f} ns  <Vm>={avg_Vm_vs_time[-1]:.4f} V")

    # Return arrays for plotting/saving
    return x, y, z, Vm

# -------------------------
# Run example
# -------------------------
if __name__ == "__main__":
    # Tip: set threads, e.g.:
    #   export FFTW_NUM_THREADS=16
    #   export OMP_NUM_THREADS=16
    dom = Domain(Nx=64, Ny=64, Nz=65)
    solver = SolverParams(save_frames=40, print_every=50, nthreads_fft=16)
    x, y, z, Vm = simulate_membrane_charging(dom=dom, solver=solver)
    np.savez_compressed("membrane_fftw_Jonly.npz", x=x, y=y, z=z, Vm=Vm)
