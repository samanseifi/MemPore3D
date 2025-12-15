from __future__ import annotations

import os
from typing import Optional

import numba
import numpy as np
from scipy import fft as spfft

from mempore3d.parameters import *
from mempore3d.solvers.poisson_solver import *
from mempore3d.solvers.leaky_dielectric_solver import *

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

    def evolve(self, psi_in: np.ndarray, sigma_total_map: np.ndarray) -> np.ndarray:
        if self.psi_hat is None:
            self.psi_hat = spfft.rfft2(psi_in, workers=self.workers)
            psi = psi_in
        else:
            psi = spfft.irfft2(self.psi_hat, s=psi_in.shape, workers=self.workers).real

        det_rhs = -self.params.mobility * (
            (self.params.line_tension / self.Cg) * (self._g_prime(psi) / self.params.transition_thickness) +
            sigma_total_map * smooth_step_derivative(psi)
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
    
def smooth_step(psi: np.ndarray) -> np.ndarray:
    """Computes a cubic Hermite smooth step function H(psi)."""
    psi_clipped = np.clip(psi, 0.0, 1.0)
    return psi_clipped**2 * (3.0 - 2.0 * psi_clipped)

def smooth_step_derivative(psi: np.ndarray) -> np.ndarray:
    psi_clipped = np.clip(psi, 0.0, 1.0)
    return 6.0 * psi_clipped * (1.0 - psi_clipped)