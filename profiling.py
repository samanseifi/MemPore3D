from __future__ import annotations

import os
import time
import cProfile
import pstats

import numpy as np

# --- your imports (as requested)
from mempore3d.core import *
from mempore3d.parameters import *


def build_inputs():
    Lx, Ly, Lz = 1000e-9, 1000e-9, 2000e-9
    Nx, Ny, Nz = 128, 128, 129  # Nz should be odd to center membrane
    custom_domain = Domain(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz)

    # Calculating dx for transition thickness
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    dx = x[1] - x[0]

    solver_params = SolverParams(
        save_frames=80,
        implicit_dt_multiplier=5.0,
        rebuild_vm_solver_every=50,
        n_tau_total=1.0,
        poisson_solver="spectral",
    )

    pore_growth_params = PhaseFieldParams(
        initial_state="pore",
        initial_pore_radius=88.0e-9,
        transition_thickness=1.5 * dx,
        sigma_area=0.0,
        mobility=5.0e8,
        line_tension=1.5e-10,
        evolve="on",
    )

    high_voltage = Electrostatics(V_applied=0.85)
    thermal_params = None

    return custom_domain, solver_params, pore_growth_params, high_voltage, thermal_params


def run_once():
    dom_in, solver, phase, elec, thermal = build_inputs()
    print("--- Starting Simulation: Voltage-Driven Pore Expansion ---")
    simulate_membrane_charging(
        dom_in=dom_in,
        solver=solver,
        phase=phase,
        elec=elec,
        thermal=thermal,
    )


def profile_cprofile(
    prof_path: str = "mempore3d_run.prof",
    top_n: int = 60,
    do_warmup: bool = True,
):
    """
    Profiles the run with cProfile and prints the highest-cost functions.

    - do_warmup=True: runs once without profiling first (useful if you have numba JIT
      or first-time FFT planning, so the profile reflects steady-state runtime).
    """
    if do_warmup:
        print("\n=== WARMUP (not profiled): compile JIT / plan FFT / allocate caches ===")
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        print(f"Warmup wall time: {t1 - t0:.3f} s\n")

    print("=== PROFILED RUN ===")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    run_once()
    t1 = time.perf_counter()
    pr.disable()

    print(f"Profiled wall time: {t1 - t0:.3f} s")

    # Save raw profile
    pr.dump_stats(prof_path)
    print(f"Saved cProfile stats to: {os.path.abspath(prof_path)}")

    # Pretty print summaries
    stats = pstats.Stats(pr).strip_dirs()

    print("\n--- TOP BY SELF TIME (tottime) ---")
    stats.sort_stats("tottime").print_stats(top_n)

    print("\n--- TOP BY CUMULATIVE TIME (cumtime) ---")
    stats.sort_stats("cumtime").print_stats(top_n)

    print("\n--- CALLERS (who called the heavy stuff) ---")
    stats.sort_stats("tottime").print_callers(30)

    print("\n--- CALLEES (what heavy stuff calls) ---")
    stats.sort_stats("tottime").print_callees(30)

    print("\nTip: visualize with snakeviz:")
    print("  pip install snakeviz")
    print(f"  snakeviz {prof_path}")


if __name__ == "__main__":
    # You can tweak these:
    profile_cprofile(
        prof_path="mempore3d_run.prof",
        top_n=60,
        do_warmup=True,   # set False if you explicitly WANT to see compile/planning time
    )
