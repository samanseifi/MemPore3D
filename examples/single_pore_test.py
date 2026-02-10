from mempore3d_petsc import *

Lx, Ly, Lz = 1000e-9, 1000e-9, 2000e-9
Nx, Ny, Nz = 128, 128, 129  # Nz should be odd to center membrane
custom_domain = Domain(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz)

# Calculating dx for transition thickness    
x = np.linspace(-Lx / 2, Lx / 2, Nx)
dx = x[1] - x[0]

print(dx)

# We need a reasonable dt and periodic rebuilding of the Vm solver.
solver_params = SolverParams(
    save_frames=80,
    implicit_dt_multiplier=100.0,    # A reasonable value for stability and speed.
    rebuild_vm_solver_every=10000,     # Rebuild as the pore shape changes.
    n_tau_total=50.0               # Simulate for 8x the membrane charging time.
)

pore_growth_params = PhaseFieldParams(
    initial_state = 'pore',
    initial_pore_radius=30.08e-9,
    transition_thickness=0.5*dx,  
    sigma_area=5e-4,
    mobility=5.0e7,
    line_tension=1.5e-11
)

high_voltage = None
thermal_params = None

print("--- Starting Simulation: Voltage-Driven Pore Expansion ---")
simulate_membrane_charging(
    dom_in=custom_domain,
    solver=solver_params,
    phase=pore_growth_params,
    elec=high_voltage,
    thermal=thermal_params
)