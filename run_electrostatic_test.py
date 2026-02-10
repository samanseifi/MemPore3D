from mempore3d.parameters import *
from mempore3d.core import *

Lx, Ly, Lz = 10000e-9, 10000e-9, 20000e-9
Nx, Ny, Nz = 128, 128, 129 # Nz should be odd to center membrane
custom_domain = Domain(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz)

# Calculating dx for transition thickness    
x = np.linspace(-Lx / 2, Lx / 2, Nx)
dx = x[1] - x[0]

# We need a reasonable dt and periodic rebuilding of the Vm solver.
solver_params = SolverParams(
    save_frames=80,
    implicit_dt_multiplier=20.0,    # A reasonable value for stability and speed.
    rebuild_vm_solver_every=25,     # Rebuild as the pore shape changes.
    n_tau_total=10.0,              # Simulate for 8x the membrane charging time.
    poisson_solver='spectral'
)

pore_growth_params = PhaseFieldParams(
    initial_state = 'pore',
    initial_pore_radius=1000e-9,
    transition_thickness=1*dx,  
    sigma_area=0,
    mobility=5.0e6,
    line_tension=1.5e-11,
    evolve='off'
)

high_voltage = Electrostatics(V_applied=5.0)
thermal_params = None

print("--- Starting Simulation: Voltage-Driven Pore Expansion ---")
simulate_membrane_charging(
    dom_in=custom_domain,
    solver=solver_params,
    phase=pore_growth_params,
    elec=high_voltage,
    thermal=thermal_params
)


# transition thickness = 1
# 32x32x33 I_pore, Vm = -1727.7708, 3.4799


# transition thickness = 0.5
# 32x32x33 I_pore, Vm = -1493.3698, 3.6861

# transition thickness = 2
# 32x32x33 I_pore, Vm = -2252.1928, 3.0185

# 64x64x33 I_pore, Vm = -1708.6704, 3.4478
# 64x64x65 I_pore, Vm = -1955.7035, 3.1642
# 64x64x129 I_pore, Vm = -2133.9233, 2.9645
# 64x64x257 I_pore, Vm = -2277.1915, 2.8107
# 64x64x513 I_pore, Vm = -2402.6361, 2.6810
# 64x64x1025 I_pore, Vm = -2517.8530, 2.5650


# 128x128x33 I_pore, Vm = -1479.6665, 3.6344
# 128x128x65 I_pore, Vm = -1693.5567, 3.3849
# 128x128x129 I_pore, Vm = -1830.1423, 3.2265
# 128x128x257 I_pore, Vm = -1926.3027, 3.1185
# 128x128x513 I_pore, Vm = -2002.9763, 3.0359

# Second Order J in Vm solver
# 32x32x33 I_pore, Vm = -1727.7708, 3.4799

# 64x64x33 I_pore, Vm = -1485.5410, 3.6505
# 64x64x65 I_pore, Vm = -1701.2502, 3.4030
# 64x64x129 I_pore, Vm = -1839.2730, 3.2456
# 64x64x257 I_pore, Vm = -1936.6703, 3.1380
# 64x64x513 I_pore, Vm =  -2014.5492, 3.0555


# 128x128x33 I_pore, Vm = -1382.3759, 3.7242
# 128x128x65 I_pore, Vm =  -1585.3919, 3.4881
# 128x128x129 I_pore, Vm = -1705.4440, 3.3473
# 128x128x257 I_pore, Vm = -1779.5703, 3.2618
# 128x128x513 I_pore, Vm = -1830.9789, 3.2046


# 256x256x33 I_pore, Vm = -1335.7143, 3.7575
# 256x256x257 I_pore, Vm = -1714.0549, 3.3126


## Second Order J in Vm solver, alpha =20
# 32x32x33 I_pore, Vm = -432.2889, 3.4292
# 64x64x65 I_pore, Vm =  -475.2013 ,3.2157
# 64x64x129 I_pore, Vm =  -501.4933 ,3.0866
# 64x64x257 I_pore, Vm =  -520.5164 ,2.9983
# 64x64x513 I_pore, Vm =  -536.5552  2.9284


## Second Order J in Vm solver, alpha =50
# 64x64x65 I_pore, Vm = -200.0414  3.1222

## Second Order J in Vm solver, alpha =500
# 64x64x65 I_pore, Vm = -22.1048  2.9250

## Second Order J in Vm solver, alpha =5000
# 64x64x65 I_pore, Vm = -2.3982  2.7488

# 64x64x33 I_pore =  nA, Vm = 3.1009 V 
# 64x64x65 I_pore = nA, Vm = 2.9629 V
# 64x64x129 I_pore = -70.2926 nA, Vm = 2.9047 V
# 64x64x257 I_pore = -141.0153 nA, Vm = 2.8816 V

# fixed sovlver, no G_pore with fixed G_pore formulation
# 64x64x33 I_pore = -3214.7480 nA, Vm = 3.9331 V 
# 64x64x65 I_pore = -4177.0990 nA, Vm = 3.4084 V
# 64x64x129 I_pore =  -4351.2880 nA, Vm = 3.1292 V
# 64x64x257 I_pore =  nA, Vm = 2.9936 V    
# 
# Final solver
# 64x64x33 -3999.3342 nA,Vm = 2.8095 V
# 64x64x65 -4083.2573 nA, Vm = 2.8095 V
# 64x64x129 -4038.2048 nA, Vm = 2.8095 V
# 64x64x257 -3998.9308 nA, Vm = 2.8095 V
# 64x64x513 -3982.2131 nA, Vm = 2.8095 V


# 128x128x65 -5180.254 nA, Vm = 3.0606 V
# 128x128x129 -5236.5649 nA, Vm = 3.0606 V
# 128x128x257 -5200.9609 nA, Vm = 3.0606 V
# 128x128x513 -5170.1680 nA, Vm = 3.0606 V

# 256x256x65 -5927.8187 nA, Vm = 3.1769 V


# 128x128x65 -5564.9765 nA, Vm = 3.1822 V
# 128x128x129 -5397.8764 nA, Vm = 3.1102 V
# 128x128x257 -5258.9070 nA, Vm = 3.0802 V