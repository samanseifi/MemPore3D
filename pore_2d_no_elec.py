import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Grid (using physical units)
# -----------------------------------------------------------------------------
Nx = Ny = 64
Lx = Ly = 1000.0e-9  # m (1000 nm, consistent with first script)
dx = Lx / Nx
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# -----------------------------------------------------------------------------
# Physical and Numerical Parameters
# -----------------------------------------------------------------------------
# --- Physical Parameters ---
gamma = 1.5e-11      # J/m, typical line tension for lipid membrane pore
R0 = 29.7e-9           # m, initial pore radius (20 nm)
sigma = 5e-4         # J/m^2, area-dependent energy (set to 0 for pure line tension model)
epsilon = 0.5 * dx     # m, interface thickness, ~2 grid cells

# --- CORRECTED PARAMETER ---
# The mobility M is a kinetic parameter that sets the timescale of closure.
# It has been increased to ensure the dynamics are visible in the chosen simulation time.
M = 50000.0            # m^2/(J*s), phase-field mobility

# --- Numerical Parameters ---
total_time = 0.5   # s (simulate for 50 nanoseconds)
nsteps = 10000
dt = total_time / nsteps
save_every = 100     # Save data every 250 steps

# -----------------------------------------------------------------------------
# Initial Condition and Model Functions
# -----------------------------------------------------------------------------
# Initial pore (phi=0 inside, 1 outside)
xc, yc = Lx / 2, Ly / 2
r = np.sqrt((X - xc)**2 + (Y - yc)**2)
phi = 0.5 * (1.0 + np.tanh((r - R0) / (np.sqrt(2) * epsilon)))

# Double-well potential and its derivative
def g(phi): return 0.25 * phi**2 * (1.0 - phi)**2
def gp(phi): return 0.5 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

# Smooth Heaviside function H(phi) and its derivative dH(phi) for the sigma term
def H(phi):
    pc = np.clip(phi, 0.0, 1.0)
    return pc**2 * (3.0 - 2.0 * pc)
def dH(phi):
    pc = np.clip(phi, 0.0, 1.0)
    return 6.0 * pc * (1.0 - pc)

Cg = np.sqrt(2.0) / 12.0

# -----------------------------------------------------------------------------
# Spectral (FFT) Setup
# -----------------------------------------------------------------------------
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
# Denominator for semi-implicit time stepping of the Laplacian term
denom = 1.0 + dt * M * (gamma / Cg) * epsilon * K2

# -----------------------------------------------------------------------------
# Energy Calculation
# -----------------------------------------------------------------------------
def energy(phi):
    gradx = np.fft.ifft2(1j * KX * np.fft.fft2(phi)).real
    grady = np.fft.ifft2(1j * KY * np.fft.fft2(phi)).real
    # Interfacial energy (from line tension)
    Ein = (gamma / Cg) * np.sum((epsilon * 0.5 * (gradx**2 + grady**2) + g(phi) / epsilon)) * dx * dx
    # Area-dependent energy
    Earea = sigma * np.sum(H(phi)) * dx * dx
    return Ein + Earea

# -----------------------------------------------------------------------------
# Main Simulation Loop
# -----------------------------------------------------------------------------
times, radii, energies = [], [], []
print("Starting pore closure simulation...")

for n in range(nsteps + 1):
    # --- Data Saving ---
    if n % save_every == 0:
        Hphi = H(phi)
        Apore = np.sum(1.0 - Hphi) * dx * dx
        R_eff = np.sqrt(max(Apore, 0.0) / np.pi)
        
        current_time_ns = n * dt * 1e9
        current_radius_nm = R_eff * 1e9
        current_energy = energy(phi)

        times.append(current_time_ns)
        radii.append(current_radius_nm)
        energies.append(current_energy)
        
        print(f"Time: {current_time_ns:5.2f} ns, Radius: {current_radius_nm:5.2f} nm")

    # --- Time Stepping ---
    if n < nsteps:
        # Explicit part of the equation (reaction terms)
        rhs = -M * ((gamma / Cg) * (gp(phi) / epsilon) + sigma * dH(phi))
        
        # Advance one step in Fourier space
        phi_hat = np.fft.fft2(phi)
        rhs_hat = np.fft.fft2(rhs)
        phi_hat_new = (phi_hat + dt * rhs_hat) / denom
        
        # Transform back to real space and clip for stability
        phi = np.fft.ifft2(phi_hat_new).real
        phi = np.clip(phi, 0.0, 1.0)

print("Simulation finished.")

# -----------------------------------------------------------------------------
# Save Outputs and Plots
# -----------------------------------------------------------------------------
# Save diagnostics to a CSV file
pd.DataFrame({
    'time_ns': times,
    'radius_nm': radii,
    'energy_J': energies
}).to_csv("pore_closure_diagnostics_physical.csv", index=False)

# Plot 1: Pore radius vs. time
plt.figure(figsize=(6, 4))
plt.plot(times, radii, marker='.')
plt.xlabel("Time (ns)")
plt.ylabel("Effective Radius (nm)")
plt.title("Pore Closure Driven by Line Tension")
plt.grid(True)
plt.tight_layout()
plt.savefig("pore_radius_vs_time_physical.png", dpi=160)

# Plot 2: Energy vs. time
plt.figure(figsize=(6, 4))
plt.plot(times, energies)
plt.xlabel("Time (ns)")
plt.ylabel("Total Energy (J)")
plt.title("System Energy vs. Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_vs_time_physical.png", dpi=160)

# Plot 3: Final phase-field configuration
plt.figure(figsize=(6, 5))
plt.imshow(phi.T, origin="lower", extent=[0, Lx*1e9, 0, Ly*1e9])
cbar = plt.colorbar()
cbar.set_label("Phase Field (phi)")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.title(f"Final Pore State at {total_time*1e9:.1f} ns")
plt.tight_layout()
plt.savefig("final_phi_physical.png", dpi=160)

plt.show()