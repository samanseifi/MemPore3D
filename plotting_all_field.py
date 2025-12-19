import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re

from mempore3d.solvers.phase_field_solver import smooth_step

# --- 1. Style Setup ---
def setup_matplotlib_for_latex():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "figure.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "xtick.top": True,
        "ytick.right": True,
    })

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# --- 2. Data Processing ---
def calculate_pore_stats(psi, x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    pore_area = np.sum(psi < 0.5) * dx * dy
    if pore_area > 0:
        return np.sqrt(pore_area / np.pi)
    return 0.0

def process_simulation_data(results_dir="simulation_results"):
    file_pattern = os.path.join(results_dir, "step_*.npz")
    files = glob.glob(file_pattern)
    files.sort(key=natural_sort_key)
    
    if not files:
        print(f"❌ No files found in '{results_dir}'")
        return None, None

    print(f"✅ Found {len(files)} files. Processing data...")

    times = []
    radii = []
    avg_vms = []

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            t = data['time']
            psi = data['psi']
            x = data['x']
            y = data['y']
            
            if 'avg_Vm' in data:
                vm_val = data['avg_Vm']
            elif 'Vm' in data:
                vm_val = np.mean(data['Vm'])
            else:
                vm_val = 0.0
            
            r = calculate_pore_stats(psi, x, y)
            times.append(t)
            radii.append(r)
            avg_vms.append(vm_val)

    print(f"   -> Loading final 3D state from: {os.path.basename(files[-1])}")
    with np.load(files[-1], allow_pickle=True) as data:
        final_state = {
            'x': data['x'],
            'y': data['y'],
            'z': data['z'],
            'psi': data['psi'],
            'Vm': data['Vm'],
            'phi_elec': data['phi_elec'],
            'time': data['time']
        }
            
    return (np.array(times), np.array(radii), np.array(avg_vms)), final_state

# --- 3. Plotting Functions ---

def plot_pore_growth(times, radii, filename="pore_radius_vs_time.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.plot(times*1e9, radii*1e9, marker=".", linestyle="-", color="#d95f02", markersize=3)
    ax.set_xlabel(r"Time $t$ (ns)")
    ax.set_ylabel(r"Pore Radius $R_{pore}$ (nm)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_vm_history(times, vms, filename="vm_vs_time.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.plot(times*1e9, vms, marker=".", linestyle="-", color="#1b9e77", markersize=3)
    ax.set_xlabel(r"Time $t$ (ns)")
    ax.set_ylabel(r"Average $V_m$ (V)")
    ax.set_xlim(left=0)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_final_phase_map(final_state, filename="final_phase_contour.pdf"):
    print(f"   -> Generating {filename}...")
    x_nm = final_state['x'] * 1e9
    y_nm = final_state['y'] * 1e9
    psi = final_state['psi']
    
    fig_width = 3.25
    aspect_ratio = psi.shape[0] / psi.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * aspect_ratio))

    im = ax.imshow(psi.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], 
                   cmap="pink_r", vmin=0.0, vmax=1.0)
    ax.contour(x_nm, y_nm, psi.T, levels=[0.5], colors='gray', linewidths=0.5, linestyles='--')

    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
    ax.set_title(rf"$t = {final_state['time']*1e9:.1f}$ ns")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Phase Field $\phi$")
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

# --- YOUR REQUESTED FUNCTION ---
def plot_potential_slice(phi_slice, E_field, x_nm, z_nm, filename="potential_slice.pdf"):
    """Plots and saves the potential slice with E-field streamlines."""
    print(f"  -> Generating {filename}...")
    Ex_slice, Ez_slice = E_field
    fig_width = 3.25
    aspect_ratio = phi_slice.shape[1] / phi_slice.shape[0] # z/x
    fig_height = fig_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(phi_slice.T, origin="lower", extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]], aspect="auto", cmap="viridis")
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.0, label=r"Membrane")

    # Plot streamlines
    ax.streamplot(x_nm, z_nm, Ex_slice.T, Ez_slice.T, color=(1, 1, 1, 0.7), density=1.2, linewidth=0.7, arrowsize=0.7)

    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$z$ ($\mu$m)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(x_nm[0], x_nm[-1])
    ax.set_ylim(z_nm[0], z_nm[-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Potential $\Phi$ (V)")
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

# --- 4. Main Execution ---
if __name__ == "__main__":
    setup_matplotlib_for_latex()
    
    results_path = "simulation_results"
    
    history, final_state = process_simulation_data(results_path)

    if history is not None:
        times, radii, vms = history
        
        # 1. Plots
        plot_pore_growth(times, radii, filename="pore_radius_vs_time.pdf")
        plot_vm_history(times, vms, filename="vm_vs_time.pdf")
        plot_final_phase_map(final_state, filename="final_phase_contour.pdf")

        # 2. Prepare Data for Potential Slice
        # We need to slice the 3D data manually before calling your function
        phi = final_state['phi_elec']
        x = final_state['x']
        y = final_state['y']
        z = final_state['z']
        H = smooth_step(final_state['psi'])

        if phi is not None and not np.all(phi == 0):
            # A) Take slice at middle Y


            # B) Calculate Gradients (E-field)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            Ex, Ey, Ez = np.gradient(-phi, dx, dy, dz)
            ny_mid = phi.shape[1] // 2
            phi_slice = phi[:, ny_mid, :]
            E_field_slice = (Ex[:, ny_mid, :], Ez[:, ny_mid, :])


            # C) Prepare Coordinates (X in nm, Z in microns)
            # This matches the labels: x (nm) and z (µm)
            x_plot = x * 1e6
            z_plot = z * 1e6

            # D) Call YOUR function
            plot_potential_slice(
                phi_slice, 
                E_field_slice, 
                x_plot, 
                z_plot, 
                filename="electric_field_cross_section.pdf"
            )
        
        print("\n✅ All plots generated successfully.")
        
        # --- 3. Compute Pore Current (Corrected Version) ---
        print("Computing pore current...")
        sigma_eff = 1.0  # S/m, replace with actual conductivity if available
        
        # A) Get grid spacings.
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        
        # B) Find the z-index of the membrane's central plane.
        k_m_idx = np.argmin(np.abs(z))
        
        # C) Calculate the z-component of the electric field (E_z = -dΦ/dz)
        delta_phi_z = phi[:, :, k_m_idx + 1] - phi[:, :, k_m_idx - 1]
        E_z = -delta_phi_z / (2.0 * dz)
        
        # D) Calculate the current density J_z = σ * E_z
        J_z = sigma_eff * E_z
        
        # E) Use the smooth phase field H for a weighted integral.
        pore_weight = 1.0 - H
        
        # F) Apply the weight to get the current density contribution from the pore.
        J_pore_weighted = J_z * pore_weight
        
        # G) Integrate over the entire xy-plane area to get the total current I_pore.
        dA = dx * dy
        I_pore = np.sum(J_pore_weighted) * dA
        
        # Convert to nanoamperes (nA) for a more readable output
        I_pore_nA = I_pore * 1e9 
        print(f"   -> Computed Pore Current (I_pore): {I_pore_nA:.4f} nA")

        print("\n✅ All plots generated and calculations completed successfully.")