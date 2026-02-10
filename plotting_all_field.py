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
        "text.usetex": True,
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

def process_simulation_data(results_dir="simulation_results", target_time=None):
    """
    Process simulation data from a results directory.

    Parameters
    ----------
    results_dir : str
        Directory containing step_*.npz files
    target_time : float, optional
        Target time in seconds to load state. If None, loads final state.
        The closest available time point will be selected.

    Returns
    -------
    history : tuple
        (times, radii, avg_vms) arrays
    state : dict
        State dictionary at the selected time point
    """
    file_pattern = os.path.join(results_dir, "step_*.npz")
    files = glob.glob(file_pattern)
    files.sort(key=natural_sort_key)

    if not files:
        print(f"No files found in '{results_dir}'")
        return None, None

    print(f"Found {len(files)} files. Processing data...")

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

    times_arr = np.array(times)

    # Select file based on target_time
    if target_time is not None:
        # Find closest time point
        idx = np.argmin(np.abs(times_arr - target_time))
        selected_file = files[idx]
        print(f"   -> Requested t={target_time*1e6:.2f} us, loading t={times_arr[idx]*1e6:.2f} us from: {os.path.basename(selected_file)}")
    else:
        selected_file = files[-1]
        print(f"   -> Loading final state from: {os.path.basename(selected_file)}")

    with np.load(selected_file, allow_pickle=True) as data:
        state = {
            'x': data['x'],
            'y': data['y'],
            'z': data['z'],
            'psi': data['psi'],
            'Vm': data['Vm'],
            'phi_elec': data['phi_elec'],
            'time': data['time']
        }

    return (times_arr, np.array(radii), np.array(avg_vms)), state

# --- 3. Plotting Functions ---

def plot_pore_growth(times, radii, filename="pore_radius_vs_time.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.plot(times*1e6, radii*1e9, marker=".", linestyle="-", color="#d95f02", markersize=3)
    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Effective Pore Radius $R_{pore}$ (nm)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_vm_history(times, vms, filename="vm_vs_time.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.plot(times*1e6, vms, marker=".", linestyle="-", color="#1b9e77", markersize=3)
    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Average $V_m$ (V)")
    ax.set_xlim(left=0)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_phase_map(state, filename="phase_contour.pdf", show_title=False, shift_x=0, shift_y=0):
    """
    Plot the phase field at a given state.

    Parameters
    ----------
    state : dict
        State dictionary containing 'x', 'y', 'psi', 'time'
    filename : str
        Output filename
    show_title : bool
        Whether to show time as title
    shift_x : float
        Shift in x direction as fraction of domain (0 to 1). Due to periodic BC.
    shift_y : float
        Shift in y direction as fraction of domain (0 to 1). Due to periodic BC.
    """
    print(f"   -> Generating {filename}...")
    x_um = state['x'] * 1e6
    y_um = state['y'] * 1e6
    psi = state['psi'].copy()
    time = float(state['time'])

    # Apply periodic shift if requested
    if shift_x != 0.0:
        shift_pts_x = int(shift_x * psi.shape[0])
        psi = np.roll(psi, shift_pts_x, axis=0)
    if shift_y != 0.0:
        shift_pts_y = int(shift_y * psi.shape[1])
        psi = np.roll(psi, shift_pts_y, axis=1)

    fig_width = 3.25
    aspect_ratio = psi.shape[0] / psi.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * aspect_ratio))

    im = ax.imshow(psi.T, origin="lower", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                   cmap="pink_r", vmin=0.0, vmax=1.0)
    ax.contour(x_um, y_um, psi.T, levels=[0.5], colors='gray', linewidths=0.5, linestyles='--')

    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    if show_title:
        ax.set_title(rf"$t = {time*1e6:.2f}\ \mu$s")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Phase Field $\phi$")
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# Keep backward compatibility
def plot_final_phase_map(final_state, filename="final_phase_contour.pdf"):
    """Deprecated: Use plot_phase_map instead."""
    plot_phase_map(final_state, filename, show_title=True)

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


# --- ADD THIS TO YOUR PLOTTING FUNCTIONS SECTION ---

def plot_vm_contour(x, y, vm_field, time, filename="final_vm_distribution.pdf", show_title=True,
                    shift_x=0.0, shift_y=0.0):
    """
    Plots the 2D distribution of the transmembrane potential Vm.

    Parameters
    ----------
    shift_x : float
        Shift in x direction as fraction of domain (0 to 1). Due to periodic BC.
    shift_y : float
        Shift in y direction as fraction of domain (0 to 1). Due to periodic BC.
    """
    print(f"   -> Generating {filename}...")
    x_um = x * 1e6
    y_um = y * 1e6
    vm = vm_field.copy()

    # Apply periodic shift if requested
    if shift_x != 0.0:
        shift_pts_x = int(shift_x * vm.shape[0])
        vm = np.roll(vm, shift_pts_x, axis=0)
    if shift_y != 0.0:
        shift_pts_y = int(shift_y * vm.shape[1])
        vm = np.roll(vm, shift_pts_y, axis=1)

    fig_width = 3.25
    aspect_ratio = vm.shape[0] / vm.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * aspect_ratio))

    # Use a diverging or sequential colormap to show potential variations
    im = ax.imshow(vm.T, origin="lower", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                   cmap="magma")

    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    if show_title:
        ax.set_title(rf"$t = {float(time)*1e6:.2f}\ \mu$s")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Potential Jump $V_m$ (V)")

    fig.tight_layout(pad=0.2)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_vm_and_phase_overlay(x, vm_field, psi_field, filename="vm_phase_overlay.pdf", shift_x=0.0):
    """
    Plots a 1D slice of both Vm and Phase Field at the center line (Y-mid).
    This visualizes how the potential jump aligns with the pore boundary.

    Parameters
    ----------
    shift_x : float
        Shift in x direction as fraction of domain (0 to 1). Due to periodic BC.
    """
    print(f"   -> Generating {filename}...")

    # Get the center slice along Y
    mid_y = vm_field.shape[1] // 2
    vm_line = vm_field[:, mid_y].copy()
    psi_line = psi_field[:, mid_y].copy()

    # Apply periodic shift if requested
    if shift_x != 0.0:
        shift_pts = int(shift_x * len(vm_line))
        vm_line = np.roll(vm_line, shift_pts)
        psi_line = np.roll(psi_line, shift_pts)

    x_um = x * 1e6

    fig, ax1 = plt.subplots(figsize=(3.5, 2.6))

    # Plot Phase Field (Left Axis)
    color_psi = '#1f77b4'  # blue
    ax1.plot(x_um, psi_line, color=color_psi, label=r"Phase $\phi$", linewidth=1.5)
    ax1.set_xlabel(r"Position $x$ ($\mu$m)")
    ax1.set_ylabel(r"Phase Field $\phi$", color=color_psi)
    ax1.tick_params(axis='y', labelcolor=color_psi)
    ax1.set_ylim(-0.05, 1.05)

    # Create twin axis for Vm (Right Axis)
    ax2 = ax1.twinx()
    color_vm = '#d62728'  # red
    ax2.plot(x_um, vm_line, '--', color=color_vm, label=r"$V_m$", linewidth=1.2)
    ax2.set_ylabel(r"Transmembrane Potential $V_m$ (V)", color=color_vm)
    ax2.tick_params(axis='y', labelcolor=color_vm)

    # Highlight the pore area where psi < 0.5
    ax1.fill_between(x_um, 0, 1, where=(psi_line < 0.5), 
                    color='gray', alpha=0.1, label="Pore Region")

    # ax1.set_title("Pore Interface Verification")
    
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    
def plot_vertical_jump_and_flux(phi, psi, z, sigma_e, filename="vertical_jump_continuity.pdf"):
    """
    Plots the potential Phi and Current Density Jz along the Z-axis.
    Overlaying these shows the potential jump vs flux continuity.
    """
    nx, ny, nz = phi.shape
    mx, my = nx // 2, ny // 2
    nz_mid = nz // 2
    
    # Extract center-line data
    z_um = z * 1e6
    phi_line = phi[mx, my, :]
    dz = z[1] - z[0]
    
    # Calculate Jz using 2nd-order center difference (away from jump)
    # and one-sided differences (at the jump)
    jz_line = np.zeros_like(phi_line)
    
    # Bulk Jz = -sigma * dPhi/dz
    # Note: Using the same logic as your current calculation
    kp = nz_mid + 1
    km = nz_mid - 1
    
    # Gradient calculation matching your solver's logic
    grad_p = (-3*phi[mx,my,kp] + 4*phi[mx,my,kp+1] - phi[mx,my,kp+2]) / (2*dz)
    grad_m = (3*phi[mx,my,km] - 4*phi[mx,my,km-1] + phi[mx,my,km-2]) / (2*dz)
    
    # Fill the Jz line for plotting
    # Simple central difference for the bulk
    jz_line[1:-1] = -sigma_e * (phi_line[2:] - phi_line[:-2]) / (2*dz)
    # Overwrite near jump with your high-accuracy one-sided gradients
    jz_line[kp] = -sigma_e * grad_p
    jz_line[km] = -sigma_e * grad_m
    # At the exact midplane, average the flux
    jz_line[nz_mid] = (jz_line[kp] + jz_line[km]) / 2.0

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

    # 1. Plot Potential (Phi) - Left Axis
    color_phi = '#1f77b4' # Blue
    # Plot top and bottom separately to show the jump
    ax1.plot(z_um[:nz_mid], phi_line[:nz_mid], 'o-', markersize=2, color=color_phi, label=r"Potential $\Phi$")
    ax1.plot(z_um[nz_mid+1:], phi_line[nz_mid+1:], 'o-', markersize=2, color=color_phi)
    
    ax1.set_xlabel(r"Vertical distance $z$ ($\mu$m)")
    ax1.set_ylabel(r"Potential $\Phi$ (V)", color=color_phi)
    ax1.tick_params(axis='y', labelcolor=color_phi)

    # 2. Plot Current Density (Jz) - Right Axis
    ax2 = ax1.twinx()
    color_j = '#d62728' # Red
    ax2.plot(z_um[1:-1], jz_line[1:-1] * 1e-3, '--', color=color_j, alpha=0.7, label=r"Flux $J_z$")
    ax2.set_ylabel(r"Current Density $J_z$ (mA/m$^2$)", color=color_j)
    ax2.tick_params(axis='y', labelcolor=color_j)

    # Set y-axis limits to be within 20% up and down of the average Jz
    avg_jz = np.mean(jz_line[1:-1] * 1e-3)
    ax2.set_ylim(avg_jz * 0.8, avg_jz * 1.2)
    
    # Vertical line at membrane
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # ax1.set_title("Interface Jump & Flux Continuity")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    setup_matplotlib_for_latex()
    results_path = "simulation_results"
    history, final_state = process_simulation_data(results_path, target_time=50e-6)

    if history is not None:
        times, radii, vms = history
        
        # 1. Standard History Plots
        plot_pore_growth(times, radii, filename="pore_radius_vs_time.pdf")
        plot_vm_history(times, vms, filename="vm_vs_time.pdf")
        plot_final_phase_map(final_state, filename="final_phase_contour.pdf")
        plot_vm_and_phase_overlay(final_state['x'], final_state['Vm'], final_state['psi'], filename="vm_phase_overlay.pdf")
        plot_vertical_jump_and_flux(final_state['phi_elec'], final_state['psi'], final_state['z'], sigma_e=1.0, filename="vertical_jump_continuity.pdf")
        
        # 2. NEW: Plot the 2D Vm field
        # We use the 'Vm' array stored in final_state
        if 'Vm' in final_state and final_state['Vm'] is not None:
            plot_vm_contour(
                final_state['x'], 
                final_state['y'], 
                final_state['Vm'], 
                final_state['time'], 
                filename="final_vm_distribution.pdf"
            )

        # 3. Potential and E-Field Slice (Existing Logic)
        phi = final_state['phi_elec']
        x, y, z = final_state['x'], final_state['y'], final_state['z']
        psi = final_state['psi']
        H = smooth_step(psi)

        if phi is not None and not np.all(phi == 0):
            ny_mid = phi.shape[1] // 2
            nz_mid = phi.shape[2] // 2
            dx, dz = x[1] - x[0], z[1] - z[0]
            
            # Extract 2D slice at middle Y
            phi_slice = phi[:, ny_mid, :]

            # Initialize Gradients
            Ex_slice = np.zeros_like(phi_slice)
            Ez_slice = np.zeros_like(phi_slice)

            # Horizontal Gradient (Ex) - Safe along X
            Ex_slice, _ = np.gradient(-phi_slice, dx, dz)

            # Vertical Gradient (Ez) - Split by the membrane to avoid the jump
            # Bottom Electrolyte
            _, Ez_slice[:, :nz_mid] = np.gradient(-phi_slice[:, :nz_mid], dx, dz)
            # Top Electrolyte
            _, Ez_slice[:, nz_mid+1:] = np.gradient(-phi_slice[:, nz_mid+1:], dx, dz)
            
            # Mask the interface index to prevent streamplot interpolation errors
            Ez_slice[:, nz_mid] = 0.0
            Ex_slice[:, nz_mid] = 0.0

            # Convert to microns for visualization
            x_plot, z_plot = x * 1e6, z * 1e6

            plot_potential_slice(
                phi_slice, 
                (Ex_slice, Ez_slice), 
                x_plot, 
                z_plot, 
                filename="electric_field_cross_section.pdf"
            )

        # 3. Stable Pore Current Calculation
        print("\nComputing stable pore current...")
        sigma_e = 1.0  # [S/m] - Adjust to your electrolyte conductivity
        dy = y[1] - y[0]
        
        # Define indices adjacent to the jump at z=0
        kp = nz_mid + 1
        km = nz_mid - 1
        
        # Calculate gradients just inside the bulk electrolyte regions
        # Using 2nd-order one-sided finite differences
        grad_p = (-3*phi[:,:,kp] + 4*phi[:,:,kp+1] - phi[:,:,kp+2]) / (2*dz)
        grad_m = (3*phi[:,:,km] - 4*phi[:,:,km-1] + phi[:,:,km-2]) / (2*dz)
        
        
        
        # Average the flux from both sides (Flux Continuity ensures they are similar)
        J_z = sigma_e * (-(grad_p + grad_m) / 2.0)
        
        # Integrate J_z over the pore area (where H is close to 0)
        pore_mask = 1.0 - H
        I_pore = np.sum(J_z * pore_mask) * (dx * dy)
        
        print(f"   -> Calculated Pore Current: {I_pore*1e9:.4f} nA")