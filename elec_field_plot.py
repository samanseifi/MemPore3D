"""
Generates publication-quality plots from the membrane charging simulation.

This script loads a .npz file and produces a separate, styled PDF file
for each key result, using LaTeX for text rendering.

Usage:
    python generate_publication_plots.py [path_to_npz_file]

If no path is provided, it defaults to 'membrane_charging_results.npz'.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def setup_matplotlib_for_latex():
    """Applies a consistent, publication-quality style to all plots."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
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

def plot_charging_curve(time_ns, avg_Vm_vs_time, filename="charging_curve.pdf"):
    """Plots and saves the average Vm charging curve."""
    print(f"  -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))  # Standard one-column width
    ax.plot(time_ns, avg_Vm_vs_time, marker=".", linestyle="-", color="#1b9e77", markersize=3)
    ax.set_xlabel(r"Time $t$ (ns)")
    ax.set_ylabel(r"Average $V_m$ (V)")
    ax.set_xlim(left=0)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_2d_map(data, x_nm, y_nm, cmap, cbar_label, filename):
    """Generic function to plot and save a 2D data map."""
    print(f"  -> Generating {filename}...")
    fig_width = 3.25
    aspect_ratio = data.shape[0] / data.shape[1]
    fig_height = fig_width * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap=cmap)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

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

def main(filepath: str):
    """Main function to load data and generate all plots."""
    # --- 1. Setup and Load Data ---
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at '{filepath}'")
        return

    setup_matplotlib_for_latex()
    print(f"✅ Loading data from '{filepath}'...")
    with np.load(filepath, allow_pickle=True) as data:
        x, y, z = data['x'], data['y'], data['z']
        Vm, phi, H = data['Vm'], data['phi_elec'], data['H']
        time_points, avg_Vm_vs_time = data['time_points'], data['avg_Vm_vs_time']

    # Scale coordinates and time for plotting
    x_nm, y_nm, z_nm = x * 1e6, y * 1e6, z * 1e6
    time_ns = time_points * 1e6

    # --- 2. Generate Each Plot ---
    print("✨ Generating plots...")

    # A) Average Vm Charging Curve
    plot_charging_curve(time_ns, avg_Vm_vs_time, filename="charging_curve_c.pdf")

    # B) Phase Field Map
    plot_2d_map(H, x_nm, y_nm, cmap="pink_r", cbar_label=r"Phase Field", filename="phase_field_map_c.pdf")

    # C) Final Vm Distribution
    plot_2d_map(Vm, x_nm, y_nm, cmap="magma", cbar_label=r"$V_m$ (V)", filename="vm_distribution_map_c.pdf")

    # D) Potential Slice with E-Field
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]

    Ex, Ey, Ez = np.gradient(-phi, dx, dy, dz)
    y_slice_idx = phi.shape[1] // 2
    phi_slice = phi[:, y_slice_idx, :]
    E_field_slice = (Ex[:, y_slice_idx, :], Ez[:, y_slice_idx, :])
    plot_potential_slice(phi_slice, E_field_slice, x_nm, z_nm, filename="potential_slice_c.pdf")

    print("\n✅ All plots generated successfully.")


npz_filepath = "membrane_simulation_results.npz"

main(npz_filepath)