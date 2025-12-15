from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------------------------------------------------------
# Visualization and Main Driver
# -----------------------------------------------------------------------------
def plot_results(x, y, z, Vm, phi_elec, psi, time_points, avg_Vm_vs_time, pore_radius_vs_time, title_prefix, elapsed_time):
    """Generates and displays plots of the simulation results."""
    fig = plt.figure(figsize=(24, 6))
    x_nm, y_nm = x * 1e9, y * 1e9
    time_ns = np.array(time_points) * 1e9

    fig.suptitle(f"{title_prefix}\n(Total Wall Clock Time: {elapsed_time:.2f} s)", fontsize=16, y=1.05)

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(time_ns, avg_Vm_vs_time, marker=".", linestyle="-", label="Avg Vm")
    ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Average Vm (V)")
    ax1.set_title("Membrane Charging"); ax1.grid(True, linestyle='--')
    
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(time_ns, np.array(pore_radius_vs_time) * 1e9, marker='.', linestyle='-', color='crimson')
    ax2.set_xlabel("Time (ns)"); ax2.set_ylabel("Effective Pore Radius (nm)")
    ax2.set_title("Pore Dynamics"); ax2.grid(True, linestyle='--')
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(Vm.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="magma")
    ax3.set_xlabel("x (nm)"); ax3.set_ylabel("y (nm)"); ax3.set_title("Final $V_m$ Distribution")
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Voltage (V)")

    ax4 = fig.add_subplot(1, 4, 4)
    im4 = ax4.imshow(psi.T, origin="lower", extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]], cmap="jet", vmin=0, vmax=1)
    ax4.set_xlabel("x (nm)"); ax4.set_ylabel("y (nm)"); ax4.set_title("Final Pore Shape ($\\psi$)")
    cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im4, cax=cax4, label="Phase Field")

    plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()