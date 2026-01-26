import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
import glob
import os
import re

from mempore3d.solvers.phase_field_solver import smooth_step


def smooth_data(data, window_length=11, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing."""
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    if window_length < polyorder + 2:
        return data
    return savgol_filter(data, window_length, polyorder)

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

def plot_pore_growth_comparison(times_list, radii_list, labels, filename="pore_radius_vs_time_comparison.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))

    # Professional color palette
    colors = plt.cm.Dark2.colors

    for i, (times, radii, label) in enumerate(zip(times_list, radii_list, labels)):
        t_plot = times * 1e6
        r_plot = radii * 1e9
        r_smooth = smooth_data(r_plot)
        color = colors[i % len(colors)]

        # Plot raw data with markers only (unfilled, matching color)
        ax.plot(t_plot, r_plot, linestyle='', marker='.', markersize=3, color=color)
        # Plot smoothed curve
        ax.plot(t_plot, r_smooth, linestyle="-", color=color, label=label)

    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Effective Pore Radius $R_{pore}$ (nm)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize='small', frameon=False)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_vm_history_comparison(times_list, vms_list, labels, filename="vm_vs_time_comparison.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(3.25, 2.4))

    # Professional color palette
    colors = plt.cm.Dark2.colors

    for i, (times, vms, label) in enumerate(zip(times_list, vms_list, labels)):
        t_plot = times * 1e6
        vm_smooth = smooth_data(vms)
        color = colors[i % len(colors)]

        # Plot raw data with markers only (unfilled, matching color)
        ax.plot(t_plot[::2], vms[::2], linestyle='', marker='.', markersize=3, color=color)
        # Plot smoothed curve
        ax.plot(t_plot, vm_smooth, linestyle="-", color=color, label=label)

    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Average $\overline{V}_m$ (V)")
    ax.set_xlim(left=0)
    ax.legend(fontsize='small', frameon=False)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    setup_matplotlib_for_latex()
    
    # Define your directories and the labels you want in the legend
    # directories = [
    #     "case_4",
    #     "case_9",
    #     "case_6"
    # ]
    # custom_legends = [
    #     r"$\sigma=0$",
    #     r"$\sigma=5\times 10^{-5}~\mathrm{J}\mathrm{m}^{-2}$",
    #     r"$\sigma=5\times 10^{-4}~\mathrm{J}{m}^{-2}$"    ]
    
    # directories = [        
    #     "case_1",
    #     "simulation_results_128x128x129_v150_M5e6",
    #     "case_10"
    # ]
    # custom_legends = [
    #     r"$M=5\times 10^{7}~\mathrm{m}^{-2}\mathrm{J}^{-1}\mathrm{s}^{-1}$",
    #     r"$M=5\times 10^{6}~\mathrm{m}^{-2}\mathrm{J}^{-1}\mathrm{s}^{-1}$",        
    #     r"$M=5\times 10^{5}~\mathrm{m}^{-2}\mathrm{J}^{-1}\mathrm{s}^{-1}$" 
    # ]
    
    # directories = [        
    #     "case_2",
    #     "case_4",
    #     "case_5"
    # ]
    # custom_legends = [
    #     r"$V_\mathrm{applied}=1.5~\mathrm{V}$",
    #     r"$V_\mathrm{applied}=1.25~\mathrm{V}$",        
    #     r"$V_\mathrm{applied}=1.0~\mathrm{V}$" 
    # ]
    
    directories = [        
        "case_11",
        "case_12"
    ]
    custom_legends = [
        r"$V_\mathrm{applied}=1.5~\mathrm{V}$",
        r"$V_\mathrm{applied}=1.25~\mathrm{V}$" 
    ]
    
    
    # Containers for aggregated data
    all_times = []
    all_radii = []
    all_vms = []
    active_labels = []

    for path, label in zip(directories, custom_legends):
        history, final_state = process_simulation_data(path)
        
        if history is not None:
            times, radii, vms = history
            all_times.append(times)
            all_radii.append(radii)
            all_vms.append(vms)
            active_labels.append(label)

    # 1. Generate Overlaid Plots
    if all_times:
        plot_pore_growth_comparison(all_times, all_radii, active_labels)
        plot_vm_history_comparison(all_times, all_vms, active_labels)