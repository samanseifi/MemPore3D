import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re

from mempore3d.solvers.phase_field_solver import smooth_step

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

def plot_pore_growth_comparison(times_list, radii_list, labels, filename="pore_radius_vs_time_comparison.pdf"):
    print(f"   -> Generating {filename}...")
    fig, ax = plt.subplots(figsize=(4.0, 3.0)) # Slightly wider for legend
    
    # Loop through each dataset
    for times, radii, label in zip(times_list, radii_list, labels):
        ax.plot(times*1e6, radii*1e9, linestyle="-", markersize=3, label=label)
    
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
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    for times, vms, label in zip(times_list, vms_list, labels):
        ax.plot(times*1e6, vms, linestyle="-", markersize=3, label=label)
    
    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Average $V_m$ (V)")
    ax.set_xlim(left=0)
    ax.legend(fontsize='small', frameon=False)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    setup_matplotlib_for_latex()
    
    # Define your directories and the labels you want in the legend
    directories = [
        "simulation_results_128x128x129_worked_3_v_125",
        "simulation_results_128x128x129_worked_2_v_150",
        "simulation_results_128x128x129_v150_320points",
        "simulation_results_128x128x129_v150_M5e6"
    ]
    custom_legends = ["V=1.25 v, M=1e7", "V=1.50 v, M=1e7", "V=1.5 v, M=1e7", "V=1.5 v, M=5e6"]
    
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